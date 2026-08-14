"""S8：Findings 画框、JSON 与 Markdown 报告。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pipeline import dump_json, load_config, resolve_path


def _font(size: int = 18):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _union_bbox(boxes: list[list[float]]) -> list[int]:
    x1 = min(int(b[0]) for b in boxes)
    y1 = min(int(b[1]) for b in boxes)
    x2 = max(int(b[2]) for b in boxes)
    y2 = max(int(b[3]) for b in boxes)
    return [x1, y1, x2, y2]


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _is_text_overlap_rule(rule_id: str) -> bool:
    rid = (rule_id or "").upper()
    return "OVERLAP" in rid or "TEXT_NO_OVERLAP" in rid


def _merge_overlap_twins(
    boxes: list[list[float]],
    *,
    facts: dict[str, Any] | None = None,
    related_ids: list[str] | None = None,
) -> list[list[int]]:
    """文本重叠：一对 twin 框合并为外接矩形。优先按 parent_id，否则按 IoU 配对。"""
    if not boxes:
        return []

    if facts and related_ids:
        by_id: dict[str, dict[str, Any]] = {}
        for a in facts.get("annotations") or []:
            if not isinstance(a, dict):
                continue
            for key in (a.get("instance_id"), a.get("id")):
                if key:
                    by_id[str(key)] = a
        groups: dict[str, list[list[float]]] = {}
        for rid in related_ids:
            ann = by_id.get(str(rid))
            if not ann or not ann.get("bbox"):
                continue
            pid = str(ann.get("parent_id") or rid)
            base = pid[: -len("_twin")] if pid.endswith("_twin") else pid
            groups.setdefault(base, []).append(list(ann["bbox"]))
        if groups:
            return [_union_bbox(g) for g in groups.values() if g]

    # 贪心：每个框与 IoU 最高且 >0 的另一框合并
    pending = [list(map(float, b)) for b in boxes if len(b) == 4]
    merged: list[list[int]] = []
    used = [False] * len(pending)
    for i, bi in enumerate(pending):
        if used[i]:
            continue
        best_j, best_iou = -1, 0.0
        for j in range(i + 1, len(pending)):
            if used[j]:
                continue
            iou = _bbox_iou(bi, pending[j])
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= 0.05:
            merged.append(_union_bbox([bi, pending[best_j]]))
            used[i] = used[best_j] = True
        else:
            merged.append(_union_bbox([bi]))
            used[i] = True
    return merged


def _display_boxes(
    finding: dict[str, Any],
    *,
    facts: dict[str, Any] | None = None,
) -> list[list[int]]:
    boxes = finding.get("evidence_bboxes") or []
    valid = [list(b) for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if _is_text_overlap_rule(str(finding.get("rule_id") or "")):
        return _merge_overlap_twins(
            valid,
            facts=facts,
            related_ids=list(finding.get("related_ids") or []),
        )
    return [_union_bbox([b]) for b in valid]


def _legend_lines(finding: dict[str, Any], box_count: int) -> list[str]:
    sev = finding.get("severity", "error")
    rule_id = finding.get("rule_id") or ""
    msg = finding.get("message") or ""
    actual = finding.get("actual")
    count = box_count
    if isinstance(actual, (int, float)) and int(actual) > 0:
        count = int(actual)
    elif isinstance(actual, dict) and actual.get("overlap_pairs") is not None:
        count = int(actual["overlap_pairs"])
    head = f"[{sev}] {rule_id}"
    if count > 1:
        detail = f"{msg}（{count}处）"
    else:
        detail = msg
    return [head, detail]


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    img_w: int,
    img_h: int,
    entries: list[tuple[tuple[int, int, int], list[str]]],
    *,
    font: ImageFont.ImageFont,
) -> None:
    if not entries:
        return
    title = "错误图例"
    pad = 10
    line_gap = 4
    block_gap = 8
    swatch = 12

    lines_flat: list[tuple[tuple[int, int, int] | None, str]] = [(None, title)]
    for color, texts in entries:
        for i, t in enumerate(texts):
            lines_flat.append((color if i == 0 else None, t))

    max_tw = 0
    total_h = pad * 2
    sizes: list[tuple[int, int]] = []
    for i, (_, text) in enumerate(lines_flat):
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        sizes.append((tw, th))
        max_tw = max(max_tw, tw + (swatch + 8 if i > 0 else 0))
        total_h += th + (block_gap if i == 0 else line_gap)
    total_h += 4
    panel_w = min(img_w - 16, max_tw + pad * 2 + 8)
    panel_h = min(img_h - 16, total_h)

    x0, y0 = 8, 8
    # 半透明白底：用实心浅底保证工程图上可读
    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill=(255, 255, 255), outline=(40, 40, 40), width=2)

    y = y0 + pad
    for i, (color, text) in enumerate(lines_flat):
        tw, th = sizes[i]
        x = x0 + pad
        if i == 0:
            draw.text((x, y), text, fill=(20, 20, 20), font=font)
        else:
            if color is not None:
                cy = y + max(0, (th - swatch) // 2)
                draw.rectangle([x, cy, x + swatch, cy + swatch], fill=color, outline=color)
                x += swatch + 8
            draw.text((x, y), text, fill=(30, 30, 30), font=font)
        y += th + (block_gap if i == 0 else line_gap)


def _table_rows_from_facts(facts: dict[str, Any] | None) -> list[dict[str, Any]]:
    """从 facts.tables / _instances 收集带 bbox 的表格行。"""
    if not facts:
        return []
    rows: list[dict[str, Any]] = []
    for t in facts.get("tables") or []:
        if isinstance(t, dict) and t.get("bbox"):
            rows.append(t)
    if rows:
        return rows
    for inst in facts.get("_instances") or []:
        if not isinstance(inst, dict) or not inst.get("bbox"):
            continue
        eid = str(inst.get("entity_id") or "")
        if eid.endswith("_table") or eid in {"info_table", "material_table", "main_table"}:
            fields = inst.get("fields") if isinstance(inst.get("fields"), dict) else {}
            rows.append(
                {
                    "bbox": inst.get("bbox"),
                    "section": (fields or {}).get("section"),
                    "label": inst.get("label") or eid,
                    "instance_id": inst.get("instance_id"),
                    "entity_id": eid,
                }
            )
    return rows


def _table_box_style(
    row: dict[str, Any],
    colors: dict[str, Any],
) -> tuple[tuple[int, int, int], str]:
    section = str(row.get("section") or "").lower()
    eid = str(row.get("entity_id") or row.get("label") or "").lower()
    if section == "material" or "material" in eid:
        color = tuple(colors.get("table_material", [16, 185, 129]))
        name = "material_table"
    elif section == "main" or "main_table" in eid:
        color = tuple(colors.get("table_main", [59, 130, 246]))
        name = "main_table"
    else:
        color = tuple(colors.get("table_other", [168, 85, 247]))
        name = str(row.get("label") or row.get("instance_id") or "table")
    return color, name  # type: ignore[return-value]


def _draw_table_boxes(
    draw: ImageDraw.ImageDraw,
    facts: dict[str, Any] | None,
    *,
    colors: dict[str, Any],
    width: int,
    show_label: bool,
    font: ImageFont.ImageFont,
) -> list[tuple[tuple[int, int, int], list[str]]]:
    """绘制表格识别框，并返回可并入图例的条目。"""
    legend: list[tuple[tuple[int, int, int], list[str]]] = []
    seen: set[str] = set()
    for row in _table_rows_from_facts(facts):
        bbox = row.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        color, name = _table_box_style(row, colors)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if show_label:
            label = f"{name}"
            # 标签画在框内左上，浅底保证可读
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lx, ly = x1 + 2, max(0, y1 + 2)
            draw.rectangle([lx, ly, lx + tw + 4, ly + th + 2], fill=(255, 255, 255))
            draw.text((lx + 2, ly), label, fill=color, font=font)
        if name not in seen:
            seen.add(name)
            legend.append((color, [f"[table] {name}", "表格识别框"]))
    return legend


def _attribute_rows_from_facts(facts: dict[str, Any] | None) -> list[dict[str, Any]]:
    """收集带 bbox 的尺寸/数字属性标注（annotations 或 number_mark 实例）。"""
    if not facts:
        return []
    rows: list[dict[str, Any]] = []
    for a in facts.get("annotations") or []:
        if isinstance(a, dict) and a.get("bbox"):
            rows.append(a)
    if rows:
        return rows
    for inst in facts.get("_instances") or []:
        if not isinstance(inst, dict) or not inst.get("bbox"):
            continue
        eid = str(inst.get("entity_id") or "")
        if eid not in {"number_mark", "annotation", "annotations"}:
            continue
        fields = inst.get("fields") if isinstance(inst.get("fields"), dict) else {}
        rows.append(
            {
                **(fields or {}),
                "bbox": inst.get("bbox"),
                "quad": inst.get("quad"),
                "raw_text": inst.get("raw_text"),
                "instance_id": inst.get("instance_id"),
                "label": inst.get("label"),
                "keep_pair": inst.get("keep_pair"),
                "entity_id": eid,
            }
        )
    return rows


def _format_attribute_label(row: dict[str, Any], *, max_text: int = 28) -> str:
    """属性可视化短标签：文本 | 类型 | 基本尺寸 | 公差 | 角。"""
    parts: list[str] = []
    text = str(row.get("text") or row.get("raw_text") or "").strip()
    if text:
        parts.append(text if len(text) <= max_text else text[: max_text - 1] + "…")
    kind = row.get("dim_kind")
    if kind:
        parts.append(str(kind))
    basic = row.get("basic_size")
    if basic not in (None, ""):
        parts.append(f"s={basic}")
    tol = row.get("tolerance")
    if tol not in (None, ""):
        parts.append(f"±{tol}")
    elif row.get("has_tolerance") is True:
        parts.append("±?")
    ang = row.get("angle")
    if ang is not None and ang != "":
        try:
            parts.append(f"∠{float(ang):g}")
        except (TypeError, ValueError):
            parts.append(f"∠{ang}")
    if row.get("dimension_parse_skipped"):
        parts.append("skip")
    elif row.get("keep_pair"):
        parts.append("overlap")
    return " | ".join(parts) if parts else str(row.get("instance_id") or "attr")


def _attribute_box_color(
    row: dict[str, Any],
    colors: dict[str, Any],
) -> tuple[int, int, int]:
    if row.get("keep_pair") or row.get("dimension_parse_skipped"):
        return tuple(colors.get("attribute_overlap", [251, 146, 60]))  # type: ignore[return-value]
    kind = str(row.get("dim_kind") or "").lower()
    kind_key = f"attribute_{kind}" if kind else ""
    if kind_key and colors.get(kind_key):
        return tuple(colors[kind_key])  # type: ignore[return-value]
    return tuple(colors.get("attribute", [14, 165, 233]))  # type: ignore[return-value]


def _draw_attribute_boxes(
    draw: ImageDraw.ImageDraw,
    facts: dict[str, Any] | None,
    *,
    colors: dict[str, Any],
    width: int,
    show_label: bool,
    font: ImageFont.ImageFont,
) -> list[tuple[tuple[int, int, int], list[str]]]:
    """绘制已识别属性框，并返回图例条目。"""
    rows = _attribute_rows_from_facts(facts)
    if not rows:
        return []
    n_ok = 0
    n_overlap = 0
    for row in rows:
        bbox = row.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        color = _attribute_box_color(row, colors)
        quad = row.get("quad")
        if (
            isinstance(quad, list)
            and len(quad) >= 4
            and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in quad[:4])
        ):
            pts = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in quad[:4]]
            draw.polygon(pts, outline=color, width=width)
            lx0 = min(p[0] for p in pts)
            ly0 = min(p[1] for p in pts)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            lx0, ly0 = x1, y1
        if row.get("keep_pair") or row.get("dimension_parse_skipped"):
            n_overlap += 1
        else:
            n_ok += 1
        if show_label:
            label = _format_attribute_label(row)
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lx, ly = lx0, max(0, ly0 - th - 4)
            draw.rectangle([lx, ly, lx + tw + 4, ly + th + 2], fill=(255, 255, 255))
            draw.text((lx + 2, ly), label, fill=color, font=font)
    legend: list[tuple[tuple[int, int, int], list[str]]] = []
    if n_ok:
        c = tuple(colors.get("attribute", [14, 165, 233]))
        legend.append((c, [f"[attr] dimension×{n_ok}", "已识别尺寸属性"]))  # type: ignore[arg-type]
    if n_overlap:
        c = tuple(colors.get("attribute_overlap", [251, 146, 60]))
        legend.append((c, [f"[attr] overlap×{n_overlap}", "重叠/跳过属性解析"]))  # type: ignore[arg-type]
    return legend


def _markdown_report(findings_payload: dict[str, Any], facts: dict[str, Any] | None = None) -> str:
    lines = [
        f"# 质检报告 — {findings_payload.get('drawing_id', '')}",
        "",
        f"- passed: **{findings_payload.get('passed')}**",
        f"- rule_set_id: `{findings_payload.get('rule_set_id')}`",
        f"- rule_set_version: `{findings_payload.get('rule_set_version')}`",
        f"- perception_backend: `{findings_payload.get('perception_backend')}`",
        f"- findings: {len(findings_payload.get('findings') or [])}",
        "",
        "## Findings",
        "",
    ]
    findings = findings_payload.get("findings") or []
    if not findings:
        lines.append("无违规项。")
    else:
        lines.append("| severity | rule_id | message | path | actual |")
        lines.append("|---|---|---|---|---|")
        for f in findings:
            lines.append(
                f"| {f.get('severity')} | `{f.get('rule_id')}` | {f.get('message')} | "
                f"`{f.get('path')}` | {f.get('actual')} |"
            )
    if facts and facts.get("meta"):
        meta = facts["meta"]
        lines.extend(
            [
                "",
                "## Meta",
                "",
                f"- size: {meta.get('width')}x{meta.get('height')}",
                f"- page: {meta.get('page')}",
                f"- source: `{meta.get('source_path')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _view_region_rows_from_facts(facts: dict[str, Any] | None) -> list[dict[str, Any]]:
    """部件可视化行：优先 facts.components，回退 _instances 中的 component。"""
    if not facts:
        return []
    rows: list[dict[str, Any]] = []
    for c in facts.get("components") or []:
        if isinstance(c, dict) and c.get("bbox"):
            rows.append(c)
    if rows:
        return rows
    for inst in facts.get("_instances") or []:
        if not isinstance(inst, dict) or not inst.get("bbox"):
            continue
        if str(inst.get("entity_id") or "") not in {"component", "components"}:
            continue
        fields = inst.get("fields") if isinstance(inst.get("fields"), dict) else {}
        rows.append(
            {
                **(fields or {}),
                "bbox": inst.get("bbox"),
                "bbox_expanded": inst.get("bbox_expanded"),
                "label": inst.get("label") or (fields or {}).get("label"),
                "instance_id": inst.get("instance_id"),
            }
        )
    return rows


def _draw_view_region_boxes(
    draw: ImageDraw.ImageDraw,
    facts: dict[str, Any] | None,
    *,
    colors: dict[str, Any],
    width_raw: int,
    width_expanded: int,
    show_label: bool,
    font: ImageFont.ImageFont,
    expand_ratio: float = 0.2,
) -> list[tuple[tuple[int, int, int], list[str]]]:
    """绘制部件原始框与扩展框。"""
    rows = _view_region_rows_from_facts(facts)
    if not rows:
        return []
    color_raw = tuple(colors.get("view_raw", [234, 179, 8]))  # type: ignore[assignment]
    color_exp = tuple(colors.get("view_expanded", [239, 68, 68]))  # type: ignore[assignment]
    page_w = int((facts or {}).get("meta", {}).get("width") or 0) or None
    page_h = int((facts or {}).get("meta", {}).get("height") or 0) or None
    n = 0
    for row in rows:
        raw = row.get("bbox")
        exp = row.get("bbox_expanded")
        if (not exp or len(exp) != 4) and raw and len(raw) == 4 and page_w and page_h:
            from pipeline.view_regions import expand_view_bbox

            exp = expand_view_bbox(list(raw), page_w, page_h, expand_ratio=expand_ratio)
        label = str(row.get("label") or row.get("instance_id") or f"view#{n}")
        if isinstance(exp, (list, tuple)) and len(exp) == 4:
            x1, y1, x2, y2 = [int(v) for v in exp]
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline=color_exp, width=width_expanded)
                if show_label:
                    tag = f"{label}/exp"
                    tb = draw.textbbox((0, 0), tag, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                    lx, ly = x1 + 2, max(0, y1 + 2)
                    draw.rectangle([lx, ly, lx + tw + 4, ly + th + 2], fill=(255, 255, 255))
                    draw.text((lx + 2, ly), tag, fill=color_exp, font=font)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            x1, y1, x2, y2 = [int(v) for v in raw]
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline=color_raw, width=width_raw)
                if show_label:
                    tag = f"{label}/raw"
                    tb = draw.textbbox((0, 0), tag, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                    lx, ly = x1 + 2, max(0, y1 + 2)
                    draw.rectangle([lx, ly, lx + tw + 4, ly + th + 2], fill=(255, 255, 255))
                    draw.text((lx + 2, ly), tag, fill=color_raw, font=font)
        n += 1
    return [
        (color_raw, [f"[view] raw ×{n}", "部件原始框"]),  # type: ignore[list-item]
        (color_exp, [f"[view] expanded ×{n}", "部件扩展框"]),  # type: ignore[list-item]
    ]


def render_report(
    image_path: str | Path,
    findings_payload: dict[str, Any],
    *,
    out_vis: str | Path | None = None,
    out_report: str | Path | None = None,
    out_markdown: str | Path | None = None,
    facts: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    cfg = config or load_config()
    render_cfg = cfg.get("render", {}) or {}
    colors = render_cfg.get("colors", {}) or {}
    width = int(render_cfg.get("box_width", 3))
    show_table_boxes = bool(render_cfg.get("show_table_boxes", False))
    table_width = int(render_cfg.get("table_box_width", width))
    table_label = bool(render_cfg.get("table_label", True))
    show_attribute_boxes = bool(render_cfg.get("show_attribute_boxes", False))
    attr_width = int(render_cfg.get("attribute_box_width", max(1, width - 1)))
    attr_label = bool(render_cfg.get("attribute_label", True))
    show_view_regions = bool(render_cfg.get("show_view_regions", False))
    view_raw_width = int(render_cfg.get("view_raw_box_width", width))
    view_exp_width = int(render_cfg.get("view_expanded_box_width", max(1, width - 1)))
    view_label = bool(render_cfg.get("view_region_label", True))
    expand_ratio = float(
        ((cfg.get("perception") or {}).get("view_regions") or {}).get("expand_ratio", 0.2)
    )

    img = Image.open(resolve_path(image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    legend_font = _font(18)
    attr_font = _font(int(render_cfg.get("attribute_font_size", 14)))

    legend_entries: list[tuple[tuple[int, int, int], list[str]]] = []

    # 部件区：先画扩展框再画原始框
    if show_view_regions:
        legend_entries.extend(
            _draw_view_region_boxes(
                draw,
                facts,
                colors=colors,
                width_raw=view_raw_width,
                width_expanded=view_exp_width,
                show_label=view_label,
                font=legend_font,
                expand_ratio=expand_ratio,
            )
        )

    # 先画属性框，再画表格与 findings，避免属性标签被完全盖住
    if show_attribute_boxes:
        legend_entries.extend(
            _draw_attribute_boxes(
                draw,
                facts,
                colors=colors,
                width=attr_width,
                show_label=attr_label,
                font=attr_font,
            )
        )

    # 表格识别框
    if show_table_boxes:
        legend_entries.extend(
            _draw_table_boxes(
                draw,
                facts,
                colors=colors,
                width=table_width,
                show_label=table_label,
                font=legend_font,
            )
        )

    for finding in findings_payload.get("findings", []):
        sev = finding.get("severity", "error")
        color = tuple(colors.get(sev, [220, 38, 38]))
        boxes = _display_boxes(finding, facts=facts)
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        legend_entries.append((color, _legend_lines(finding, len(boxes))))

    _draw_legend(draw, img.width, img.height, legend_entries, font=legend_font)

    drawing_id = findings_payload.get("drawing_id") or "drawing"
    vis_dir = resolve_path(cfg["work_dirs"]["vis"])
    report_dir = resolve_path(cfg["work_dirs"]["reports"])
    vis_path = resolve_path(out_vis) if out_vis else vis_dir / f"{drawing_id}_annotated.jpg"
    report_path = resolve_path(out_report) if out_report else report_dir / f"{drawing_id}_report.json"
    md_path = resolve_path(out_markdown) if out_markdown else report_dir / f"{drawing_id}_report.md"

    vis_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(vis_path, quality=92)
    dump_json(findings_payload, report_path)
    md_path.write_text(_markdown_report(findings_payload, facts), encoding="utf-8")
    return {"vis": vis_path, "report": report_path, "markdown": md_path}
