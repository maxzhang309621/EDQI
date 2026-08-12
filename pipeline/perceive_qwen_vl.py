"""S4+S5 主路径：Qwen3-VL 定位 + 字段解析。

参考: https://github.com/QwenLM/Qwen3-VL
权重本地目录默认: models/Qwen3-VL-4B-Instruct
未下载权重时自动回退 mock。
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import load_config, model_path_ready, resolve_model_source, resolve_path
from pipeline.dimension_parse import (
    bbox_geometry_ok,
    clip_fields_to_schema,
    enrich_dimension_fields_from_text,
    is_valid_dimension_mark,
    text_has_diameter_or_angle_symbol,
)
from pipeline.perceive_common import (
    expand_bbox,
    extract_json_payload_safe,
    filter_instances_outside_bboxes,
    is_table_entity_id,
    mock_perceive,
    norm_bbox_to_pixels,
)
from pipeline.table_layout_ocr import (
    apply_product_split_to_boxes,
    extract_fields_above_labels,
)

_VLM = None
_PROCESSOR = None
_VLM_SOURCE: str | None = None

# 单次 perceive_qwen_vl 调用内的推理计时（秒）
_TIMING: dict[str, float | int] = {
    "vlm_load_s": 0.0,
    "vlm_generate_s": 0.0,
    "vlm_generate_n": 0,
}


def _reset_vlm_timing() -> None:
    _TIMING["vlm_load_s"] = 0.0
    _TIMING["vlm_generate_s"] = 0.0
    _TIMING["vlm_generate_n"] = 0


def _snapshot_vlm_timing(*, wall_s: float) -> dict[str, float | int]:
    return {
        "wall_s": round(float(wall_s), 3),
        "vlm_load_s": round(float(_TIMING["vlm_load_s"]), 3),
        "vlm_generate_s": round(float(_TIMING["vlm_generate_s"]), 3),
        "vlm_generate_n": int(_TIMING["vlm_generate_n"]),
    }


def _is_dimension_marks_entity(ent: dict[str, Any] | None) -> bool:
    return bool(ent) and str(ent.get("parse_kind") or "").lower() == "dimension_marks"


def _is_table_entity(ent: dict[str, Any]) -> bool:
    if ent.get("parse_kind") == "table":
        return True
    eid = str(ent.get("entity_id") or "")
    return eid in {"info_table", "material_table", "main_table"} or eid.endswith("_table")


def _section_of(inst: dict[str, Any], plan: list[dict[str, Any]]) -> str:
    eid = str(inst.get("entity_id") or "")
    if eid == "material_table" or eid.endswith("material_table"):
        return "material"
    if eid == "main_table" or eid.endswith("main_table"):
        return "main"
    for e in plan:
        if e.get("entity_id") == eid:
            return str(e.get("section") or "")
    return ""


def enforce_material_main_boundary(
    boxes: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    plan: list[dict[str, Any]] | None = None,
    gap: int = 2,
    min_main_height: int = 80,
) -> tuple[list[dict[str, Any]], list[str]]:
    """硬分界：主表上沿必须在物料表下沿之下，尽量不裁掉主表下半内容。

    - 上抬 main.y1 到 material.y2 + gap（保留 main.y2，保护下方主表字段）
    - 若物料框过厚导致主表过矮：回缩 material.y2，再设 main.y1，保证主表至少 min_main_height
    """
    plan = plan or []
    notes: list[str] = []
    mat_idxs = [
        i
        for i, b in enumerate(boxes)
        if _section_of(b, plan) == "material" or b.get("entity_id") == "material_table"
    ]
    main_idxs = [
        i
        for i, b in enumerate(boxes)
        if _section_of(b, plan) == "main" or b.get("entity_id") == "main_table"
    ]
    if not mat_idxs or not main_idxs:
        return boxes, notes

    out = [dict(b) for b in boxes]
    for mi in mat_idxs:
        for ni in main_idxs:
            mb = list(out[mi]["bbox"])
            nb = list(out[ni]["bbox"])
            mx1, my1, mx2, my2 = [int(v) for v in mb]
            nx1, ny1, nx2, ny2 = [int(v) for v in nb]

            # 物料应在上：若物料整体在主表下方，交换语义上不合理时仍按 y 排序修正
            if my1 > ny1 + 10 and my2 > ny2 - 10:
                # 模型把两框上下颠倒：用更靠上的作物料
                notes.append("boundary_swap_hint: material_below_main")
                if ny1 < my1:
                    # treat current main as upper strip — swap roles for geometry only
                    mx1, my1, mx2, my2, nx1, ny1, nx2, ny2 = nx1, ny1, nx2, ny2, mx1, my1, mx2, my2
                    out[mi]["bbox"] = [mx1, my1, mx2, my2]
                    out[ni]["bbox"] = [nx1, ny1, nx2, ny2]

            mb = list(out[mi]["bbox"])
            nb = list(out[ni]["bbox"])
            mx1, my1, mx2, my2 = [int(v) for v in mb]
            nx1, ny1, nx2, ny2 = [int(v) for v in nb]

            # 需要的主表高度：优先保留原主表高度，至少 min_main_height
            desired_main_h = max(min_main_height, ny2 - ny1)
            max_mat_bottom = max(my1 + 8, ny2 - desired_main_h - gap)
            if my2 > max_mat_bottom:
                notes.append(f"boundary_trim_material_y2:{my2}->{max_mat_bottom}")
                my2 = max_mat_bottom
                out[mi]["bbox"] = [mx1, my1, mx2, max(my1 + 4, my2)]

            new_main_y1 = int(out[mi]["bbox"][3]) + gap
            if new_main_y1 > ny1:
                notes.append(f"boundary_raise_main_y1:{ny1}->{new_main_y1}")
            ny1 = max(ny1, new_main_y1)
            # 不减小 y2，必要时略向下扩以保高度
            if ny2 - ny1 < min_main_height:
                ny2 = min(page_h, ny1 + max(min_main_height, desired_main_h))
            if ny1 >= ny2 - 4:
                ny1 = max(0, ny2 - min_main_height)
            out[ni]["bbox"] = [
                max(0, nx1),
                max(0, ny1),
                min(page_w, nx2),
                min(page_h, ny2),
            ]
            # 记录分界供裁剪时禁止上扩越过
            out[ni]["_y1_floor"] = int(out[mi]["bbox"][3]) + gap
            out[mi]["_boundary_peer"] = out[ni].get("entity_id")
    return out, notes


def refine_material_main_boxes(
    page_image: Image.Image,
    boxes: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    plan: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """优先 OCR 定位 Product 行硬分界；失败则回退几何硬分界。"""
    plan = plan or []
    notes: list[str] = []
    split_boxes, split_notes = apply_product_split_to_boxes(page_image, boxes, plan)
    notes.extend(split_notes)
    if any(n.startswith("product_split:row=") for n in split_notes):
        notes.append("boundary_source:product_row_ocr")
        return split_boxes, notes
    out, bound_notes = enforce_material_main_boundary(
        boxes, page_w=page_w, page_h=page_h, plan=plan, gap=2, min_main_height=max(80, page_h // 12)
    )
    notes.extend(bound_notes)
    notes.append("boundary_source:geometry_fallback")
    return out, notes


def _use_ocr_above_cells(ent: dict[str, Any]) -> bool:
    return _is_table_entity(ent) and str(ent.get("read_mode") or "").lower() == "above_cells"


def _coerce_instance_list(
    raw: Any,
    *,
    notes: list[str] | None = None,
    tag: str = "parse",
) -> list[Any]:
    """把模型 JSON 统一成 instance 列表；单对象勿丢成 []。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("instances", "data"):
            val = raw.get(key)
            if isinstance(val, list):
                return val
        if raw.get("entity_id") or raw.get("bbox_2d") or raw.get("bbox"):
            if notes is not None:
                notes.append(f"{tag}_single_object_wrapped")
            return [raw]
        if notes is not None:
            notes.append(f"{tag}_dict_without_instances")
        return []
    if notes is not None:
        notes.append(f"{tag}_json_unexpected_type={type(raw).__name__}")
    return []


def _dimension_marks_locate_rules() -> list[str]:
    return [
        "对于尺寸标注（parse_kind=dimension_marks / number_mark）—严格模式：",
        "只定位视图区中的尺寸数字与公差标注（含 Ø/⌀/Φ 直径、R 半径、长度数值、角度°、±公差）。",
        "bbox 必须紧贴该尺寸文字，禁止拉成横贯半页/整页的细长条。",
        "严禁定位表格框内任何文字：material_table、main_table、标题栏/参数表/BOM 单元格内的数字与代号一律不要输出为 number_mark。",
        "禁止定位：标题栏/图框表格、图号、材料牌号、Siemens/版权、比例、页码、"
        "零件名、粗糙度代号旁非尺寸串、坐标刻度、网格线、任何非尺寸文本。",
        "拿不准是否为尺寸标注时宁可漏检，不要输出。",
        "每个独立尺寸标注各一个 bbox；重叠/压盖的尺寸也要分开框。",
    ]


def _dimension_marks_pass2_prompt(ent: dict[str, Any]) -> str:
    allowed = [str(f.get("name")) for f in (ent.get("fields") or []) if f.get("name")]
    field_desc = json_fields(ent)
    allow_s = ", ".join(allowed) if allowed else "text, dim_kind, basic_size, tolerance, has_tolerance, angle"
    return (
        "严格判定该局部图是否为工程视图中的尺寸标注。只输出 JSON。\n"
        f"fields 只允许这些键（不得增删）: {allow_s}\n"
        "若不是尺寸标注，全部字段填 null，raw_text=\"\"（宁缺毋滥）。\n"
        f"字段说明: {field_desc}\n"
        "规则:\n"
        "- 优先检查数字旁是否有 Ø/⌀/Φ（直径）或 °（角度）；有则 dim_kind=diameter|angle，"
        "并在 text 中写出符号（如 Ø10、45°）；\n"
        "- R/r 后直接跟数字 → radius；普通长度数值/±公差 → length；\n"
        "- 若裁剪内容明显是表格/标题栏单元格（标签旁的表值、图号、材料等），全部填 null；\n"
        "- 一律拒绝：Max./MIN/TYP/REF、粗糙度(Rz/Ra)、视图字母、气泡号、图号、表值、残片；\n"
        "- dim_kind 只能是 diameter|radius|length|angle，否则 null；\n"
        "- basic_size=基本尺寸；tolerance=公差（无则 null）；has_tolerance 为布尔；\n"
        "- angle=文本相对水平线朝向角（度，竖排≈±90）；\n"
        "- 禁止输出未声明字段，禁止编造，禁止把非尺寸硬套成 length。\n"
        "格式: {\"fields\":{...},\"raw_text\":\"...\"}"
    )


def filter_ocr_dimension_candidates_with_vlm(
    image_path: str | Path,
    candidates: list[dict[str, Any]],
    ent: dict[str, Any],
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    notes: list[str] | None = None,
    allow_mock_fallback: bool = True,
    generate_fn=None,
) -> list[dict[str, Any]]:
    """对 OCR 初筛后的尺寸候选做 VLM crop 精筛；保留 OCR bbox，只更新字段。"""
    import json

    if not candidates:
        if notes is not None:
            notes.append("vlm_dim_filter=0/0")
        return []

    cfg = config or load_config()
    model_cfg = cfg.get("models", {}).get("qwen3_vl", {})
    meta = meta or {}
    image = Image.open(resolve_path(image_path)).convert("RGB")
    width, height = image.size
    meta.setdefault("width", width)
    meta.setdefault("height", height)

    flags = _dimension_ent_strict_flags(ent)
    allowed = [str(f.get("name")) for f in (ent.get("fields") or []) if f.get("name")]
    expand = float(ent.get("vlm_filter_crop_expand", model_cfg.get("crop_expand", 0.12)))
    max_tokens = int(ent.get("max_new_tokens") or model_cfg.get("max_new_tokens", 512))
    # 精筛只需短 JSON
    max_tokens = min(max_tokens, 512)
    field_prompt = _dimension_marks_pass2_prompt(ent)

    use_mock = False
    model = processor = None
    if generate_fn is None:
        if not model_path_ready(model_cfg):
            if allow_mock_fallback:
                use_mock = True
                if notes is not None:
                    notes.append("vlm_dim_filter_mock=true")
            else:
                raise FileNotFoundError(model_cfg.get("path"))
        else:
            model, processor = _load_vlm(model_cfg)

    kept: list[dict[str, Any]] = []
    rejected = 0
    ocr_symbol_fallback = 0
    for inst in candidates:
        bbox = inst.get("bbox")
        if not bbox or len(bbox) != 4:
            rejected += 1
            continue
        ocr_fields = dict(inst.get("fields") or {})
        ocr_text = str(inst.get("raw_text") or ocr_fields.get("text") or "")
        # 纯数字 / OCR 标成待 VLM 认符号：加大裁剪以看见 Ø/°
        local_expand = expand
        role = str(ocr_fields.get("ocr_role") or "")
        bare_num = bool(
            re.fullmatch(r"[±+\-]?\d+(?:[.,]\d+)?", ocr_text.strip())
            or re.fullmatch(r"[Rr]\s*\d+(?:[.,]\d+)?(?:\s*[±+\-].*)?", ocr_text.strip())
        )
        if role == "vlm_symbol" or (bare_num and not re.search(r"[Ø⌀Φφø°º]", ocr_text)):
            local_expand = min(0.55, max(expand * 2.8, expand + 0.22))
        crop_box = expand_bbox(list(bbox), width, height, local_expand)
        crop = image.crop(tuple(crop_box))
        try:
            if generate_fn is not None:
                text2 = generate_fn(crop, field_prompt, max_tokens)
            elif use_mock:
                # 无权重时沿用 OCR 初筛结果，避免整批属性被清空
                payload = {
                    "fields": {
                        "text": ocr_text,
                        "dim_kind": ocr_fields.get("dim_kind"),
                        "basic_size": ocr_fields.get("basic_size"),
                        "tolerance": ocr_fields.get("tolerance"),
                        "has_tolerance": ocr_fields.get("has_tolerance"),
                        "angle": ocr_fields.get("angle"),
                    },
                    "raw_text": ocr_text,
                }
                text2 = json.dumps(payload, ensure_ascii=False)
            else:
                text2 = _vlm_generate(model, processor, crop, field_prompt, max_tokens)
            parsed = extract_json_payload_safe(text2, default={})
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            fields = parsed.get("fields", parsed) if isinstance(parsed, dict) else {}
            if not isinstance(fields, dict):
                fields = {}
            raw_text = str(parsed.get("raw_text") or "") if isinstance(parsed, dict) else ""
            # 以 VLM 为准；若 OCR 已有 Ø/° 证据而 VLM 判空，允许规则回填，避免误杀直径/角度
            vlm_text = str(fields.get("text") or raw_text or "").strip()
            vlm_kind = str(fields.get("dim_kind") or "").strip().lower()
            if vlm_kind in {"", "null", "none"}:
                vlm_kind = ""
            ocr_symbol = text_has_diameter_or_angle_symbol(ocr_text) or str(
                ocr_fields.get("dim_kind") or ""
            ).lower() in {"diameter", "angle"}
            if not vlm_text and not vlm_kind:
                if ocr_symbol and is_valid_dimension_mark(
                    ocr_fields,
                    text=ocr_text,
                    require_dim_kind=flags["require_dim_kind"],
                    require_basic_size=flags["require_basic_size"],
                ):
                    fields = dict(ocr_fields)
                    vlm_text = ocr_text
                    ocr_symbol_fallback += 1
                else:
                    rejected += 1
                    continue
            text_v = vlm_text or ocr_text
            fields = enrich_dimension_fields_from_text(fields, text_v, bbox=bbox)
            if flags["strict"]:
                fields = clip_fields_to_schema(fields, allowed)
            # 再次确认：声明字段外的杂讯已裁掉后，仍须是合法尺寸
            if not is_valid_dimension_mark(
                fields,
                text=text_v,
                require_dim_kind=flags["require_dim_kind"],
                require_basic_size=flags["require_basic_size"],
            ):
                # OCR 已有直径/角度符号时再兜一次
                if ocr_symbol and is_valid_dimension_mark(
                    ocr_fields,
                    text=ocr_text,
                    require_dim_kind=flags["require_dim_kind"],
                    require_basic_size=flags["require_basic_size"],
                ):
                    fields = enrich_dimension_fields_from_text(dict(ocr_fields), ocr_text, bbox=bbox)
                    if flags["strict"]:
                        fields = clip_fields_to_schema(fields, allowed)
                    text_v = ocr_text
                    ocr_symbol_fallback += 1
                else:
                    rejected += 1
                    continue
            if not bbox_geometry_ok(
                bbox,
                page_w=width,
                page_h=height,
                max_width_ratio=flags["max_bbox_width_ratio"],
                max_aspect_ratio=flags["max_aspect_ratio"],
                max_height_ratio=float(ent.get("max_bbox_height_ratio", 0.12)),
                max_area_ratio=float(ent.get("max_bbox_area_ratio", 0.035)),
            ):
                rejected += 1
                continue
            fields["ocr_prescreen"] = True
            fields["vlm_filtered"] = True
            if role:
                fields["ocr_role"] = role
            out = dict(inst)
            out["fields"] = fields
            out["raw_text"] = text_v or out.get("raw_text") or ""
            if fields.get("angle") is not None:
                out["angle"] = fields.get("angle")
            out["label"] = out.get("label") or ent.get("locate_query") or "number_mark"
            kept.append(out)
        except Exception:
            rejected += 1
            continue

    if notes is not None:
        notes.append(f"vlm_dim_filter={len(kept)}/{len(candidates)}")
        notes.append(f"vlm_dim_rejected={rejected}")
        if ocr_symbol_fallback:
            notes.append(f"vlm_dim_ocr_symbol_fallback={ocr_symbol_fallback}")
    return kept


def _dimension_ent_strict_flags(ent: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict": bool(ent.get("strict_fields_only", True)),
        "require_dim_kind": bool(ent.get("require_dim_kind", True)),
        "require_basic_size": bool(ent.get("require_basic_size", True)),
        "exclude_table_regions": bool(ent.get("exclude_table_regions", True)),
        "max_bbox_width_ratio": float(ent.get("max_bbox_width_ratio", 0.22)),
        "max_aspect_ratio": float(ent.get("max_aspect_ratio", 8.0)),
        "max_bbox_height_ratio": float(ent.get("max_bbox_height_ratio", 0.12)),
        "max_bbox_area_ratio": float(ent.get("max_bbox_area_ratio", 0.035)),
    }


def _finalize_dimension_instances(
    instances: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """严格按 drawing_parse 配置过滤尺寸属性；非尺寸/畸形框/表格内丢弃。"""
    from pipeline.perceive_common import build_table_exclude_regions

    dim_ents = {e["entity_id"]: e for e in plan if _is_dimension_marks_entity(e)}
    if not dim_ents:
        return instances

    kept: list[dict[str, Any]] = []
    dropped = 0
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        eid = inst.get("entity_id")
        ent = dim_ents.get(eid)
        if not ent:
            kept.append(inst)
            continue
        flags = _dimension_ent_strict_flags(ent)
        allowed = [str(f.get("name")) for f in (ent.get("fields") or []) if f.get("name")]
        fields = dict(inst.get("fields") or {})
        text_v = str(fields.get("text") or inst.get("raw_text") or "")
        fields = enrich_dimension_fields_from_text(fields, text_v, bbox=inst.get("bbox"))
        if flags["strict"]:
            fields = clip_fields_to_schema(fields, allowed)
        # 全图/分块 VLM 定位的尺寸视为已确认（避免 sanitize 按 vlm_require 误杀）
        fields["vlm_filtered"] = True
        inst = dict(inst)
        inst["fields"] = fields
        if fields.get("text"):
            inst["raw_text"] = fields.get("text")
        if fields.get("angle") is not None:
            inst["angle"] = fields.get("angle")

        if not bbox_geometry_ok(
            inst.get("bbox"),
            page_w=page_w,
            page_h=page_h,
            max_width_ratio=flags["max_bbox_width_ratio"],
            max_aspect_ratio=flags["max_aspect_ratio"],
            max_height_ratio=flags["max_bbox_height_ratio"],
            max_area_ratio=flags["max_bbox_area_ratio"],
        ):
            dropped += 1
            continue
        if not is_valid_dimension_mark(
            fields,
            text=text_v,
            require_dim_kind=flags["require_dim_kind"],
            require_basic_size=flags["require_basic_size"],
        ):
            dropped += 1
            continue
        kept.append(inst)

    if any(_dimension_ent_strict_flags(e)["exclude_table_regions"] for e in dim_ents.values()):
        pad = max(
            (float(e.get("exclude_table_pad", 2.0)) for e in dim_ents.values()),
            default=2.0,
        )
        expand_up = max(
            (float(e.get("exclude_table_expand_up", 0.12)) for e in dim_ents.values()),
            default=0.12,
        )
        table_regions = build_table_exclude_regions(
            instances,
            page_w=page_w,
            page_h=page_h,
            pad=pad,
            expand_up_frac=expand_up,
        )
        if table_regions:
            dim_only = [i for i in kept if i.get("entity_id") in dim_ents]
            other = [i for i in kept if i.get("entity_id") not in dim_ents]
            filtered, n_tab = filter_instances_outside_bboxes(
                dim_only, table_regions, pad=0.0, entity_ids=None
            )
            dropped += n_tab
            kept = other + filtered
            if notes is not None:
                notes.append(f"dimension_exclude_table_regions={n_tab}")

    if notes is not None:
        notes.append(f"dimension_strict_dropped={dropped}")
    return kept


def _build_plan_prompt(plan: list[dict[str, Any]], *, locate_only: bool = False) -> str:
    has_dims = any(_is_dimension_marks_entity(e) for e in plan)
    if locate_only:
        lines = [
            "你是工程图纸质检感知模块。本轮只做定位，不要填写字段内容。",
            "对于表格类实体：每种表格各输出一个整表 bbox，不要拆成单元格；"
            "material_table 与 main_table 必须分开定位，禁止合成一个大框。"
            "分界参考：标题栏 Product 行以上为 material_table；Product 行及以下为 main_table。",
            "只输出一个简短 JSON 数组，不要 markdown 代码块，不要解释。",
            '每个元素格式: {"bbox_2d":[x1,y1,x2,y2],"label":"...","entity_id":"...","fields":{}}',
            "bbox_2d 使用 0-1000 归一化坐标。fields 必须为 {}。",
            "",
        ]
        if has_dims:
            lines.extend(_dimension_marks_locate_rules())
            lines.append("")
        lines.append("实体计划:")
    else:
        lines = [
            "你是工程图纸质检感知模块。请在图中定位下列实体，并只根据图面可见内容填写字段。",
            "禁止编造图中不存在的文字或数值。",
            "必须找出全部相关实例（every/all），不要合并相邻文字；每个独立数字/尺寸标注各输出一个框。",
            "若存在相互压盖、交叉、重叠的数字，也必须分别给出各自的 bbox。",
            "对于表格类实体：每种表格各输出一个整表 bbox，不要拆成单元格；"
            "material_table 与 main_table 必须分开定位，禁止合成一个大框。"
            "分界：以标题栏中 Product 字段所在行为界——"
            "Product 行以上为 material_table；Product 行及其以下为 main_table。",
            "只输出 JSON 数组，不要解释。每个元素格式:",
            '{"bbox_2d":[x1,y1,x2,y2],"label":"...","entity_id":"...","fields":{...},"raw_text":"..."}',
            "bbox_2d 使用 0-1000 归一化坐标。",
            "",
        ]
        if has_dims:
            lines.extend(_dimension_marks_locate_rules())
            lines.append("")
        lines.append("实体计划:")
    for ent in plan:
        field_desc = ", ".join(
            f"{f['name']}({f.get('parse_hint', f['name'])})" for f in ent.get("fields", [])
        )
        many_hint = "【多实例，全部检出】" if ent.get("cardinality") == "many" else ""
        table_hint = ""
        if _is_table_entity(ent):
            mode = ent.get("read_mode") or "cell_content"
            table_hint = f"【整表一个框|section={ent.get('section')}|read_mode={mode}】"
        dim_hint = "【尺寸属性|排除标题栏/图号/材料】" if _is_dimension_marks_entity(ent) else ""
        if locate_only:
            lines.append(
                f"- entity_id={ent['entity_id']}, query={ent['locate_query']}"
                f"（本轮只要框，fields={{}}）{many_hint}{table_hint}{dim_hint}"
            )
        else:
            lines.append(
                f"- entity_id={ent['entity_id']}, query={ent['locate_query']}, "
                f"fields=[{field_desc}] {many_hint}{table_hint}{dim_hint}"
            )
    return "\n".join(lines)


def json_fields(ent: dict[str, Any]) -> str:
    parts = []
    for f in ent.get("fields", []):
        parts.append(f"{f['name']}<=图面「{f.get('parse_hint', f['name'])}」")
    return ", ".join(parts)


def _table_pass2_prompt(ent: dict[str, Any]) -> str:
    read_mode = str(ent.get("read_mode") or "cell_content").lower()
    as_array = bool(ent.get("value_as_array", read_mode == "above_cells"))
    named_lines = []
    for f in ent.get("fields") or []:
        hint = f.get("parse_hint", f["name"])
        locate_by = str(f.get("locate_by") or "label").lower()
        if locate_by == "size_position":
            named_lines.append(
                f"- {f['name']}: 【无标签，按版式定位】{hint}。"
                f"在本表中部寻找面积明显较大的内容单元格，读取其【格内】全部可见文字；"
                f"不要用带 Document/Tolerance/Created by/Surface/Finish/Page/Scale 等标签的小格；"
                f"找不到则填 null。"
            )
            continue
        if read_mode == "above_cells":
            named_lines.append(
                f"- {f['name']}: 先定位名称匹配「{hint}」的标签单元格，"
                f"再读取该标签单元格正上方、同一列中所有内容单元格（自上而下）。"
                f"{'有多格则 fields.' + f['name'] + ' 必须为字符串数组；仅一格也用单元素数组；' if as_array else ''}"
                f"找不到则填 null。禁止读取标签格自身文字、禁止读取其它列。"
            )
        else:
            named_lines.append(
                f"- {f['name']}: 定位名称匹配「{hint}」的标签，"
                f"只读取该字段对应取值单元格【内部】的文字（通常在标签右侧或同一单元格）。"
                f"禁止读取单元格上方物料表内容，禁止读取邻列、零件名大格、图框外文字。"
                f"找不到则填 null。"
            )

    mode_header = (
        "读值规则【above_cells】：字段标签在下，取值在标签单元格上方同列；多格用数组。\n"
        if read_mode == "above_cells"
        else "读值规则【cell_content】：只读标签对应单元格内（或紧邻取值格内）文字，勿取上方格；"
        "另有 locate_by=size_position 的字段按中部大格识别，勿与带标签小格混淆。\n"
    )
    pairs_part = (
        "\n严格只填写上面列出的具名字段。"
        "不要输出其它未列出的字段，不要猜测，fields.pairs 必须为 []。"
        "component_name（若存在）不得填入 Surface/Finish、Created by、Document Subclass 等字段。"
    )
    array_hint = (
        '\n数组字段示例: "volume": ["12.3"] 或 "article_no": ["A1", "A2"]。'
        if as_array
        else '\n标量字段示例: "page": "1/1", "component_name": "Part name"。'
    )

    return (
        f"该局部图像对应「{ent.get('locate_query', '表格')}」(entity_id={ent.get('entity_id')}, "
        f"section={ent.get('section')})。\n"
        + mode_header
        + "若裁剪区不是目标表，返回上述具名字段均为 null、pairs=[]。\n"
        "只根据图面可见内容填写，禁止编造。\n"
        "填写下列具名字段：\n"
        + "\n".join(named_lines)
        + pairs_part
        + array_hint
        + '\n只输出 JSON 对象，格式: {"fields":{...},"raw_text":"..."}'
    )


def _to_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(x).strip() for x in value if x is not None and str(x).strip() != ""]
        return items or None
    text = str(value).strip()
    return [text] if text else None


def _to_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(x).strip() for x in value if x is not None and str(x).strip() != ""]
        if not items:
            return None
        return items[0] if len(items) == 1 else items
    text = str(value).strip()
    return text if text else None


def _normalize_table_fields(fields: dict[str, Any], ent: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(fields or {})
    as_array = bool(ent.get("value_as_array", str(ent.get("read_mode") or "") == "above_cells"))
    for f in ent.get("fields") or []:
        name = f["name"]
        if name not in out:
            out[name] = None
        elif out[name] == "":
            out[name] = None
        elif as_array:
            out[name] = _to_str_list(out[name])
        else:
            out[name] = _to_scalar(out[name])
    # 元数据写入 fields，便于落入 facts.tables
    out["section"] = ent.get("section")
    out["read_mode"] = ent.get("read_mode")
    pairs = out.get("pairs")
    if not isinstance(pairs, list):
        out["pairs"] = []
    else:
        cleaned = []
        named = {str(f["name"]).lower() for f in (ent.get("fields") or [])}
        for item in pairs:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name is None or str(name).strip() == "":
                continue
            if str(name).strip().lower() in named:
                continue
            cleaned.append({"name": str(name).strip(), "content": item.get("content")})
        out["pairs"] = cleaned
    return out


def vlm_is_loaded() -> bool:
    return _VLM is not None


def warmup_qwen_vl(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """进程内预热 Qwen3-VL：只加载一次，供后续多次 perceive/run 复用。"""
    cfg = config or load_config()
    model_cfg = cfg.get("models", {}).get("qwen3_vl", {})
    if not model_path_ready(model_cfg):
        return {"ok": False, "loaded": False, "reason": "weights_not_ready", "path": model_cfg.get("path")}
    already = vlm_is_loaded()
    t0 = time.perf_counter()
    _load_vlm(model_cfg)
    return {
        "ok": True,
        "loaded": True,
        "already_loaded": already,
        "source": _VLM_SOURCE,
        "warmup_s": round(time.perf_counter() - t0, 3),
    }


def _load_vlm(model_cfg: dict[str, Any]):
    global _VLM, _PROCESSOR, _VLM_SOURCE
    source = resolve_model_source(model_cfg)
    if _VLM is not None and _VLM_SOURCE == source:
        return _VLM, _PROCESSOR

    if not model_path_ready(model_cfg):
        raise FileNotFoundError(
            f"本地 VLM 权重未就绪: {model_cfg.get('path')}。"
            f"请下载 {model_cfg.get('hf_id')} 到该目录，或先用 perception_backend=mock。"
        )

    import torch
    try:
        import torchvision  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Qwen3-VL 需要 torchvision。请在当前 venv 中安装，例如:\n"
            "  pip install torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            "若使用 CUDA 版 torch，请安装匹配的 CUDA 版 torchvision（见 https://pytorch.org）。"
        ) from e
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.perf_counter()
    dtype = model_cfg.get("dtype", "auto")
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        dtype=dtype,
        device_map=model_cfg.get("device", "auto"),
    )
    processor = AutoProcessor.from_pretrained(source)
    _VLM, _PROCESSOR, _VLM_SOURCE = model, processor, source
    load_s = time.perf_counter() - t0
    _TIMING["vlm_load_s"] = float(_TIMING["vlm_load_s"]) + load_s
    print(f"[timing] vlm_load={load_s:.2f}s", flush=True)
    return model, processor


def _vlm_generate(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    t0 = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    gen_s = time.perf_counter() - t0
    _TIMING["vlm_generate_s"] = float(_TIMING["vlm_generate_s"]) + gen_s
    _TIMING["vlm_generate_n"] = int(_TIMING["vlm_generate_n"]) + 1
    n = int(_TIMING["vlm_generate_n"])
    print(f"[timing] vlm_generate#{n}={gen_s:.2f}s", flush=True)
    return text[0] if text else ""


def _normalize_instances(
    raw_items: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    width: int,
    height: int,
    coord_norm: float,
) -> list[dict[str, Any]]:
    by_query = {e["entity_id"]: e for e in plan}
    counters: dict[str, int] = {}
    out = []
    for item in raw_items:
        eid = item.get("entity_id")
        if not eid:
            label = item.get("label", "")
            for e in plan:
                if e["locate_query"] in str(label) or str(label) in e["locate_query"]:
                    eid = e["entity_id"]
                    break
        eid = eid or "unknown"
        idx = counters.get(eid, 0)
        counters[eid] = idx + 1
        bbox_raw = item.get("bbox_2d") or item.get("bbox") or [0, 0, 0, 0]
        bbox = norm_bbox_to_pixels(bbox_raw, width, height, coord_norm=coord_norm)
        fields = item.get("fields") or {}
        ent = by_query.get(eid, {})
        if _is_table_entity(ent) and isinstance(fields, dict):
            fields = _normalize_table_fields(fields, ent)
        needs = False
        for f in ent.get("fields", []):
            if f.get("required") and not fields.get(f["name"]):
                needs = True
        out.append(
            {
                "entity_id": eid,
                "instance_id": f"{eid}#{idx}",
                "label": item.get("label") or ent.get("locate_query") or eid,
                "bbox": bbox,
                "fields": fields,
                "raw_text": item.get("raw_text") or "",
                "confidence": float(item.get("confidence", 0.8)),
                "needs_review": bool(item.get("needs_review", needs)),
            }
        )
    return out


def perceive_qwen_vl(
    image_path: str | Path,
    plan: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    allow_mock_fallback: bool = True,
) -> dict[str, Any]:
    cfg = config or load_config()
    model_cfg = cfg.get("models", {}).get("qwen3_vl", {})
    meta = meta or {}
    image = Image.open(resolve_path(image_path)).convert("RGB")
    width, height = image.size
    meta.setdefault("width", width)
    meta.setdefault("height", height)

    _reset_vlm_timing()
    wall_t0 = time.perf_counter()

    if not model_path_ready(model_cfg):
        if allow_mock_fallback:
            payload = mock_perceive(plan, meta)
            payload["backend"] = "mock"
            payload["note"] = "qwen_vl 权重未就绪，已回退 mock"
            payload["timing"] = _snapshot_vlm_timing(wall_s=time.perf_counter() - wall_t0)
            return payload
        raise FileNotFoundError(model_cfg.get("path"))

    model, processor = _load_vlm(model_cfg)
    max_tokens = int(model_cfg.get("max_new_tokens", 2048))
    coord_norm = float(model_cfg.get("coord_norm", 1000))
    two_pass = bool(model_cfg.get("two_pass", True))
    default_expand = float(model_cfg.get("crop_expand", 0.15))

    if two_pass:
        # Pass1 只要短 bbox JSON；过大 max_new_tokens 易废话截断导致整表丢失
        locate_tokens = int(model_cfg.get("locate_max_new_tokens") or min(768, max_tokens))
        text1 = _vlm_generate(
            model, processor, image, _build_plan_prompt(plan, locate_only=True), locate_tokens
        )
        raw1 = extract_json_payload_safe(text1, default=[])
        notes: list[str] = []
        if raw1 is None or raw1 == []:
            preview = (text1 or "").strip().replace("\n", " ")[:240]
            notes.append(f"pass1_json_empty preview={preview!r}")
        raw1 = _coerce_instance_list(raw1, notes=notes, tag="pass1")
        boxes = _normalize_instances(list(raw1), plan, width, height, coord_norm)
        boxes, bound_notes = refine_material_main_boxes(
            image, boxes, page_w=width, page_h=height, plan=plan
        )
        notes.extend(bound_notes)

        instances = []
        for box_inst in boxes:
            eid = box_inst["entity_id"]
            ent = next((e for e in plan if e["entity_id"] == eid), None)
            if not ent:
                instances.append({k: v for k, v in box_inst.items() if not str(k).startswith("_")})
                continue

            # 物料表 above_cells：OCR 先找标签，再读正上方同列；空则为 null
            if _use_ocr_above_cells(ent):
                try:
                    fields, raw_text, ocr_notes = extract_fields_above_labels(
                        image,
                        list(box_inst["bbox"]),
                        list(ent.get("fields") or []),
                        ocr_enhance=ent.get("ocr_enhance")
                        if isinstance(ent.get("ocr_enhance"), dict)
                        else None,
                    )
                    notes.extend(ocr_notes)
                    fields = _normalize_table_fields(fields, ent)
                    out_inst = {
                        **{k: v for k, v in box_inst.items() if not str(k).startswith("_")},
                        "fields": fields,
                        "raw_text": raw_text,
                    }
                    for f in ent.get("fields", []):
                        if f.get("required") and not out_inst["fields"].get(f["name"]):
                            out_inst["needs_review"] = True
                except Exception as exc:
                    notes.append(f"above_cells_ocr_fail:{eid}:{type(exc).__name__}")
                    out_inst = {k: v for k, v in box_inst.items() if not str(k).startswith("_")}
                    out_inst["needs_review"] = True
                instances.append(out_inst)
                continue

            expand = ent.get("crop_expand")
            if expand is None:
                expand = default_expand
            y1_floor = box_inst.get("_y1_floor")
            crop_box = expand_bbox(
                box_inst["bbox"],
                width,
                height,
                float(expand),
                y1_floor=int(y1_floor) if y1_floor is not None else None,
            )
            crop = image.crop(tuple(crop_box))
            tok = int(ent.get("max_new_tokens") or max_tokens)
            if _is_table_entity(ent):
                field_prompt = _table_pass2_prompt(ent)
            elif _is_dimension_marks_entity(ent):
                field_prompt = _dimension_marks_pass2_prompt(ent)
            else:
                field_prompt = (
                    f"读取该局部图像中与「{ent['locate_query']}」相关的字段，只输出 JSON 对象 fields。\n"
                    f"字段: {json_fields(ent)}\n"
                    "禁止编造。格式: {\"fields\":{...},\"raw_text\":\"...\"}"
                )
            try:
                text2 = _vlm_generate(model, processor, crop, field_prompt, tok)
                parsed = extract_json_payload_safe(text2, default={})
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]
                fields = parsed.get("fields", parsed) if isinstance(parsed, dict) else {}
                if not isinstance(fields, dict):
                    fields = {}
                if _is_table_entity(ent):
                    fields = _normalize_table_fields(fields, ent)
                raw_text = parsed.get("raw_text", "") if isinstance(parsed, dict) else ""
                if _is_dimension_marks_entity(ent):
                    text_v = str(fields.get("text") or raw_text or "")
                    fields = enrich_dimension_fields_from_text(
                        fields, text_v, bbox=box_inst.get("bbox")
                    )
                    raw_text = text_v or raw_text
                box_inst = {
                    **{k: v for k, v in box_inst.items() if not str(k).startswith("_")},
                    "fields": fields,
                    "raw_text": raw_text,
                }
                if _is_dimension_marks_entity(ent) and fields.get("angle") is not None:
                    box_inst["angle"] = fields.get("angle")
                for f in ent.get("fields", []):
                    if f.get("required") and not box_inst["fields"].get(f["name"]):
                        box_inst["needs_review"] = True
                if not fields and not raw_text:
                    box_inst["needs_review"] = True
            except Exception:
                box_inst = {k: v for k, v in box_inst.items() if not str(k).startswith("_")}
                box_inst["needs_review"] = True
            else:
                box_inst = {k: v for k, v in box_inst.items() if not str(k).startswith("_")}
            instances.append(box_inst)
        instances = _finalize_dimension_instances(
            instances, plan, page_w=width, page_h=height, notes=notes
        )
        timing = _snapshot_vlm_timing(wall_s=time.perf_counter() - wall_t0)
        print(
            f"[timing] qwen_vl wall={timing['wall_s']:.2f}s "
            f"generate={timing['vlm_generate_s']:.2f}s "
            f"calls={timing['vlm_generate_n']} "
            f"load={timing['vlm_load_s']:.2f}s",
            flush=True,
        )
        payload = {"backend": "qwen_vl", "instances": instances, "timing": timing}
        if notes:
            payload["notes"] = notes
        return payload

    text = _vlm_generate(model, processor, image, _build_plan_prompt(plan, locate_only=False), max_tokens)
    raw = extract_json_payload_safe(text, default=[])
    notes = []
    if not raw:
        preview = (text or "").strip().replace("\n", " ")[:120]
        notes.append(f"single_pass_json_empty preview={preview!r}")
    raw = _coerce_instance_list(raw, notes=notes, tag="single_pass")
    instances = _normalize_instances(list(raw), plan, width, height, coord_norm)
    instances, bound_notes = refine_material_main_boxes(
        image, instances, page_w=width, page_h=height, plan=plan
    )
    notes.extend(bound_notes)
    # single-pass：物料表字段改走 OCR above_cells（覆盖 VL 可能误填的 fields）
    for i, inst in enumerate(instances):
        eid = inst.get("entity_id")
        ent = next((e for e in plan if e["entity_id"] == eid), None)
        if not ent or not _use_ocr_above_cells(ent) or not inst.get("bbox"):
            continue
        try:
            fields, raw_text, ocr_notes = extract_fields_above_labels(
                image,
                list(inst["bbox"]),
                list(ent.get("fields") or []),
                ocr_enhance=ent.get("ocr_enhance")
                if isinstance(ent.get("ocr_enhance"), dict)
                else None,
            )
            notes.extend(ocr_notes)
            inst["fields"] = _normalize_table_fields(fields, ent)
            inst["raw_text"] = raw_text or inst.get("raw_text", "")
        except Exception as exc:
            notes.append(f"above_cells_ocr_fail:{eid}:{type(exc).__name__}")
            inst["needs_review"] = True
        instances[i] = inst
    for i, inst in enumerate(instances):
        eid = inst.get("entity_id")
        ent = next((e for e in plan if e["entity_id"] == eid), None)
        if _is_dimension_marks_entity(ent):
            fields = inst.get("fields") if isinstance(inst.get("fields"), dict) else {}
            text_v = str(fields.get("text") or inst.get("raw_text") or "")
            fields = enrich_dimension_fields_from_text(fields, text_v, bbox=inst.get("bbox"))
            inst["fields"] = fields
            if fields.get("angle") is not None:
                inst["angle"] = fields.get("angle")
            if text_v and not inst.get("raw_text"):
                inst["raw_text"] = text_v
        for k in list(inst.keys()):
            if str(k).startswith("_"):
                del inst[k]
        instances[i] = inst
    instances = _finalize_dimension_instances(
        instances, plan, page_w=width, page_h=height, notes=notes
    )
    timing = _snapshot_vlm_timing(wall_s=time.perf_counter() - wall_t0)
    print(
        f"[timing] qwen_vl wall={timing['wall_s']:.2f}s "
        f"generate={timing['vlm_generate_s']:.2f}s "
        f"calls={timing['vlm_generate_n']} "
        f"load={timing['vlm_load_s']:.2f}s",
        flush=True,
    )
    payload = {"backend": "qwen_vl", "instances": instances, "timing": timing}
    if notes:
        payload["notes"] = notes
    return payload
