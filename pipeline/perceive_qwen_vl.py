"""S4+S5 主路径：Qwen3-VL 定位 + 字段解析。

参考: https://github.com/QwenLM/Qwen3-VL
权重本地目录默认: models/Qwen3-VL-8B-Instruct
未下载权重时自动回退 mock。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import load_config, model_path_ready, resolve_model_source, resolve_path
from pipeline.dimension_parse import enrich_dimension_fields_from_text, parse_dimension_text
from pipeline.perceive_common import (
    expand_bbox,
    extract_json_payload_safe,
    mock_perceive,
    norm_bbox_to_pixels,
)
from pipeline.table_layout_ocr import (
    apply_product_split_to_boxes,
    extract_fields_above_labels,
)
from pipeline.text_angle import normalize_text_angle, text_angle_from_bbox

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
        "对于尺寸标注（parse_kind=dimension_marks / number_mark）：",
        "只定位视图中的尺寸数字与公差标注（含 Ø/⌀/Φ 直径、R 半径、长度、角度°、±公差）。",
        "必须同时检出水平、竖排（沿尺寸线垂直书写）以及大角度倾斜的尺寸文字；"
        "竖排/倾斜的数字也要各自给出 bbox，勿因朝向漏检。",
        "禁止定位：标题栏/图框表格文字、图号、材料牌号、BOM、Siemens/版权、比例、页码、"
        "零件名、表面粗糙度符号旁的非尺寸长串、坐标轴刻度。",
        "每个独立尺寸标注各一个 bbox；重叠/压盖的尺寸也要分开框。",
    ]


def _dimension_marks_pass2_prompt(ent: dict[str, Any]) -> str:
    field_desc = json_fields(ent)
    return (
        "读取该局部图中的工程尺寸标注属性。只输出 JSON。\n"
        "局部图可能已旋转至近水平；也可能仍含竖排或倾斜文字——按工程图正向阅读顺序识读，"
        "勿把竖排数字读反或漏读。\n"
        "若裁剪区不是尺寸标注（图号/材料/标题栏等），fields 全部填 null。\n"
        f"字段: {field_desc}\n"
        "规则:\n"
        "- text=可见原文（正向阅读，如竖排 12 仍写 \"12\"，不要写成倒序）；\n"
        "- dim_kind=diameter|radius|length|angle|null；\n"
        "- basic_size=基本尺寸（±前）；tolerance=公差（±后，无则 null）；\n"
        "- has_tolerance=是否标明公差（布尔）；\n"
        "- angle=原图中文本相对水平线朝向角（度：水平≈0，竖排≈90 或 -90，"
        "倾斜约 ±30~±60；与局部图是否已旋转无关，填原图朝向）。\n"
        "禁止编造。格式: {\"fields\":{...},\"raw_text\":\"...\"}"
    )


def _deskew_crop_for_vlm(crop: Image.Image, angle_deg: float, *, min_side: int = 64) -> Image.Image:
    """将裁剪旋到近水平，过小则放大，便于 VLM 识读竖排/倾斜尺寸。"""
    ang = float(angle_deg)
    while ang < -180.0:
        ang += 360.0
    while ang > 180.0:
        ang -= 360.0
    out = crop
    if abs(ang) >= 1e-3:
        out = crop.rotate(-ang, expand=True, fillcolor=(255, 255, 255))
    dw, dh = out.size
    if max(dw, dh) < min_side and max(dw, dh) > 0:
        scale = float(min_side) / float(max(dw, dh))
        out = out.resize(
            (max(1, int(dw * scale)), max(1, int(dh * scale))),
            Image.Resampling.LANCZOS,
        )
    return out


def _dimension_read_looks_weak(fields: dict[str, Any] | None) -> bool:
    """Pass2 结果是否过弱，值得换朝向再试。"""
    f = fields if isinstance(fields, dict) else {}
    text = str(f.get("text") or "").strip()
    if not text:
        return True
    parsed = parse_dimension_text(text)
    if parsed.get("dim_kind") is None and f.get("dim_kind") is None:
        return True
    if parsed.get("basic_size") is None and f.get("basic_size") in (None, ""):
        # 有类型符号但无数字 → 弱
        if not any(ch.isdigit() for ch in text):
            return True
    return False


def _dimension_pass2_angles(
    bbox: list[Any] | None,
    ent: dict[str, Any],
) -> list[float]:
    """首次 deskew 角 + 可选重试朝向（供 crop.rotate(-ang)；±90 需区分旋向）。"""
    geom = text_angle_from_bbox(bbox)
    deskew_on = bool(ent.get("vlm_deskew_reread", True))
    min_ang = float(ent.get("vlm_deskew_min_angle", 8.0))
    # bbox 粗估只有 0/90；保留字面角，勿把 -90 归一成 90
    primary = float(geom) if deskew_on and abs(geom) >= min_ang else 0.0
    if primary > 90.0:
        primary = 90.0
    if primary < -90.0:
        primary = -90.0
    angles = [primary]

    if not bool(ent.get("vlm_orientation_retry", True)):
        return angles

    if abs(primary) >= 60.0:
        # 对面竖排旋向 + 不旋转
        candidates = [-primary if abs(primary) > 1e-6 else -90.0, 0.0]
    elif abs(primary) < 1e-6:
        candidates = [90.0, -90.0, 45.0, -45.0]
    else:
        candidates = [0.0, 90.0, -90.0, -primary]

    max_extra = max(0, int(ent.get("vlm_orientation_retry_max", 2)))
    for cand in candidates:
        c = float(cand)
        if any(abs(c - a) < 5.0 for a in angles):
            continue
        angles.append(c)
        if len(angles) - 1 >= max_extra:
            break
    return angles


def _parse_dimension_pass2_fields(
    text2: str,
    *,
    bbox: list[Any] | None,
    read_angle: float,
) -> tuple[dict[str, Any], str]:
    parsed = extract_json_payload_safe(text2, default={})
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    fields = parsed.get("fields", parsed) if isinstance(parsed, dict) else {}
    if not isinstance(fields, dict):
        fields = {}
    raw_text = parsed.get("raw_text", "") if isinstance(parsed, dict) else ""
    text_v = str(fields.get("text") or raw_text or "")
    fields = enrich_dimension_fields_from_text(fields, text_v, bbox=bbox)
    # deskew/重试所用朝向即原图文本角估计，优先于模型乱估
    if abs(float(read_angle)) >= 1e-3:
        fields["angle"] = normalize_text_angle(read_angle)
    elif fields.get("angle") in (None, ""):
        fields["angle"] = text_angle_from_bbox(bbox)
    raw_text = text_v or raw_text
    return fields, str(raw_text or "")


def _read_dimension_crop_with_orientation(
    model,
    processor,
    crop: Image.Image,
    ent: dict[str, Any],
    *,
    bbox: list[Any] | None,
    max_tokens: int,
    notes: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """对尺寸裁剪做 deskew / 有限朝向重试，取首个非弱结果。"""
    field_prompt = _dimension_marks_pass2_prompt(ent)
    angles = _dimension_pass2_angles(bbox, ent)
    best_fields: dict[str, Any] = {}
    best_raw = ""
    for i, ang in enumerate(angles):
        view = _deskew_crop_for_vlm(crop, ang)
        text2 = _vlm_generate(model, processor, view, field_prompt, max_tokens)
        fields, raw_text = _parse_dimension_pass2_fields(text2, bbox=bbox, read_angle=ang)
        if notes is not None and i > 0:
            notes.append(f"dimension_orient_retry angle={ang}")
        if not _dimension_read_looks_weak(fields):
            fields["vlm_deskew_angle"] = float(ang)
            return fields, raw_text
        if i == 0 or (raw_text and not best_raw):
            best_fields, best_raw = fields, raw_text
            best_fields["vlm_deskew_angle"] = float(ang)
    return best_fields, best_raw


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
        dim_hint = (
            "【尺寸属性|含竖排/倾斜|排除标题栏/图号/材料】"
            if _is_dimension_marks_entity(ent)
            else ""
        )
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
                field_prompt = None
            else:
                field_prompt = (
                    f"读取该局部图像中与「{ent['locate_query']}」相关的字段，只输出 JSON 对象 fields。\n"
                    f"字段: {json_fields(ent)}\n"
                    "禁止编造。格式: {\"fields\":{...},\"raw_text\":\"...\"}"
                )
            try:
                if _is_dimension_marks_entity(ent):
                    fields, raw_text = _read_dimension_crop_with_orientation(
                        model,
                        processor,
                        crop,
                        ent,
                        bbox=box_inst.get("bbox"),
                        max_tokens=tok,
                        notes=notes,
                    )
                else:
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
