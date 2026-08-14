"""感知通用：坐标变换、JSON 抽取、mock 后端。"""
from __future__ import annotations

import json
import re
from typing import Any


def norm_bbox_to_pixels(
    bbox: list[float],
    width: int,
    height: int,
    *,
    coord_norm: float = 1000.0,
) -> list[int]:
    """将 0~coord_norm 归一化框转为像素 xyxy。若已是像素则原样裁剪。"""
    if len(bbox) != 4:
        raise ValueError(f"bbox 长度必须为 4: {bbox}")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # 启发：若最大值 <= coord_norm 且图尺寸更大，视为归一化坐标
    if max(x1, y1, x2, y2) <= coord_norm + 1 and max(width, height) > coord_norm:
        x1 = x1 / coord_norm * width
        x2 = x2 / coord_norm * width
        y1 = y1 / coord_norm * height
        y2 = y2 / coord_norm * height
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [
        int(max(0, min(width - 1, round(x1)))),
        int(max(0, min(height - 1, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    ]


def expand_bbox(
    bbox: list[int],
    width: int,
    height: int,
    ratio: float = 0.15,
    *,
    y1_floor: int | None = None,
) -> list[int]:
    """按比例外扩 bbox；可选 y1_floor 禁止上沿越过（用于主表不侵入物料表）。"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * ratio), int(bh * ratio)
    ny1 = max(0, y1 - dy)
    if y1_floor is not None:
        ny1 = max(ny1, int(y1_floor))
    return [
        max(0, x1 - dx),
        ny1,
        min(width, x2 + dx),
        min(height, y2 + dy),
    ]


def is_table_entity_id(entity_id: str | None, *, parse_kind: str | None = None) -> bool:
    if str(parse_kind or "").lower() == "table":
        return True
    eid = str(entity_id or "")
    return eid in {"info_table", "material_table", "main_table"} or eid.endswith("_table")


def bbox_center_xy(bbox: list[Any]) -> tuple[float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bbox(x: float, y: float, bbox: list[Any], *, pad: float = 0.0) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def collect_table_bboxes(instances: list[dict[str, Any]] | None) -> list[list[int]]:
    """从感知实例中收集表格整框（像素 xyxy）。"""
    out: list[list[int]] = []
    for inst in instances or []:
        if not isinstance(inst, dict):
            continue
        if not is_table_entity_id(inst.get("entity_id"), parse_kind=inst.get("parse_kind")):
            continue
        bbox = inst.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1, y1, x2, y2])
    return out


_TABLE_META_FIELD_KEYS = frozenset(
    {
        "pairs",
        "read_mode",
        "extract_backend",
        "section",
        "bbox",
        "raw_text",
        "label",
        "instance_id",
        "confidence",
        "needs_review",
        "parse_kind",
        "value_as_array",
        "crop_expand",
        "max_new_tokens",
        "ocr_enhance",
    }
)


def _norm_table_token(text: str) -> str:
    return re.sub(r"[^a-z0-9.+]", "", (text or "").lower())


def collect_table_value_tokens(instances: list[dict[str, Any]] | None) -> set[str]:
    """收集表格已解析字段值（用于剔除表格外重复的图号/材料等 OCR）。"""
    tokens: set[str] = set()
    for inst in instances or []:
        if not isinstance(inst, dict):
            continue
        if not is_table_entity_id(inst.get("entity_id"), parse_kind=inst.get("parse_kind")):
            continue
        fields = inst.get("fields") or {}
        if not isinstance(fields, dict):
            continue
        for key, val in fields.items():
            if key in _TABLE_META_FIELD_KEYS:
                continue
            vals = val if isinstance(val, list) else [val]
            for item in vals:
                if item is None:
                    continue
                s = str(item).strip()
                if len(s) < 3:
                    continue
                tok = _norm_table_token(s)
                if len(tok) >= 3:
                    tokens.add(tok)
    return tokens


def build_table_exclude_regions(
    instances: list[dict[str, Any]] | None,
    *,
    page_w: int,
    page_h: int,
    pad: float = 2.0,
    expand_up_frac: float = 0.12,
) -> list[list[int]]:
    """表格排除区：整表框 + 物料表上方取值区（above_cells）上扩 + 联合外接矩形。"""
    regions: list[list[int]] = []
    raw_boxes: list[list[int]] = []
    pw, ph = max(1, int(page_w)), max(1, int(page_h))
    pad_i = int(max(0.0, pad))
    up = int(max(0.0, expand_up_frac) * ph)

    for inst in instances or []:
        if not isinstance(inst, dict):
            continue
        if not is_table_entity_id(inst.get("entity_id"), parse_kind=inst.get("parse_kind")):
            continue
        bbox = inst.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        fields = inst.get("fields") or {}
        eid = str(inst.get("entity_id") or "")
        read_mode = str(fields.get("read_mode") or "").lower()
        if read_mode == "above_cells" or "material" in eid:
            y1 = max(0, y1 - up)
        x1 = max(0, x1 - pad_i)
        y1 = max(0, y1 - pad_i)
        x2 = min(pw, x2 + pad_i)
        y2 = min(ph, y2 + pad_i)
        box = [x1, y1, x2, y2]
        raw_boxes.append(box)
        regions.append(box)

    if raw_boxes:
        ux1 = min(b[0] for b in raw_boxes)
        uy1 = min(b[1] for b in raw_boxes)
        ux2 = max(b[2] for b in raw_boxes)
        uy2 = max(b[3] for b in raw_boxes)
        regions.append([ux1, uy1, ux2, uy2])
    return regions


def text_matches_table_value(text: str, tokens: set[str] | None) -> bool:
    """OCR 文本是否与表格字段值相同/互为长子串（图号、材料代号等）。"""
    if not tokens:
        return False
    tn = _norm_table_token(text)
    if len(tn) < 4:
        return False
    for tok in tokens:
        if len(tok) < 4:
            continue
        if tn == tok:
            return True
        # 长字段：允许 OCR 截断/粘连
        if len(tok) >= 8 and len(tn) >= 6 and (tok in tn or tn in tok):
            return True
    return False


def filter_instances_matching_table_values(
    instances: list[dict[str, Any]] | None,
    tokens: set[str] | None,
    *,
    entity_ids: set[str] | frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    items = list(instances or [])
    if not tokens:
        return items, 0
    keep: list[dict[str, Any]] = []
    dropped = 0
    for inst in items:
        if not isinstance(inst, dict):
            continue
        eid = str(inst.get("entity_id") or "")
        if entity_ids is not None and eid not in entity_ids:
            keep.append(inst)
            continue
        text = str(inst.get("raw_text") or (inst.get("fields") or {}).get("text") or "")
        if text_matches_table_value(text, tokens):
            dropped += 1
            continue
        keep.append(inst)
    return keep, dropped


def filter_instances_outside_bboxes(
    instances: list[dict[str, Any]] | None,
    regions: list[list[int]] | None,
    *,
    pad: float = 0.0,
    entity_ids: set[str] | frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """丢掉中心点落在 regions 内的实例（默认仅处理数字标注类）。

    用于「非表格尺寸属性」：表格框内 OCR 数字不参与尺寸解析/重叠配对。
    返回 (保留列表, 剔除数量)。
    """
    items = list(instances or [])
    if not regions:
        return items, 0
    keep: list[dict[str, Any]] = []
    dropped = 0
    for inst in items:
        if not isinstance(inst, dict):
            continue
        eid = str(inst.get("entity_id") or "")
        if entity_ids is not None and eid not in entity_ids:
            keep.append(inst)
            continue
        center = bbox_center_xy(inst.get("bbox") or [])
        if center is None:
            keep.append(inst)
            continue
        cx, cy = center
        if any(point_in_bbox(cx, cy, r, pad=pad) for r in regions):
            dropped += 1
            continue
        keep.append(inst)
    return keep, dropped


def filter_dimension_marks_matching_overlap_pairs(
    instances: list[dict[str, Any]] | None,
    overlap_pairs: list[dict[str, Any]] | None,
    *,
    entity_ids: set[str] | frozenset[str] | None = None,
    iou_thr: float = 0.25,
) -> tuple[list[dict[str, Any]], int]:
    """丢掉与已确认重叠文本（keep_pair）空间重合的尺寸属性实例。

    匹配条件：IoU≥阈值，或尺寸框中心落在重叠框内。
    重叠实例本身不由此函数处理（应单独保留 keep_pair）。
    """
    from pipeline.perceive_utils import iou_xyxy

    items = list(instances or [])
    pairs = [
        p
        for p in (overlap_pairs or [])
        if isinstance(p, dict) and p.get("keep_pair") and p.get("bbox") and len(p.get("bbox") or []) == 4
    ]
    if not pairs:
        return items, 0
    eids = entity_ids
    keep: list[dict[str, Any]] = []
    dropped = 0
    for inst in items:
        if not isinstance(inst, dict):
            continue
        eid = str(inst.get("entity_id") or "")
        if eids is not None and eid not in eids:
            keep.append(inst)
            continue
        # 已是重叠证据则保留（一般不会出现在 VLM 列表）
        if inst.get("keep_pair"):
            keep.append(inst)
            continue
        bbox = inst.get("bbox")
        if not bbox or len(bbox) != 4:
            keep.append(inst)
            continue
        hit = False
        center = bbox_center_xy(bbox)
        for pair in pairs:
            pb = pair["bbox"]
            try:
                if iou_xyxy([float(x) for x in bbox], [float(x) for x in pb]) >= float(iou_thr):
                    hit = True
                    break
            except (TypeError, ValueError):
                pass
            if center is not None and point_in_bbox(center[0], center[1], pb, pad=0.0):
                hit = True
                break
        if hit:
            dropped += 1
            continue
        keep.append(inst)
    return keep, dropped


def _strip_code_fence(text: str) -> str:
    """去掉 ``` / ```json 围栏；兼容模型截断导致未闭合的情况。"""
    text = text.strip()
    closed = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if closed:
        return closed.group(1).strip()
    opened = re.search(r"```(?:json)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if opened:
        body = opened.group(1).strip()
        body = re.sub(r"```\s*$", "", body).strip()
        return body
    return text


def _raw_decode_objects(text: str) -> list[Any]:
    """从截断/杂讯文本中尽量捞出完整 JSON 对象（用于 Pass1 半截数组）。"""
    decoder = json.JSONDecoder()
    objs: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        i = end
    return objs


def extract_json_payload(text: str) -> Any:
    """从模型输出中提取 JSON（支持 ```json 块与截断半截数组）。

    空输出或无法解析时抛出 ValueError / JSONDecodeError。
    """
    if text is None:
        raise ValueError("空模型输出")
    text = str(text).strip()
    if not text:
        raise ValueError("空模型输出")
    # Qwen thinking / 杂讯包裹
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"<\|.*?\|>", "", text).strip()
    if not text:
        raise ValueError("空模型输出（清洗后）")
    text = _strip_code_fence(text)
    if not text:
        raise ValueError("空模型输出（代码块为空）")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 优先：截断数组里已完整的对象（避免误取单个 {} 丢掉同批其它实例）
        if "[" in text:
            objs = _raw_decode_objects(text)
            if objs:
                return objs
        # 尝试截取首尾括号
        for left, right in (("[", "]"), ("{", "}")):
            i, j = text.find(left), text.rfind(right)
            if i >= 0 and j > i:
                try:
                    return json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    continue
        objs = _raw_decode_objects(text)
        if objs:
            return objs if len(objs) > 1 else objs[0]
        raise


def extract_json_payload_safe(text: str, *, default: Any = None) -> Any:
    """extract_json_payload 的容错版：解析失败返回 default。"""
    try:
        return extract_json_payload(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        return default


def mock_perceive(plan: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """无权重时的演示感知：按计划生成占位 instances。

    若存在 work_dirs 旁路或 data 侧车 `{drawing_id}.mock.json`，可覆盖 fields/bbox。
    """
    from pathlib import Path

    from pipeline import ROOT, load_json

    w = int(meta.get("width") or 1000)
    h = int(meta.get("height") or 800)
    did = meta.get("drawing_id")
    override = None
    if did:
        for cand in (
            ROOT / "data" / "samples" / f"{did}.mock.json",
            ROOT / "data" / "gold" / f"{did}.mock.json",
        ):
            if cand.exists():
                override = load_json(cand)
                break

    instances = []
    for i, ent in enumerate(plan):
        eid = ent["entity_id"]
        ov_list = (override or {}).get("instances") or []
        ov = next((x for x in ov_list if x.get("entity_id") == eid), None)

        if eid == "title_block":
            bbox = [int(w * 0.55), int(h * 0.75), int(w * 0.98), int(h * 0.98)]
            defaults = {"part_no": "A-1001", "material": "45钢", "name": "示例零件"}
            fields = {
                f["name"]: defaults.get(f["name"], f"mock_{f['name']}")
                for f in ent.get("fields", [])
            }
            raw = "图号 A-1001 材料 45钢"
        elif eid == "info_table" or eid.endswith("_table") or ent.get("parse_kind") == "table":
            bbox = [int(w * 0.62), int(h * 0.70), int(w * 0.97), int(h * 0.96)]
            if eid == "material_table" or ent.get("section") == "material":
                bbox = [int(w * 0.62), int(h * 0.70), int(w * 0.97), int(h * 0.82)]
                defaults = {
                    "article_no": ["21940"],
                    "material_designation": ["-sheet DIN EN 13599 Cu-ETP-R290"],
                    "volume": ["7295.9"],
                    "surface": ["90.0"],
                    "weight": ["65.1"],
                }
            elif eid == "main_table" or ent.get("section") == "main":
                bbox = [int(w * 0.62), int(h * 0.82), int(w * 0.97), int(h * 0.96)]
                defaults = {
                    "component_name": "Contact lever (short)",
                    "document_number": "25001745002A",
                    "document_subclass": "Part drawing",
                    "page": "1/1",
                    "scale": "1:1",
                    "revision": "001",
                    "revision_state": "AA",
                    "tolerance_1": "ISO 8015",
                    "tolerance_2": "2768-H",
                }
            elif eid == "aux_table" or ent.get("section") == "aux" or str(ent.get("read_mode") or "") == "bbox_only":
                bbox = [int(w * 0.62), int(h * 0.62), int(w * 0.97), int(h * 0.68)]
                defaults = {}
            else:
                defaults = {
                    "article_no": "21940",
                    "volume": "7295.9",
                    "weight": "65.1",
                    "page": "1/1",
                    "scale": "1:1",
                    "document_number": "25001745002A",
                    "revision": "001",
                }
            fields = {
                f["name"]: defaults.get(f["name"])
                for f in ent.get("fields", [])
            }
            if str(ent.get("read_mode") or "") == "bbox_only" or ent.get("section") == "aux":
                fields = {"section": "aux", "read_mode": "bbox_only", "pairs": []}
            elif bool(ent.get("value_as_array")):
                for k, v in list(fields.items()):
                    if v is None:
                        continue
                    if not isinstance(v, list):
                        fields[k] = [str(v)]
                if bool(ent.get("extract_all_pairs", False)):
                    named = {str(f["name"]).lower() for f in ent.get("fields", [])}
                    extra = [
                        {"name": "MATERIAL", "content": "Steel"},
                        {"name": "SCALE", "content": "1:1"},
                    ]
                    fields["pairs"] = [p for p in extra if p["name"].lower() not in named]
                else:
                    fields["pairs"] = []
                fields["section"] = ent.get("section")
                fields["read_mode"] = ent.get("read_mode")
            else:
                if bool(ent.get("extract_all_pairs", False)):
                    named = {str(f["name"]).lower() for f in ent.get("fields", [])}
                    extra = [
                        {"name": "MATERIAL", "content": "Steel"},
                        {"name": "SCALE", "content": "1:1"},
                    ]
                    fields["pairs"] = [p for p in extra if p["name"].lower() not in named]
                else:
                    fields["pairs"] = []
                fields["section"] = ent.get("section")
                fields["read_mode"] = ent.get("read_mode")
            raw = f"mock {eid} section={ent.get('section')}"
        else:
            y0 = 40 + i * 80
            bbox = [40, y0, 200, y0 + 60]
            fields = {f["name"]: f"mock_{f['name']}" for f in ent.get("fields", [])}
            raw = " ".join(f"{k}={v}" for k, v in fields.items())

        if ov:
            if ov.get("bbox"):
                bbox = ov["bbox"]
            if "fields" in ov:
                fields = {**fields, **(ov.get("fields") or {})}
            if "raw_text" in ov:
                raw = ov["raw_text"]
            needs = bool(ov.get("needs_review", False))
            conf = float(ov.get("confidence", 0.5))
        else:
            needs, conf = True, 0.5

        instances.append(
            {
                "entity_id": eid,
                "instance_id": f"{eid}#0",
                "label": ent.get("locate_query", eid),
                "bbox": bbox,
                "fields": fields,
                "raw_text": raw,
                "confidence": conf,
                "needs_review": needs,
            }
        )
    return {"backend": "mock", "instances": instances}
