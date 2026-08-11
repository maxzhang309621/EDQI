"""非表格尺寸标注文本解析：类型（直径/半径/长度/角度）与基本尺寸/公差。"""
from __future__ import annotations

import re
from typing import Any

# 直径前缀（含常见 OCR 变体；O/Q/D 常为 Ø 误读；不用裸 0 开头小数）
_DIAM_PREFIX = re.compile(
    r"^[\s]*(?:Ø|⌀|Ф|ф|Φ|φ|ø|∅|[OoQqDd])(?=[\s]*\d)",
)
# 半径：R/r 后接数字（允许空格）；排除 Rz/Ra 粗糙度
_RADIUS_PREFIX = re.compile(r"^[\s]*[Rr](?![\s]*[azhcfAZHCF])(?=[\s]*\d)")
# 角度后缀（° 常被 OCR 成 o/O）
_ANGLE_SUFFIX = re.compile(r"(?:°|º|deg\.?|(?<=\d)[oO])\s*$", re.IGNORECASE)
_ANGLE_INLINE = re.compile(r"(?:°|º|deg\.?|(?<=\d)[oO])(?=\s*[±+\-]|\s*$)", re.IGNORECASE)
# ± 或 +/-
_PLUS_MINUS = re.compile(r"(?:±|\+/-|/\+-\s*)")
# 抽取数字（含小数）
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# 单独检出的直径/角度符号（OCR 常与数字拆成两框；含常见误读）
_DIM_SYMBOL_ONLY = re.compile(
    r"^(?:Ø|⌀|Ф|ф|Φ|φ|ϕ|ø|∅|˚|°|º|ₒ|[OoQqDd]|¢)$"
)

# 非尺寸属性词头：Max.3 / TYP 5 / MIN 0.2 等（带数字但不是 Ø/R/长度/角度标注）
_NON_DIM_WORD_PREFIX = re.compile(
    r"(?i)^\s*(?:"
    r"max|min|typ(?:ical)?|ref(?:erence)?|approx(?:\.|imate)?|"
    r"nom(?:inal)?|basic|see|note|notes?|item|qty|quantity|"
    r"thru|through|equal|eq\.?|u\.?\s*o\.?\s*s\.?|"
    r"chamfer|break\s*edge|spot\s*face|drill|tap|deep|thk|thick(?:ness)?"
    r")\.?\s*"
)


def has_non_dimension_word_prefix(text: str) -> bool:
    """Max./MIN/TYP 等未定义属性词头（即使后面有数字也应排除）。"""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_NON_DIM_WORD_PREFIX.match(t))


def _normalize_text(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("，", ",").replace("．", ".")
    # 常见直径符号 OCR 归一
    for ch in ("⌀", "∅", "Ф", "ф", "Φ", "φ", "ϕ", "ø", "¢"):
        t = t.replace(ch, "Ø")
    for ch in ("˚", "º", "ₒ"):
        t = t.replace(ch, "°")
    # OCR 常把 ± 拆成 + / - 或 +-
    t = re.sub(r"\+\s*/\s*-", "±", t)
    t = re.sub(r"\+\s*-\s*", "±", t)
    # 45o / 45O → 45°（仅数字后的拉丁 o）
    t = re.sub(r"(?<=\d)[oO](?=\s*[±+\-]|\s*$)", "°", t)
    # O12 / Q12 / D12（紧贴数字）→ Ø12；避免匹配 Rz
    t = re.sub(r"(?i)^([OoQqDd])(?=\s*\d)", "Ø", t)
    # "o 12" / "O 12" 间距形式
    t = re.sub(r"(?i)^([OoQqDd])\s+(?=\d)", "Ø", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_ocr_dimension_text(text: str) -> str:
    """对外暴露的 OCR 尺寸文本归一（符号变体 → 标准 Ø/°/±）。"""
    return _normalize_text(text)


def is_dim_symbol_only(text: str) -> bool:
    """是否为单独直径/角度符号框（无数字）。"""
    t = (text or "").strip()
    return bool(t) and bool(_DIM_SYMBOL_ONLY.match(t))


def _normalize_number_token(tok: str) -> str:
    """保留原始小数形态，仅统一小数点为 '.'。"""
    return (tok or "").strip().replace(",", ".")


def _split_basic_and_tolerance(body: str) -> tuple[str | None, str | None, bool]:
    """从去掉类型前缀/后缀后的主体中拆基本尺寸与公差。"""
    body = (body or "").strip()
    if not body:
        return None, None, False
    m = _PLUS_MINUS.search(body)
    if not m:
        nums = _NUM.findall(body)
        if not nums:
            return None, None, False
        return _normalize_number_token(nums[0]), None, False
    left, right = body[: m.start()], body[m.end() :]
    left_nums = _NUM.findall(left)
    right_nums = _NUM.findall(right)
    basic = _normalize_number_token(left_nums[0]) if left_nums else None
    tol = _normalize_number_token(right_nums[0]) if right_nums else None
    if basic is None and tol is None:
        return None, None, False
    # 有 ± 即标明公差属性（即便右侧 OCR 失败也记 has_tolerance）
    return basic, tol, True


def parse_dimension_text(text: str) -> dict[str, Any]:
    """解析尺寸标注文本。

    返回:
      dim_kind: diameter|radius|length|angle|null
      basic_size: str|null
      tolerance: str|null
      has_tolerance: bool
    """
    raw = (text or "").strip()
    t = _normalize_text(raw)
    empty = {
        "dim_kind": None,
        "basic_size": None,
        "tolerance": None,
        "has_tolerance": False,
    }
    if not t:
        return empty
    # Max.3 / TYP 5 等：不是尺寸标注
    if has_non_dimension_word_prefix(raw) or has_non_dimension_word_prefix(t):
        return empty
    # 纯符号：无法定尺寸，留给邻近合并 / VLM
    if is_dim_symbol_only(t) or (not re.search(r"\d", t)):
        return empty

    dim_kind: str | None = None
    body = t

    # 角度：° 可在数值后、± 前（如 45°±1），不要求整串以 ° 结尾
    if _ANGLE_SUFFIX.search(body) or _ANGLE_INLINE.search(body):
        dim_kind = "angle"
        body = _ANGLE_INLINE.sub("", body)
        body = _ANGLE_SUFFIX.sub("", body).strip()
    elif _DIAM_PREFIX.match(body):
        dim_kind = "diameter"
        body = _DIAM_PREFIX.sub("", body, count=1).strip()
    elif _RADIUS_PREFIX.match(body):
        dim_kind = "radius"
        body = _RADIUS_PREFIX.sub("", body, count=1).strip()
    else:
        dim_kind = "length"

    basic, tol, has_tol = _split_basic_and_tolerance(body)
    if basic is None and tol is None and dim_kind is not None:
        # 主体拆不出数字时仍保留类型，尝试整串取首个数字
        nums = _NUM.findall(t)
        if nums:
            basic = _normalize_number_token(nums[0])
        else:
            dim_kind = None

    return {
        "dim_kind": dim_kind,
        "basic_size": basic,
        "tolerance": tol,
        "has_tolerance": bool(has_tol),
    }


def merge_dimension_fields(fields: dict[str, Any] | None, text: str) -> dict[str, Any]:
    """把解析结果合并进 fields（保留原有键）。"""
    out = dict(fields or {})
    parsed = parse_dimension_text(text if text is not None else str(out.get("text") or ""))
    out.update(parsed)
    return out


_DIM_FIELD_KEYS = ("dim_kind", "basic_size", "tolerance", "has_tolerance")


def apply_dimension_parse_to_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为实例补充尺寸属性；已确认重叠（keep_pair）的只标跳过，不做属性解析。"""
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("keep_pair"):
            fields = dict(inst.get("fields") or {})
            for k in _DIM_FIELD_KEYS:
                fields.pop(k, None)
            fields["dimension_parse_skipped"] = "overlap"
            inst["fields"] = fields
            continue
        text_v = str(inst.get("raw_text") or (inst.get("fields") or {}).get("text") or "")
        inst["fields"] = merge_dimension_fields(inst.get("fields") or {}, text_v)
    return instances


def enrich_dimension_fields_from_text(
    fields: dict[str, Any] | None,
    text: str | None = None,
    *,
    bbox: list[Any] | None = None,
) -> dict[str, Any]:
    """用文本规则补齐空缺尺寸字段；不覆盖已有非空值。缺 angle 时用 bbox 粗估。"""
    from pipeline.text_angle import text_angle_from_bbox

    out = dict(fields or {})
    text_v = text if text is not None else str(out.get("text") or "")
    if text_v and not out.get("text"):
        out["text"] = text_v
    parsed = parse_dimension_text(text_v)
    for key in _DIM_FIELD_KEYS:
        cur = out.get(key)
        if cur is None or cur == "":
            out[key] = parsed.get(key)
    if out.get("angle") is None or out.get("angle") == "":
        out["angle"] = text_angle_from_bbox(bbox)
    else:
        try:
            out["angle"] = float(out["angle"])
        except (TypeError, ValueError):
            out["angle"] = text_angle_from_bbox(bbox)
    return out


_ALLOWED_DIM_KINDS = frozenset({"diameter", "radius", "length", "angle"})


def clip_fields_to_schema(
    fields: dict[str, Any] | None,
    allowed_names: list[str] | set[str] | frozenset[str] | None,
) -> dict[str, Any]:
    """只保留配置中声明的字段名。"""
    out = dict(fields or {})
    if not allowed_names:
        return out
    allow = set(allowed_names)
    return {k: v for k, v in out.items() if k in allow}


def is_valid_dimension_mark(
    fields: dict[str, Any] | None,
    *,
    text: str | None = None,
    require_dim_kind: bool = True,
    require_basic_size: bool = True,
) -> bool:
    """是否为配置允许的尺寸属性（非图号/材料/粗糙度/表值等杂讯）。"""
    f = fields or {}
    text_v = str(text if text is not None else f.get("text") or "").strip()
    kind = str(f.get("dim_kind") or "").strip().lower() or None
    if kind == "null":
        kind = None
    basic = f.get("basic_size")
    if basic is not None:
        basic = str(basic).strip() or None

    if not text_v or text_v in {".", "——", "-", "–", "—"}:
        return False
    # Max.3 / MIN / TYP 等未定义属性（即使带数字也排除）
    if has_non_dimension_word_prefix(text_v):
        return False
    # 比例 5:1 等非尺寸
    if re.fullmatch(r"\d+\s*[:：]\s*\d+", text_v):
        return False
    # 表面粗糙度 / 视图字母 / 气泡序号 / 残片
    if re.match(r"(?i)^R[azhcf]\d", text_v):
        return False
    if re.fullmatch(r"[A-Za-z]{1,2}", text_v):
        return False
    if re.fullmatch(r"\(\d+\)", text_v):
        return False
    if re.fullmatch(r"[+]?\d+\.$", text_v) or re.fullmatch(r"[±+\-]\s*\d+(?:[.,]\d+)?", text_v):
        return False
    if re.search(r"(?i)图中无|无尺寸|not\s*a\s*dim", text_v):
        return False
    # 字母+数字但非 R/Ø 尺寸（如 4G、M8x1 以外的杂讯词头已在 prefix 处理）
    if re.match(r"(?i)^[A-Za-z]{2,}\.?\s*\d", text_v) and not re.match(
        r"(?i)^(?:Ø|R|M)\s*\d", _normalize_text(text_v)
    ):
        return False

    if require_dim_kind and kind not in _ALLOWED_DIM_KINDS:
        # 尝试从文本再解析一次
        parsed = parse_dimension_text(text_v)
        kind = parsed.get("dim_kind")
        if basic is None:
            basic = parsed.get("basic_size")
        if kind not in _ALLOWED_DIM_KINDS:
            return False
    if require_basic_size and not basic:
        parsed = parse_dimension_text(text_v)
        basic = parsed.get("basic_size")
        if not basic:
            return False
    # 拒绝明显非尺寸长串
    if len(text_v) > 32:
        return False
    if re.search(r"(?i)(?:din\s*en|sheet|cu[\s\-]?etp|siemens|material|article)", text_v):
        return False
    if re.match(r"^\d{7,}", text_v) and kind == "length" and "±" not in text_v and "°" not in text_v:
        return False
    # 表格体积/重量量级：无公差的超大长度值
    try:
        basic_f = float(str(basic).replace(",", "."))
        if kind == "length" and basic_f >= 1000 and "±" not in text_v and "°" not in text_v:
            return False
    except (TypeError, ValueError):
        # basic 非数字（如误读成字母）
        if kind in {"length", "diameter", "radius", "angle"} and not re.search(r"\d", str(basic or "")):
            return False
    return True


def bbox_geometry_ok(
    bbox: list[Any] | None,
    *,
    page_w: int,
    page_h: int,
    max_width_ratio: float = 0.22,
    max_aspect_ratio: float = 8.0,
    min_side: float = 4.0,
    max_height_ratio: float = 0.12,
    max_area_ratio: float = 0.035,
    max_side_ratio: float = 0.22,
) -> bool:
    """过滤整幅/半幅/分块级假框（仅允许紧贴尺寸文字的小框）。"""
    if not bbox or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if bw < min_side or bh < min_side:
        return False
    pw, ph = max(1, int(page_w)), max(1, int(page_h))
    if bw > max_width_ratio * pw:
        return False
    if bh > max_height_ratio * ph:
        return False
    if max(bw, bh) > max_side_ratio * max(pw, ph):
        return False
    if (bw * bh) > max_area_ratio * pw * ph:
        return False
    ar = bw / max(bh, 1.0)
    if ar > max_aspect_ratio:
        return False
    # 竖排允许较高，但高宽比也受限
    if bh / max(bw, 1.0) > max_aspect_ratio:
        return False
    return True


def prescreen_ocr_dimension_candidates(
    instances: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    require_dim_kind: bool = True,
    require_basic_size: bool = True,
    max_bbox_width_ratio: float = 0.22,
    max_aspect_ratio: float = 8.0,
    max_height_ratio: float = 0.12,
    max_area_ratio: float = 0.035,
    max_candidates: int | None = None,
    exclude_bboxes: list[list[Any]] | None = None,
    exclude_pad: float = 0.0,
    exclude_table_texts: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """OCR 属性初筛：保留 keep_pair；非重叠实例按尺寸规则过滤后作为 VLM 候选。

    返回 (candidates, keep_pairs, dropped_non_pair)。
    """
    from pipeline.perceive_common import (
        bbox_center_xy,
        point_in_bbox,
        text_matches_table_value,
    )

    keep_pairs: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    dropped = 0
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("keep_pair"):
            # 重叠证据也丢弃半页级假框，避免污染可视化
            if not bbox_geometry_ok(
                inst.get("bbox"),
                page_w=page_w,
                page_h=page_h,
                max_width_ratio=max(max_bbox_width_ratio, 0.35),
                max_aspect_ratio=max(max_aspect_ratio, 12.0),
                max_height_ratio=0.25,
                max_area_ratio=0.08,
            ):
                dropped += 1
                continue
            keep_pairs.append(inst)
            continue
        fields = dict(inst.get("fields") or {})
        text_v = str(inst.get("raw_text") or fields.get("text") or "")
        fields = enrich_dimension_fields_from_text(fields, text_v, bbox=inst.get("bbox"))
        if not bbox_geometry_ok(
            inst.get("bbox"),
            page_w=page_w,
            page_h=page_h,
            max_width_ratio=max_bbox_width_ratio,
            max_aspect_ratio=max_aspect_ratio,
            max_height_ratio=max_height_ratio,
            max_area_ratio=max_area_ratio,
        ):
            dropped += 1
            continue
        if exclude_bboxes:
            center = bbox_center_xy(inst.get("bbox") or [])
            if center and any(
                point_in_bbox(center[0], center[1], bb, pad=float(exclude_pad))
                for bb in exclude_bboxes
            ):
                dropped += 1
                continue
        if exclude_table_texts and text_matches_table_value(text_v, exclude_table_texts):
            dropped += 1
            continue
        if not is_valid_dimension_mark(
            fields,
            text=text_v,
            require_dim_kind=require_dim_kind,
            require_basic_size=require_basic_size,
        ):
            dropped += 1
            continue
        out = dict(inst)
        out["fields"] = {**fields, "ocr_prescreen": True}
        if fields.get("text") and not out.get("raw_text"):
            out["raw_text"] = fields.get("text")
        if fields.get("angle") is not None:
            out["angle"] = fields.get("angle")
        candidates.append(out)

    if max_candidates is not None and max_candidates >= 0 and len(candidates) > max_candidates:
        # 优先高置信度；同置信度保留原序
        ranked = sorted(
            enumerate(candidates),
            key=lambda t: (-float(t[1].get("confidence") or 0.0), t[0]),
        )
        keep_idx = {i for i, _ in ranked[:max_candidates]}
        dropped += len(candidates) - max_candidates
        candidates = [c for i, c in enumerate(candidates) if i in keep_idx]

    return candidates, keep_pairs, dropped


def sanitize_number_mark_instances(
    instances: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    allowed_field_names: list[str] | set[str] | frozenset[str] | None = None,
    require_dim_kind: bool = True,
    require_basic_size: bool = True,
    max_bbox_width_ratio: float = 0.22,
    max_aspect_ratio: float = 8.0,
    max_height_ratio: float = 0.12,
    max_area_ratio: float = 0.035,
    exclude_bboxes: list[list[Any]] | None = None,
    exclude_pad: float = 0.0,
    exclude_table_texts: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """最终清理：非 keep_pair 的 number_mark 必须是合法尺寸小框；大范围假框一律丢弃。"""
    from pipeline.perceive_common import bbox_center_xy, point_in_bbox, text_matches_table_value

    mark_ids = {"number_mark", "annotation", "annotations"}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for inst in instances or []:
        if not isinstance(inst, dict):
            continue
        eid = str(inst.get("entity_id") or "")
        if eid not in mark_ids:
            kept.append(inst)
            continue

        geom_ok = bbox_geometry_ok(
            inst.get("bbox"),
            page_w=page_w,
            page_h=page_h,
            max_width_ratio=max_bbox_width_ratio if not inst.get("keep_pair") else max(max_bbox_width_ratio, 0.35),
            max_aspect_ratio=max_aspect_ratio if not inst.get("keep_pair") else max(max_aspect_ratio, 12.0),
            max_height_ratio=max_height_ratio if not inst.get("keep_pair") else 0.25,
            max_area_ratio=max_area_ratio if not inst.get("keep_pair") else 0.08,
        )
        if not geom_ok:
            dropped += 1
            continue

        if inst.get("keep_pair"):
            kept.append(inst)
            continue

        fields = dict(inst.get("fields") or {})
        text_v = str(inst.get("raw_text") or fields.get("text") or "")
        if exclude_bboxes:
            center = bbox_center_xy(inst.get("bbox") or [])
            if center and any(
                point_in_bbox(center[0], center[1], bb, pad=float(exclude_pad))
                for bb in exclude_bboxes
            ):
                dropped += 1
                continue
        if exclude_table_texts and text_matches_table_value(text_v, exclude_table_texts):
            dropped += 1
            continue
        fields = enrich_dimension_fields_from_text(fields, text_v, bbox=inst.get("bbox"))
        if allowed_field_names:
            fields = clip_fields_to_schema(fields, allowed_field_names)
        if not is_valid_dimension_mark(
            fields,
            text=text_v,
            require_dim_kind=require_dim_kind,
            require_basic_size=require_basic_size,
        ):
            dropped += 1
            continue
        out = dict(inst)
        out["fields"] = fields
        if fields.get("text"):
            out["raw_text"] = fields.get("text")
        if fields.get("angle") is not None:
            out["angle"] = fields.get("angle")
        kept.append(out)
    return kept, dropped


def _attr_rank(inst: dict[str, Any]) -> tuple:
    """去重保留优先级：更具体类型 > 带公差 > VLM确认 > 高置信度。"""
    fields = inst.get("fields") or {}
    kind = str(fields.get("dim_kind") or "").lower()
    kind_score = {"diameter": 4, "radius": 4, "angle": 4, "length": 1}.get(kind, 0)
    has_tol = 1 if fields.get("has_tolerance") or fields.get("tolerance") not in (None, "") else 0
    vlm = 1 if fields.get("vlm_filtered") else 0
    conf = float(inst.get("confidence") or 0)
    text = str(inst.get("raw_text") or fields.get("text") or "")
    return (kind_score, has_tol, vlm, conf, len(text))


def dedupe_dimension_attribute_instances(
    instances: list[dict[str, Any]],
    *,
    iou_thr: float = 0.40,
    center_dist_ratio: float = 0.75,
) -> tuple[list[dict[str, Any]], int]:
    """去掉重复尺寸属性（同位置/同基本尺寸近邻只留一条）。keep_pair / 非 number_mark 不动。"""
    from pipeline.perceive_utils import iou_xyxy

    mark_ids = {"number_mark", "annotation", "annotations"}
    others: list[dict[str, Any]] = []
    attrs: list[dict[str, Any]] = []
    for inst in instances or []:
        if not isinstance(inst, dict):
            continue
        eid = str(inst.get("entity_id") or "")
        if eid not in mark_ids or inst.get("keep_pair"):
            others.append(inst)
            continue
        attrs.append(inst)

    attrs_sorted = sorted(attrs, key=_attr_rank, reverse=True)
    kept_attrs: list[dict[str, Any]] = []
    dropped = 0

    def _basic(inst: dict[str, Any]) -> str:
        f = inst.get("fields") or {}
        b = f.get("basic_size")
        if b is None or b == "":
            return ""
        return str(b).strip().replace(",", ".")

    def _center(b: list[Any]) -> tuple[float, float] | None:
        if not b or len(b) != 4:
            return None
        return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)

    for cand in attrs_sorted:
        cb = cand.get("bbox")
        if not cb:
            dropped += 1
            continue
        cc = _center(cb)
        cw = max(1.0, float(cb[2]) - float(cb[0]))
        ch = max(1.0, float(cb[3]) - float(cb[1]))
        c_basic = _basic(cand)
        dup = False
        for s in kept_attrs:
            sb = s.get("bbox")
            if not sb:
                continue
            if iou_xyxy([float(x) for x in cb], [float(x) for x in sb]) >= iou_thr:
                dup = True
                break
            sc = _center(sb)
            if cc and sc:
                dist = ((cc[0] - sc[0]) ** 2 + (cc[1] - sc[1]) ** 2) ** 0.5
                ref = max(cw, ch, float(sb[2] - sb[0]), float(sb[3] - sb[1]), 8.0)
                same_basic = bool(c_basic) and c_basic == _basic(s)
                if same_basic and dist <= center_dist_ratio * ref:
                    dup = True
                    break
                # 文本几乎相同且中心很近
                ct = str(cand.get("raw_text") or (cand.get("fields") or {}).get("text") or "").strip()
                st = str(s.get("raw_text") or (s.get("fields") or {}).get("text") or "").strip()
                if ct and st and ct == st and dist <= center_dist_ratio * ref:
                    dup = True
                    break
        if dup:
            dropped += 1
            continue
        kept_attrs.append(cand)

    return others + kept_attrs, dropped
