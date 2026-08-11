"""非表格尺寸标注文本解析：类型（直径/半径/长度/角度）与基本尺寸/公差。"""
from __future__ import annotations

import re
from typing import Any

# 直径前缀（含常见 OCR 变体；不用裸 O/0，避免误伤普通数字）
_DIAM_PREFIX = re.compile(r"^[\s]*(?:Ø|⌀|Ф|ф|Φ|φ|ø)", re.IGNORECASE)
# 半径：R/r 后接数字（允许空格）
_RADIUS_PREFIX = re.compile(r"^[\s]*[Rr](?=[\s]*\d)")
# 角度后缀
_ANGLE_SUFFIX = re.compile(r"(?:°|º|deg\.?)\s*$", re.IGNORECASE)
# ± 或 +/-
_PLUS_MINUS = re.compile(r"(?:±|\+/-|/\+-\s*)")
# 抽取数字（含小数）
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _normalize_text(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("，", ",").replace("．", ".")
    # OCR 常把 ± 拆成 + / - 或 +-
    t = re.sub(r"\+\s*/\s*-", "±", t)
    t = re.sub(r"\+\s*-\s*", "±", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


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
    if not t or not re.search(r"\d", t):
        return empty

    dim_kind: str | None = None
    body = t

    # 角度：° 可在数值后、± 前（如 45°±1），不要求整串以 ° 结尾
    if _ANGLE_SUFFIX.search(body) or re.search(r"(?:°|º|deg\.?)", body, re.IGNORECASE):
        dim_kind = "angle"
        body = re.sub(r"(?:°|º|deg\.?)", "", body, flags=re.IGNORECASE).strip()
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
    """是否为配置允许的尺寸属性（非图号/材料等杂讯）。"""
    f = fields or {}
    text_v = str(text if text is not None else f.get("text") or "").strip()
    kind = str(f.get("dim_kind") or "").strip().lower() or None
    if kind == "null":
        kind = None
    basic = f.get("basic_size")
    if basic is not None:
        basic = str(basic).strip() or None

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
    return True


def bbox_geometry_ok(
    bbox: list[Any] | None,
    *,
    page_w: int,
    page_h: int,
    max_width_ratio: float = 0.28,
    max_aspect_ratio: float = 8.0,
    min_side: float = 4.0,
) -> bool:
    """过滤整幅/半幅细长假框。"""
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
    if bh > 0.35 * ph:
        return False
    ar = bw / max(bh, 1.0)
    if ar > max_aspect_ratio:
        return False
    return True
