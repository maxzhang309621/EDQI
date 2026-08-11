"""标题栏物料表/主表：OCR 定位 Product 行分界 + 物料表「标签上方取值」。"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from PIL import Image


def _quad_to_xyxy(box) -> list[int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _norm(s: str) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"\[.*?\]", " ", t)
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _aliases(parse_hint: str, name: str) -> list[str]:
    """只保留短标签别名；截掉中文说明/禁止项，避免误匹配其它列。"""
    hint = str(parse_hint or "")
    # 说明文字通常从第一个汉字开始
    cjk = re.search(r"[\u4e00-\u9fff]", hint)
    if cjk:
        hint = hint[: cjk.start()]
    hint = re.split(r"[；;]", hint)[0]
    parts = re.split(r"[/|]", hint)
    aliases: list[str] = []
    for p in parts:
        a = _norm(p)
        a = re.sub(r"\s*\[.*?\]\s*", " ", a).strip()
        a = re.sub(r"[.:：]+$", "", a).strip()
        if len(a) < 2 or len(a) > 48:
            continue
        aliases.append(a)
    if name:
        aliases.append(_norm(name.replace("_", " ")))
    # 常见缩写补全
    extras = {
        "article_no": ["article no", "article number"],
        "additional_material_information": [
            "additional material information",
            "additional material info",
        ],
        "material_designation": ["material designation", "-sheet"],
        "volume": ["volume"],
        "surface": ["surface"],
        "weight": ["weight", "mass", "veight", "eight"],
        "surface": ["surface", "surfoce", "sur face"],
    }
    for a in extras.get(name, []):
        aliases.append(_norm(a))
    seen = set()
    out: list[str] = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _x_overlap_ratio(a: list[int], b: list[int]) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    if inter <= 0:
        return 0.0
    wa = max(1, ax2 - ax1)
    wb = max(1, bx2 - bx1)
    return inter / float(min(wa, wb))


def run_ocr_boxes(
    image: Image.Image,
    conf_th: float = 0.35,
    *,
    enhance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """返回 [{text, score, bbox}]，bbox 为相对 image 的 xyxy。

    enhance（仅物料表小字用，不影响数字重叠 OCR）：
      scale: 放大倍数（默认 1）
      border: 白边像素（默认 0；小字标题栏建议 16~24）
      contrast: 对比度（默认 1.0）
    """
    try:
        from pipeline.ocr_engine import get_rapid_ocr
    except ImportError:
        return []
    try:
        engine = get_rapid_ocr()
    except ImportError:
        return []

    from PIL import ImageEnhance, ImageOps

    enh = enhance or {}
    scale = float(enh.get("scale") or 1.0)
    border = int(enh.get("border") or 0)
    contrast = float(enh.get("contrast") or 1.0)

    rgb = image.convert("RGB")
    if border > 0:
        rgb = ImageOps.expand(rgb, border=border, fill=(255, 255, 255))
    if scale > 1.0 + 1e-6:
        w, h = rgb.size
        rgb = rgb.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    if abs(contrast - 1.0) > 1e-6:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)

    result, _ = engine(np.array(rgb))
    items = []
    for row in result or []:
        if not row or len(row) < 3:
            continue
        box, text, score = row[0], str(row[1]), float(row[2])
        if score < conf_th:
            continue
        t = (text or "").strip()
        if not t:
            continue
        bb = _quad_to_xyxy(box)
        # 映射回原始 crop 坐标
        if scale > 1.0 + 1e-6:
            bb = [int(round(v / scale)) for v in bb]
        if border > 0:
            bb = [bb[0] - border, bb[1] - border, bb[2] - border, bb[3] - border]
        items.append({"text": t, "score": score, "bbox": bb})
    return items


def union_bbox(boxes: list[list[int]]) -> list[int] | None:
    valid = [b for b in boxes if b and len(b) == 4 and b[2] > b[0] and b[3] > b[1]]
    if not valid:
        return None
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def split_title_block_by_product(
    page_image: Image.Image,
    seed_boxes: list[list[int]],
    *,
    conf_th: float = 0.35,
    pad: int = 12,
    search_up_pad: int = 40,
) -> tuple[list[int] | None, list[int] | None, list[str]]:
    """以 Product 行分界：以上=物料表，Product 行及以下=主表。

    物料框高度保持种子/VL 顶边，不人为加高（小字靠 OCR enhance 解决）。
    """
    notes: list[str] = []
    uni = union_bbox(seed_boxes)
    if uni is None:
        notes.append("product_split:no_seed_bbox")
        return None, None, notes

    w, h = page_image.size
    x1 = max(0, uni[0] - pad)
    x2 = min(w, uni[2] + pad)
    y2 = min(h, uni[3] + pad)
    # 仅搜索区略向上，便于找到 Product；最终物料顶边仍用种子顶
    seed_y1 = max(0, uni[1] - pad)
    search_y1 = max(0, uni[1] - max(pad, search_up_pad))
    crop = page_image.crop((x1, search_y1, x2, y2))
    items = run_ocr_boxes(crop, conf_th=conf_th)
    if not items:
        notes.append("product_split:ocr_empty")
        return None, None, notes

    pat = re.compile(r"\bproduct\b", re.I)
    candidates = []
    for it in items:
        if not pat.search(it["text"]):
            continue
        if len(it["text"]) > 48:
            continue
        candidates.append(it)
    if not candidates:
        notes.append("product_split:product_not_found")
        return None, None, notes

    candidates.sort(key=lambda it: it["bbox"][1])
    chosen = candidates[-1]
    prod_y_local = int(chosen["bbox"][1])
    prod_text = chosen["text"]
    prod_y = search_y1 + prod_y_local
    notes.append(f"product_split:row={prod_text!r},y={prod_y}")

    mat_y1 = min(seed_y1, max(0, prod_y - 4))
    mat = [x1, mat_y1, x2, max(mat_y1 + 4, prod_y)]
    main = [x1, prod_y, x2, y2]
    notes.append(f"product_split:material_bbox={mat}")
    if mat[3] - mat[1] < 20:
        notes.append("product_split:material_too_thin")
    if main[3] - main[1] < 40:
        notes.append("product_split:main_too_thin")
    return mat, main, notes


def _match_label(item_text: str, aliases: list[str]) -> float:
    t = _norm(item_text)
    if not t:
        return 0.0
    best = 0.0
    for a in aliases:
        if not a:
            continue
        if t == a:
            return 1.0
        # 短别名（如 mass/surface）禁止子串误伤其它长标签
        if len(a) <= 8:
            tw = re.findall(r"[a-z0-9\u4e00-\u9fff]+", t)
            aw = re.findall(r"[a-z0-9\u4e00-\u9fff]+", a)
            if aw and tw == aw:
                best = max(best, 0.95)
            elif aw and len(aw) == 1 and aw[0] in tw and len(tw) <= 2:
                # "surface" ≈ "surface mm2"；拒绝 "additional material information"
                best = max(best, 0.9)
            else:
                # OCR 错字：surfoce≈surface, veight≈weight, rolume≈volume
                ratio = SequenceMatcher(None, t.replace(" ", ""), a.replace(" ", "")).ratio()
                if ratio >= 0.72 and len(a) >= 4:
                    best = max(best, 0.82)
            continue
        if a in t or t in a:
            best = max(best, 0.85)
        tw = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", t))
        aw = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", a))
        if tw and aw and aw.issubset(tw) and len(aw) >= 2:
            best = max(best, 0.8)
        ratio = SequenceMatcher(None, t, a).ratio()
        if ratio >= 0.75:
            best = max(best, 0.8 + 0.15 * (ratio - 0.75))
    return best


def extract_fields_above_labels(
    page_image: Image.Image,
    table_bbox: list[int],
    fields: list[dict[str, Any]],
    *,
    conf_th: float = 0.35,
    col_overlap: float = 0.25,
    max_above_gap: int = 100,
    ocr_enhance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """物料表：先定位字段标签，再收集该标签正上方同列文本；无则 null。

    不改 bbox 几何；小字通过 ocr_enhance（加边/放大/对比度）识别。
    """
    notes: list[str] = []
    x1, y1, x2, y2 = [int(v) for v in table_bbox]
    w, h = page_image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {}, "", ["above_cells:invalid_bbox"]

    # 默认小字增强（仅本路径；数字重叠 OCR 不走这里）
    # crop_pad：只扩大 OCR 读区，不改 facts/结果图里的表格 bbox
    enh = {
        "scale": 4.0,
        "border": 20,
        "contrast": 1.8,
        "crop_pad": 16,
        **(ocr_enhance or {}),
    }
    use_conf = float(enh.get("confidence_threshold") or conf_th)
    crop_pad = max(0, int(enh.get("crop_pad") or 0))
    ox1 = max(0, x1 - crop_pad)
    oy1 = max(0, y1 - crop_pad)
    ox2 = min(w, x2 + crop_pad)
    oy2 = min(h, y2 + crop_pad)
    crop = page_image.crop((ox1, oy1, ox2, oy2))
    items = run_ocr_boxes(crop, conf_th=use_conf, enhance=enh)
    notes.append(
        f"above_cells:ocr_items={len(items)} "
        f"enhance=scale:{enh.get('scale')},border:{enh.get('border')},crop_pad:{crop_pad}"
    )
    if not items:
        fields_out = {f["name"]: None for f in fields}
        fields_out["pairs"] = []
        return fields_out, "", notes

    for it in items:
        b = it["bbox"]
        # 映射到页坐标（相对扩边后的 crop）
        it["pb"] = [b[0] + ox1, b[1] + oy1, b[2] + ox1, b[3] + oy1]

    # 全部字段别名，用于排除「上方也是标签」
    all_aliases: list[str] = []
    for f in fields:
        all_aliases.extend(_aliases(str(f.get("parse_hint") or ""), f["name"]))

    # 1) 先为每个字段定位标签（各用唯一 OCR 框）
    used_label_ids: set[int] = set()
    field_labels: dict[str, dict[str, Any]] = {}
    out: dict[str, Any] = {}
    raw_parts: list[str] = []

    for f in fields:
        name = f["name"]
        aliases = _aliases(str(f.get("parse_hint") or ""), name)
        best_i, best_score = -1, 0.0
        for i, it in enumerate(items):
            if i in used_label_ids:
                continue
            sc = _match_label(it["text"], aliases)
            if sc < 0.8:
                continue
            if sc > best_score or (
                sc == best_score and best_i >= 0 and it["pb"][1] > items[best_i]["pb"][1]
            ):
                best_score, best_i = sc, i
        if best_i < 0:
            out[name] = None
            notes.append(f"above_cells:label_miss:{name}")
            continue
        used_label_ids.add(best_i)
        field_labels[name] = {"idx": best_i, "item": items[best_i]}
        raw_parts.append(f"{name}={items[best_i]['text']}")

    # 2) 内容格只归属其「正下方、同列、最近」的标签（避免上层空列误吸下层数值）
    assigned: dict[str, list[tuple[int, str]]] = {n: [] for n in field_labels}
    label_ids = {meta["idx"] for meta in field_labels.values()}
    for j, it in enumerate(items):
        if j in label_ids:
            continue
        if _match_label(it["text"], all_aliases) >= 0.8:
            continue
        pb = it["pb"]
        best_name, best_gap = None, None
        for name, meta in field_labels.items():
            lb = meta["item"]["pb"]
            if pb[3] > lb[1] + 2:
                continue
            gap = lb[1] - pb[3]
            if gap < 0 or gap > max_above_gap:
                continue
            if _x_overlap_ratio(lb, pb) < col_overlap:
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best_name = gap, name
        if best_name is not None:
            assigned[best_name].append((pb[1], it["text"]))

    for f in fields:
        name = f["name"]
        if name not in field_labels:
            continue
        rows = sorted(assigned.get(name) or [], key=lambda x: x[0])
        values = [t.strip() for _, t in rows if str(t).strip()]
        if not values:
            out[name] = None
            notes.append(f"above_cells:empty_above:{name}")
        else:
            out[name] = values
            raw_parts.append(f"{name}_vals={values}")

    out["pairs"] = []
    out["read_mode"] = "above_cells"
    out["extract_backend"] = "ocr_above_cells"
    return out, " | ".join(raw_parts), notes


def apply_product_split_to_boxes(
    page_image: Image.Image,
    boxes: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """按 Product 行重写 material_table / main_table 的 bbox。"""
    notes: list[str] = []
    mat_i = [
        i
        for i, b in enumerate(boxes)
        if b.get("entity_id") == "material_table"
        or any(
            e.get("entity_id") == b.get("entity_id") and e.get("section") == "material"
            for e in plan
        )
    ]
    main_i = [
        i
        for i, b in enumerate(boxes)
        if b.get("entity_id") == "main_table"
        or any(
            e.get("entity_id") == b.get("entity_id") and e.get("section") == "main" for e in plan
        )
    ]
    if not mat_i and not main_i:
        return boxes, notes

    seeds = []
    for i in mat_i + main_i:
        bb = boxes[i].get("bbox")
        if bb:
            seeds.append(list(bb))
    mat_bb, main_bb, sn = split_title_block_by_product(page_image, seeds)
    notes.extend(sn)
    if mat_bb is None or main_bb is None:
        return boxes, notes

    out = [dict(b) for b in boxes]
    for i in mat_i:
        out[i]["bbox"] = mat_bb
        out[i]["_y1_floor"] = None
    for i in main_i:
        out[i]["bbox"] = main_bb
        out[i]["_y1_floor"] = main_bb[1]  # 裁剪时不要扩到 Product 之上
    return out, notes
