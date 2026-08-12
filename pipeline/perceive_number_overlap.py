"""数字重叠感知（墨迹为主）。

1) OCR 定位数字（原图优先 + 竖排 ±90；小图再补放大 OCR）
2) 两框墨迹像素 AND / 框内笔画组 AND
3) 粘连叠印：骨架交叉率 + 边缘密度 + 墨迹/骨架比（CAD 空心字叠印特征）
4) 仅确认重叠才输出 keep_pair（规则 only_keep_pair）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import load_config, resolve_path
from pipeline.dimension_parse import apply_dimension_parse_to_instances
from pipeline.drawing_parse_plan import dimension_marks_uses_ocr_parse, get_dimension_marks_config
from pipeline.perceive_common import (
    filter_instances_matching_table_values,
    filter_instances_outside_bboxes,
)
from pipeline.perceive_utils import iou_xyxy, reindex_instances
from pipeline.text_angle import (
    get_instance_angle,
    normalize_text_angle,
    suggest_extra_page_angles,
    text_angle_from_bbox,
    text_angle_from_quad,
    text_angle_in_original,
)
_MOSTLY_SYMBOL = re.compile(r"^[\s\-–—·•./\\|()\[\]{}<>~`'\",:;_+*=#@$%^&±]+$")


def _quad_to_xyxy(box) -> list[int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _box_side(b: list[int]) -> float:
    return float(min(max(1, b[2] - b[0]), max(1, b[3] - b[1])))


def _box_area(b: list[int]) -> float:
    return float(max(1, b[2] - b[0]) * max(1, b[3] - b[1]))


def _clip_bbox(bbox: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    return [x1, y1, x2, y2]


def _looks_like_number_mark(text: str) -> bool:
    """尺寸/数值候选：含数字或单独直径/角度符号（符号常与数字拆框）。"""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return False
    from pipeline.dimension_parse import has_non_dimension_word_prefix, is_dim_symbol_only

    # Max.3 / TYP 等：直接排除，不进尺寸候选
    if has_non_dimension_word_prefix(t):
        return False
    # 单独 Ø/° 等：保留，后续与邻近数字框合并
    if is_dim_symbol_only(t):
        return True
    if _MOSTLY_SYMBOL.match(t):
        return False
    if not re.search(r"\d", t):
        return False
    # 长图号 / 料号（常与标题栏 document_number 重复）
    if re.fullmatch(r"\d{8,}[A-Za-z]?", t):
        return False
    # 长数字开头的图号变体（如 25001743T.2），无公差/直径符号
    if re.match(r"\d{7,}", t) and not re.search(r"[±Ø⌀ФфΦφøRr°ºOoQqDd]", t):
        return False
    # 材料代号 / 标准号（常与物料表字段重复）
    if re.search(r"(?i)(?:din\s*en|sheet|cu[\s\-]?etp|material|siemens)", t):
        return False
    letters = len(re.findall(r"[A-Za-z]", t))
    digits = len(re.findall(r"\d", t))
    # Surface / Finish、Siemens 2025 等：字母偏多且不像 Ø/R/±/° 尺寸
    if letters >= 4 and letters >= digits and not re.search(r"[Ø⌀ФфΦφøRr±°º]", t):
        return False
    if letters >= 5 and letters >= digits * 2:
        return False
    return True


def _merge_symbol_number_boxes(
    instances: list[dict[str, Any]],
    *,
    gap_ratio: float = 2.8,
) -> list[dict[str, Any]]:
    """把单独符号框（Ø/°）与邻近数字框合并，避免直径/角度因符号漏检被拆丢。"""
    from pipeline.dimension_parse import is_dim_symbol_only, normalize_ocr_dimension_text

    if not instances:
        return instances

    symbols: list[tuple[int, dict[str, Any]]] = []
    others: list[tuple[int, dict[str, Any]]] = []
    for idx, inst in enumerate(instances):
        text = str(inst.get("raw_text") or (inst.get("fields") or {}).get("text") or "").strip()
        if is_dim_symbol_only(text):
            symbols.append((idx, inst))
        else:
            others.append((idx, inst))
    if not symbols:
        return instances

    used_other: set[int] = set()
    used_sym: set[int] = set()
    merged: list[dict[str, Any]] = []

    def _center(b: list[int]) -> tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _side(b: list[int]) -> float:
        return float(max(b[2] - b[0], b[3] - b[1], 1))

    for si, sym in symbols:
        sb = sym.get("bbox")
        if not sb or len(sb) != 4:
            continue
        sx, sy = _center(sb)
        st = str(sym.get("raw_text") or (sym.get("fields") or {}).get("text") or "").strip()
        best_j = None
        best_dist = 1e18
        for oj, other in others:
            if oj in used_other:
                continue
            if other.get("keep_pair"):
                continue
            ob = other.get("bbox")
            if not ob or len(ob) != 4:
                continue
            ot = str(other.get("raw_text") or (other.get("fields") or {}).get("text") or "")
            if not re.search(r"\d", ot):
                continue
            ox, oy = _center(ob)
            dist = ((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5
            gap = gap_ratio * max(_side(sb), _side(ob))
            # 也允许轴对齐邻近（符号贴在数字左侧/右侧/上下）
            axis_near = (
                abs(sx - ox) <= gap and abs(sy - oy) <= gap
            ) or (
                abs(sy - oy) <= 0.85 * max(_side(sb), _side(ob))
                and (
                    abs(sb[2] - ob[0]) <= gap
                    or abs(ob[2] - sb[0]) <= gap
                    or abs(sb[3] - ob[1]) <= gap
                    or abs(ob[3] - sb[1]) <= gap
                )
            )
            if dist > gap and not axis_near:
                continue
            if dist < best_dist:
                best_dist = dist
                best_j = oj
        if best_j is None:
            continue
        other = next(o for j, o in others if j == best_j)
        ob = list(other["bbox"])
        ot = str(other.get("raw_text") or (other.get("fields") or {}).get("text") or "").strip()
        # 符号在左/上 → 前缀；角度符优先作后缀
        sym_left = sx <= (ob[0] + ob[2]) / 2.0
        is_angle_sym = st in {"°", "º", "˚", "ₒ"}
        if is_angle_sym:
            combined = f"{ot}°"
        else:
            combined = f"{st}{ot}" if sym_left else f"{ot}{st}"
        combined = normalize_ocr_dimension_text(combined)
        nb = [
            min(sb[0], ob[0]),
            min(sb[1], ob[1]),
            max(sb[2], ob[2]),
            max(sb[3], ob[3]),
        ]
        conf = max(float(sym.get("confidence") or 0), float(other.get("confidence") or 0))
        out = dict(other)
        out["bbox"] = nb
        out["raw_text"] = combined
        fields = dict(out.get("fields") or {})
        fields["text"] = combined
        fields["symbol_merged"] = True
        out["fields"] = fields
        out["confidence"] = conf
        merged.append(out)
        used_other.add(best_j)
        used_sym.add(si)

    result: list[dict[str, Any]] = []
    for idx, inst in enumerate(instances):
        if idx in used_sym or idx in used_other:
            continue
        # 未合并的纯符号框不再单独保留（无尺寸值）
        text = str(inst.get("raw_text") or (inst.get("fields") or {}).get("text") or "").strip()
        if is_dim_symbol_only(text):
            continue
        result.append(inst)
    result.extend(merged)
    return result


def _normalize_ocr_instance_text(inst: dict[str, Any]) -> dict[str, Any]:
    from pipeline.dimension_parse import normalize_ocr_dimension_text

    out = dict(inst)
    fields = dict(out.get("fields") or {})
    raw = str(out.get("raw_text") or fields.get("text") or "")
    norm = normalize_ocr_dimension_text(raw)
    if norm and norm != raw:
        out["raw_text"] = norm
        fields["text"] = norm
        out["fields"] = fields
    elif fields.get("text"):
        fields["text"] = normalize_ocr_dimension_text(str(fields.get("text")))
        out["fields"] = fields
    return out


def _is_titleish_or_tiny_text(text: str, bbox: list[int], *, page_h: int) -> bool:
    """标题栏小字、图号、比例等：不做粘连叠印判定（仍可走两框墨迹 AND）。"""
    t = (text or "").strip()
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    short, long = min(bw, bh), max(bw, bh)
    # 过小框（图框栏常见）
    if short < 14 or long < 34:
        return True
    # 图幅下部标题栏区域的矮横条文字
    if page_h > 0 and y1 > 0.78 * page_h and bh <= 26 and bw >= 40:
        return True
    # 长图号 / 料号
    if re.fullmatch(r"\d{7,}[A-Za-z]?", t):
        return True
    # 比例、页码
    if re.fullmatch(r"\d+\s*[:/]\s*\d+", t):
        return True
    # 过短干净数字（空心字易被骨架特征误伤）
    if re.fullmatch(r"\d{1,4}([.]\d{1,3})?", t) and long < 45:
        return True
    return False


def _ocr_looks_garbled_overlap(text: str) -> bool:
    """叠印 OCR 常出现异常空格、多小数点、破碎符号、数字粘连。"""
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"\d\s+\.\d|\d\s+\d|\.\s+\d", t):
        return True
    if re.search(r"\d\s+[±]|[±]\s+\d", t):
        return True
    if t.count(".") >= 2 and "±" not in t and "+-" not in t and "℃" not in t:
        return True
    if "  " in t:
        return True
    # 竖排叠印常被读成 5140.1 / 3390.1 / 32:3±0.2 / 32.3+0.2
    if re.search(r"\d:\d", t):
        return True
    if re.search(r"\d\.\d\+\d|\d\+\d\.\d", t):
        return True
    if re.search(r"[±+\-]\d*\.$", t):
        return True
    if re.search(r"[±]\d$", t) or t.endswith("±0") or t.endswith("±."):
        return True
    if re.fullmatch(r"\d{3,4}\.\d", t) and "±" not in t:
        return True
    return False


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    import cv2

    if gray.size == 0:
        return gray
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)


def _full_ink_mask(image: Image.Image) -> np.ndarray:
    return _binarize_ink(np.array(image.convert("L")))


def _remove_long_lines(ink: np.ndarray) -> np.ndarray:
    """去掉尺寸线等长直线；核长度封顶，避免把竖排数字整列抹掉。"""
    import cv2

    h, w = ink.shape
    # 核边长封顶：过大时竖排文字会被当成「竖线」删掉
    hlen = int(min(72, max(18, w // 12)))
    vlen = int(min(72, max(18, h // 12)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    return cv2.subtract(ink, cv2.bitwise_or(horiz, vert))


def _morph_skeleton(bw: np.ndarray) -> np.ndarray:
    import cv2

    img = bw.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _ink_in_bbox(ink: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = ink.shape
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = ink[y1:y2, x1:x2] > 0
    return mask


def _ink_overlap_stats(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    area_a = float(mask_a.sum())
    area_b = float(mask_b.sum())
    min_ab = min(area_a, area_b)
    return {
        "ink_inter": inter,
        "ink_a": area_a,
        "ink_b": area_b,
        "ink_ratio": (inter / min_ab) if min_ab > 0 else 0.0,
    }


def _is_ink_overlap(
    stats: dict[str, float],
    *,
    min_ink_pixels: float,
    min_ink_ratio: float,
) -> bool:
    return stats["ink_inter"] >= min_ink_pixels and stats["ink_ratio"] >= min_ink_ratio


def _fused_ink_metrics(image: Image.Image, bbox: list[int] | None = None) -> dict[str, float]:
    """粘连叠印墨迹特征：骨架交叉率、边缘密度、墨迹/骨架比。"""
    import cv2

    gray = np.array(image.convert("L"))
    ink = _binarize_ink(gray)
    ink = _remove_long_lines(ink)
    h, w = ink.shape
    if bbox is not None:
        x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
        crop = ink[y1:y2, x1:x2]
        gc = gray[y1:y2, x1:x2]
    else:
        crop, gc = ink, gray

    ys, xs = np.where(crop > 0)
    if len(xs) < 12:
        return {"junc_ps": 0.0, "edge": 0.0, "ips": 99.0, "sk": 0.0, "ink": 0.0}
    # 再收紧到墨迹外接
    yy1, yy2 = max(0, int(ys.min()) - 1), min(crop.shape[0], int(ys.max()) + 2)
    xx1, xx2 = max(0, int(xs.min()) - 1), min(crop.shape[1], int(xs.max()) + 2)
    crop = crop[yy1:yy2, xx1:xx2]
    gc = gc[yy1:yy2, xx1:xx2]

    try:
        sk = cv2.ximgproc.thinning(crop) if hasattr(cv2, "ximgproc") else _morph_skeleton(crop)
    except Exception:
        sk = _morph_skeleton(crop)

    ink_n = float((crop > 0).sum())
    sk_n = float((sk > 0).sum())
    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    nbr = cv2.filter2D((sk > 0).astype(np.uint8), -1, k)
    junc = float(((sk > 0) & (nbr >= 13)).sum())
    junc_ps = junc / max(sk_n, 1.0)
    edges = cv2.Canny(gc, 40, 120)
    edge = float((edges > 0).mean())
    ips = ink_n / max(sk_n, 1.0)
    return {
        "junc_ps": float(junc_ps),
        "edge": float(edge),
        "ips": float(ips),
        "sk": sk_n,
        "ink": ink_n,
    }


def _is_fused_overlap(
    metrics: dict[str, float],
    *,
    junc_thr: float,
    edge_thr: float,
    ips_max: float,
) -> bool:
    if metrics.get("sk", 0) < 55:
        return False
    junc = metrics.get("junc_ps", 0.0)
    edge = metrics.get("edge", 0.0)
    ips = metrics.get("ips", 99.0)
    # 三者同时满足，降低空心正常字误报
    return junc >= junc_thr and edge >= edge_thr and ips <= ips_max


def _within_box_ink_group_overlap(
    ink: np.ndarray,
    bbox: list[int],
    *,
    min_ink_pixels: float,
    min_ink_ratio: float,
) -> dict[str, float] | None:
    import cv2

    h, w = ink.shape
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    crop = (ink[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    if crop.size < 30 or int((crop > 0).sum()) < 40:
        return None

    best: dict[str, float] | None = None
    for erode_it in (1, 2):
        eroded = cv2.erode(crop, np.ones((2, 2), np.uint8), iterations=erode_it)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        comps: list[int] = []
        min_area = max(20, int(0.03 * (crop > 0).sum()))
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                comps.append(i)
        if len(comps) < 2:
            continue
        comps = sorted(comps, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:4]
        dilate_it = erode_it + 1
        kernel = np.ones((3, 3), np.uint8)
        for a_i in range(len(comps)):
            for b_i in range(a_i + 1, len(comps)):
                ma = (labels == comps[a_i]).astype(np.uint8) * 255
                mb = (labels == comps[b_i]).astype(np.uint8) * 255
                da = cv2.dilate(ma, kernel, iterations=dilate_it)
                db = cv2.dilate(mb, kernel, iterations=dilate_it)
                inter_mask = (da > 0) & (db > 0) & (crop > 0)
                inter = float(inter_mask.sum())
                area_a = float(((da > 0) & (crop > 0)).sum())
                area_b = float(((db > 0) & (crop > 0)).sum())
                min_ab = min(area_a, area_b)
                ratio = (inter / min_ab) if min_ab > 0 else 0.0
                stats_d = {"ink_inter": inter, "ink_a": area_a, "ink_b": area_b, "ink_ratio": ratio}
                if not _is_ink_overlap(stats_d, min_ink_pixels=min_ink_pixels, min_ink_ratio=min_ink_ratio):
                    continue
                if best is None or stats_d["ink_inter"] > best["ink_inter"]:
                    best = stats_d
        if best is not None:
            return best
    return None


def _bboxes_may_touch(a: list[int], b: list[int], *, pad: int = 2) -> bool:
    return not (
        a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]
    )


def _dedupe_boxes(instances: list[dict[str, Any]], *, iou_thr: float = 0.85) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for cand in sorted(instances, key=lambda x: float(x.get("confidence") or 0), reverse=True):
        if any(iou_xyxy(cand["bbox"], s["bbox"]) >= iou_thr for s in kept):
            continue
        kept.append(cand)
    return kept


def _cluster_near_duplicates(instances: list[dict[str, Any]], *, iou_thr: float = 0.35) -> list[dict[str, Any]]:
    """合并近重复框；避免宽横条（材料牌号）吞掉竖排尺寸。"""
    kept: list[dict[str, Any]] = []
    for cand in sorted(instances, key=lambda x: float(x.get("confidence") or 0), reverse=True):
        ca = cand["bbox"]
        cx = (ca[0] + ca[2]) / 2.0
        cy = (ca[1] + ca[3]) / 2.0
        aw, ah = max(1.0, ca[2] - ca[0]), max(1.0, ca[3] - ca[1])
        cand_vert = ah >= 1.35 * aw
        merged = False
        for s in kept:
            sb = s["bbox"]
            sw, sh = max(1.0, sb[2] - sb[0]), max(1.0, sb[3] - sb[1])
            kept_horiz = sw >= 1.8 * sh
            # 竖排数字不要被横向长牌号框吞掉
            if cand_vert and kept_horiz and ah > 1.2 * sh:
                continue
            if iou_xyxy(ca, sb) >= iou_thr:
                merged = True
                break
            # 中心落入对方：需有一定 IoU，避免大框误吞
            if sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]:
                if iou_xyxy(ca, sb) >= 0.12 or (aw * ah) > 0.35 * (sw * sh):
                    merged = True
                    break
                continue
            sx = (sb[0] + sb[2]) / 2.0
            sy = (sb[1] + sb[3]) / 2.0
            if ca[0] <= sx <= ca[2] and ca[1] <= sy <= ca[3]:
                if iou_xyxy(ca, sb) >= 0.12:
                    merged = True
                    break
                continue
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            ref = max(aw, sw, ah, sh, 8)
            if dist < 0.45 * ref:
                merged = True
                break
            ang_a = abs(
                float(
                    cand.get("page_rotate")
                    if cand.get("page_rotate") is not None
                    else cand.get("angle")
                    or 0
                )
            )
            ang_b = abs(
                float(
                    s.get("page_rotate")
                    if s.get("page_rotate") is not None
                    else s.get("angle")
                    or 0
                )
            )
            if ang_a >= 80 and ang_b >= 80 and dist < 1.1 * ref:
                merged = True
                break
        if not merged:
            kept.append(cand)
    return kept


def _rotate_image(image: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 1e-6:
        return image
    return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def _rotate_box_back(
    bbox: list[int],
    angle: float,
    rot_w: int,
    rot_h: int,
    orig_w: int,
    orig_h: int,
) -> list[int]:
    if abs(angle) < 1e-6:
        return bbox
    import math

    rad = math.radians(angle)
    cx_r, cy_r = rot_w / 2.0, rot_h / 2.0
    cx_o, cy_o = orig_w / 2.0, orig_h / 2.0
    cos_a, sin_a = math.cos(-rad), math.sin(-rad)
    x1, y1, x2, y2 = bbox
    pts = []
    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        dx, dy = x - cx_r, y - cy_r
        pts.append((cos_a * dx - sin_a * dy + cx_o, sin_a * dx + cos_a * dy + cy_o))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [
        int(max(0, min(orig_w - 1, min(xs)))),
        int(max(0, min(orig_h - 1, min(ys)))),
        int(max(1, min(orig_w, max(xs)))),
        int(max(1, min(orig_h, max(ys)))),
    ]


def _bbox_has_ink(ink: np.ndarray, bbox: list[int], *, min_pixels: int = 12) -> bool:
    h, w = ink.shape
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    return int((ink[y1:y2, x1:x2] > 0).sum()) >= min_pixels


def _ocr_once(
    engine,
    image: Image.Image,
    ink: np.ndarray,
    *,
    conf_th: float,
    min_side: float,
    img_min_side: float,
    entity_id: str,
    angle: float = 0.0,
    orig_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """整图 OCR；angle 为页图旋转角，fields.angle 为原图文本相对水平线的朝向。"""
    page_rot = float(angle)
    work = _rotate_image(image, page_rot) if abs(page_rot) > 1e-6 else image
    result, _ = engine(np.array(work.convert("RGB")))
    ow, oh = orig_size or image.size
    rw, rh = work.size
    out: list[dict[str, Any]] = []
    for row in result or []:
        if not row or len(row) < 3:
            continue
        box, text, score = row[0], str(row[1]).strip(), float(row[2])
        if score < conf_th:
            continue
        if not _looks_like_number_mark(text):
            continue
        bbox = _quad_to_xyxy(box)
        quad_ang = text_angle_from_quad(box)
        text_ang = text_angle_in_original(quad_ang, page_rot)
        if abs(page_rot) > 1e-6:
            bbox = _rotate_box_back(bbox, page_rot, rw, rh, ow, oh)
            if not _bbox_has_ink(ink, bbox):
                continue
        side = _box_side(bbox)
        # 小裁剪图放宽最小边
        if side < min_side and side < 0.008 * img_min_side and min(ow, oh) > 200:
            continue
        out.append(
            {
                "entity_id": entity_id,
                "label": "数字标注",
                "bbox": bbox,
                "fields": {"text": text, "angle": text_ang},
                "raw_text": text,
                "confidence": score,
                "needs_review": score < 0.6,
                "angle": text_ang,
                "page_rotate": page_rot,
            }
        )
    return out


def _deskew_reread_instance(
    engine,
    image: Image.Image,
    inst: dict[str, Any],
    *,
    conf_th: float,
    min_abs_angle: float = 8.0,
    pad_ratio: float = 0.12,
) -> dict[str, Any]:
    """按文本角 deskew 裁剪后重读 OCR，置信度更高则替换文本。"""
    ang = get_instance_angle(inst)
    if abs(ang) < min_abs_angle:
        return inst
    bbox = inst.get("bbox")
    if not bbox or len(bbox) != 4:
        return inst
    w, h = image.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = max(2, int(max(bw, bh) * pad_ratio))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return inst
    crop = image.crop((cx1, cy1, cx2, cy2))
    # 转到近水平；过小则放大
    deskew = crop.rotate(-ang, expand=True, fillcolor=(255, 255, 255))
    dw, dh = deskew.size
    if max(dw, dh) < 48:
        scale = 48.0 / max(dw, dh)
        deskew = deskew.resize(
            (max(1, int(dw * scale)), max(1, int(dh * scale))),
            Image.Resampling.LANCZOS,
        )
    try:
        result, _ = engine(np.array(deskew.convert("RGB")))
    except Exception:
        return inst
    best_text = None
    best_score = -1.0
    for row in result or []:
        if not row or len(row) < 3:
            continue
        text, score = str(row[1]).strip(), float(row[2])
        if score < conf_th or not _looks_like_number_mark(text):
            continue
        if score > best_score:
            best_score = score
            best_text = text
    if best_text is None:
        return inst
    old_score = float(inst.get("confidence") or 0.0)
    # 明显更好，或旧文本乱而新文本更干净
    old_text = str(inst.get("raw_text") or "")
    better = best_score >= old_score + 0.03
    rescue = best_score >= conf_th and _ocr_looks_garbled_overlap(old_text) and not _ocr_looks_garbled_overlap(
        best_text
    )
    if not (better or rescue):
        return inst
    out = dict(inst)
    fields = dict(out.get("fields") or {})
    fields["text"] = best_text
    fields["angle"] = normalize_text_angle(ang)
    fields["ocr_deskew"] = True
    out["fields"] = fields
    out["raw_text"] = best_text
    out["confidence"] = best_score
    out["needs_review"] = best_score < 0.6
    out["angle"] = fields["angle"]
    return out


def _ensure_angle_fields(inst: dict[str, Any]) -> dict[str, Any]:
    """保证 fields.angle / top-level angle 一致。"""
    ang = get_instance_angle(inst)
    fields = dict(inst.get("fields") or {})
    fields["angle"] = ang
    inst = dict(inst)
    inst["fields"] = fields
    inst["angle"] = ang
    return inst


def _emit_overlap_pair_from_bbox(
    entity_id: str,
    bbox: list[int],
    *,
    text: str,
    score: float,
    parent_id: str,
    stats: dict[str, float],
    reason: str,
    angle: float | None = None,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = bbox
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    # 竖条文本左右切效果差，改按长边切开
    if bh >= bw:
        a = [x1, y1, x2, y1 + max(2, int(bh * 0.72))]
        b = [x1, y1 + int(bh * 0.28), x2, y2]
    else:
        a = [x1, y1, x1 + max(2, int(bw * 0.72)), y2]
        b = [x1 + int(bw * 0.28), y1, x2, y2]
    payload = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in stats.items()}
    ang = normalize_text_angle(angle) if angle is not None else text_angle_from_bbox(bbox)
    return [
        {
            "entity_id": entity_id,
            "label": "墨迹重叠A",
            "bbox": a,
            "fields": {"text": text, "angle": ang, "overlap_reason": reason, **payload},
            "raw_text": text,
            "confidence": score,
            "needs_review": True,
            "parent_id": parent_id,
            "keep_pair": True,
            "angle": ang,
        },
        {
            "entity_id": entity_id,
            "label": "墨迹重叠B",
            "bbox": b,
            "fields": {"text": text, "angle": ang, "overlap_reason": reason, **payload},
            "raw_text": text,
            "confidence": score,
            "needs_review": True,
            "parent_id": parent_id + "_twin",
            "keep_pair": True,
            "angle": ang,
        },
    ]


def _local_ocr_ink_windows(
    engine,
    image: Image.Image,
    ink: np.ndarray,
    *,
    conf_th: float,
    min_side: float,
    entity_id: str,
    angles: list[float],
    page_h: int,
    existing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """在紧凑墨迹窗上补 OCR，专打全图/分块仍漏的竖排叠印。"""
    import cv2

    h, w = ink.shape
    existing = existing or []
    # 先去长直线再找紧凑文字团，否则尺寸线会把整图粘成一块
    ink_txt = _remove_long_lines(ink)
    band = cv2.morphologyEx(ink_txt, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    # 竖排/横排分别加强连通
    band = cv2.bitwise_or(
        band,
        cv2.morphologyEx(ink_txt, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8), iterations=1),
    )
    band = cv2.bitwise_or(
        band,
        cv2.morphologyEx(ink_txt, cv2.MORPH_CLOSE, np.ones((2, 5), np.uint8), iterations=1),
    )
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    out: list[dict[str, Any]] = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 90 or area > 3500:
            continue
        if min(bw, bh) < 12 or max(bw, bh) < 36 or max(bw, bh) > 200:
            continue
        if y > 0.78 * page_h and bh <= 28:
            continue
        ar = bw / max(bh, 1)
        # 只要偏横或偏竖的文本条
        if 0.75 <= ar <= 1.35 and area < 400:
            continue
        bbox = [x, y, x + bw, y + bh]
        # 已被已有框较好覆盖则跳过
        if any(iou_xyxy(bbox, e["bbox"]) > 0.35 for e in existing if e.get("bbox")):
            continue
        # 先看叠印特征，不够则跳过，少做无效 OCR
        metrics = _fused_ink_metrics(image, bbox)
        vertical = bh >= 1.35 * bw
        if not _is_fused_overlap(
            metrics,
            junc_thr=0.68 if vertical else 0.72,
            edge_thr=0.23 if vertical else 0.28,
            ips_max=1.58,
        ):
            continue
        pad = 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = image.crop((x0, y0, x1, y1))
        crop_ink = ink[y0:y1, x0:x1]
        angs = [0.0, 90.0, -90.0] if vertical else [0.0, 90.0, -90.0]
        for ang in angs:
            part = _ocr_once(
                engine,
                crop,
                crop_ink,
                conf_th=max(0.28, conf_th - 0.07),
                min_side=min_side,
                img_min_side=float(min(x1 - x0, y1 - y0)),
                entity_id=entity_id,
                angle=float(ang),
                orig_size=(x1 - x0, y1 - y0),
            )
            for inst in part:
                bx = inst["bbox"]
                inst["bbox"] = [bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0]
                inst["fields"] = {**(inst.get("fields") or {}), "local_window": True}
                out.append(inst)
    return out


def _ocr_tiled(
    engine,
    image: Image.Image,
    ink: np.ndarray,
    *,
    conf_th: float,
    min_side: float,
    entity_id: str,
    angles: list[float],
    tile_size: int = 768,
    overlap: float = 0.25,
    min_ink_frac: float = 0.008,
) -> list[dict[str, Any]]:
    """大图分块 OCR：补全图漏检的局部叠印/竖排数字。"""
    w, h = image.size
    if max(w, h) < tile_size + 80:
        return []
    step = max(64, int(tile_size * (1.0 - overlap)))
    out: list[dict[str, Any]] = []
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            if x1 - x0 < 96 or y1 - y0 < 96:
                continue
            patch_ink = ink[y0:y1, x0:x1]
            if float((patch_ink > 0).mean()) < min_ink_frac:
                continue
            crop = image.crop((x0, y0, x1, y1))
            crop_ink = patch_ink
            for ang in angles:
                part = _ocr_once(
                    engine,
                    crop,
                    crop_ink,
                    conf_th=conf_th,
                    min_side=min_side,
                    img_min_side=float(min(x1 - x0, y1 - y0)),
                    entity_id=entity_id,
                    angle=ang,
                    orig_size=(x1 - x0, y1 - y0),
                )
                for inst in part:
                    bx = inst["bbox"]
                    inst["bbox"] = [bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0]
                    inst["fields"] = {**(inst.get("fields") or {}), "tile_origin": [x0, y0]}
                    out.append(inst)
    return out


def _scan_fused_blobs(
    image: Image.Image,
    entity_id: str,
    *,
    junc_thr: float,
    edge_thr: float,
    ips_max: float,
    min_side: int,
    page_h: int | None = None,
) -> list[dict[str, Any]]:
    """无 OCR 覆盖时：墨迹团 + 叠印特征兜底。"""
    import cv2

    ink = _remove_long_lines(_full_ink_mask(image))
    h, w = ink.shape
    ph = page_h or h
    band_h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8), iterations=2)
    band_v = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=2)
    band = cv2.bitwise_or(band_h, band_v)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    out: list[dict[str, Any]] = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 80 or min(bw, bh) < max(10, min_side // 2):
            continue
        if max(bw, bh) < 34 or max(bw, bh) > 240:
            continue
        if bw > w * 0.3 or bh > h * 0.3:
            continue
        if y > 0.78 * ph and bh <= 28:
            continue
        bbox = [x, y, x + bw, y + bh]
        metrics = _fused_ink_metrics(image, bbox)
        vertical = bh >= 1.35 * bw
        ok = _is_fused_overlap(
            metrics,
            junc_thr=junc_thr - (0.04 if vertical else 0.0),
            edge_thr=edge_thr - (0.05 if vertical else 0.0),
            ips_max=ips_max + (0.05 if vertical else 0.0),
        )
        if not ok:
            continue
        reason = f"fused_blob_j={metrics['junc_ps']:.2f},e={metrics['edge']:.2f},ips={metrics['ips']:.2f}"
        out.extend(
            _emit_overlap_pair_from_bbox(
                entity_id,
                bbox,
                text="",
                score=min(1.0, 0.5 + metrics["junc_ps"] * 0.4),
                parent_id=f"fused_blob{i}",
                stats=metrics,
                reason=reason,
            )
        )
    return out


def perceive_number_overlap(
    image_path: str | Path,
    plan: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    allow_mock_fallback: bool = True,
    rules: list[dict[str, Any]] | None = None,
    exclude_bboxes: list[list[int]] | None = None,
    exclude_pad: float = 0.0,
    exclude_table_texts: set[str] | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    overlap_cfg = (cfg.get("perception") or {}).get("number_overlap") or {}
    ocr_cfg = cfg.get("models", {}).get("ocr", {})
    dim_cfg = get_dimension_marks_config(cfg)
    # ocr / ocr_locate_vlm_filter：本路径做尺寸解析与规则初筛；纯 vlm 属性不在此解析
    parse_dims = dimension_marks_uses_ocr_parse(cfg)
    # drawing_parse.dimension_marks.detect_overlap；若启用重叠规则则强制开
    detect_overlap = bool(dim_cfg.get("detect_overlap", False))
    if rules and any(str(r.get("rule_id") or "") == "NUM_TEXT_NO_OVERLAP" for r in rules):
        detect_overlap = True
    # plan 中带 parse_kind=dimension_marks 且未开重叠规则时，尊重 yaml
    if any(e.get("parse_kind") == "dimension_marks" for e in (plan or [])):
        if not (rules and any(str(r.get("rule_id") or "") == "NUM_TEXT_NO_OVERLAP" for r in rules)):
            detect_overlap = bool(dim_cfg.get("detect_overlap", False))
    # 规则重叠实体在 ocr plan 中、属性走 VLM 时：本路只做重叠检测
    if not any(e.get("parse_kind") == "dimension_marks" for e in (plan or [])):
        if rules and any(str(r.get("rule_id") or "") == "NUM_TEXT_NO_OVERLAP" for r in rules):
            detect_overlap = True
            parse_dims = False

    conf_th = float(overlap_cfg.get("confidence_threshold", ocr_cfg.get("confidence_threshold", 0.35)))
    min_side = int(overlap_cfg.get("min_box_side", 8))
    min_ink_pixels = float(overlap_cfg.get("min_ink_pixels", 20))
    min_ink_ratio = float(overlap_cfg.get("min_ink_ratio", 0.06))
    # 重叠检测 OCR：仅用 perception.number_overlap（ocr_angles 等），不改其行为
    angles = [float(a) for a in overlap_cfg.get("ocr_angles", [0, 90, -90])]
    # 尺寸属性专用：自适应角 / deskew；一旦走重叠检测则强制关闭，避免干扰
    if detect_overlap:
        angle_adapt = False
        deskew_reread = False
        deskew_min_angle = 8.0
    else:
        angle_adapt = bool(dim_cfg.get("ocr_angle_adapt", False))
        deskew_reread = bool(dim_cfg.get("ocr_deskew_reread", False))
        deskew_min_angle = float(dim_cfg.get("ocr_deskew_min_angle", 8.0))
    fused_enabled = bool(overlap_cfg.get("fused_enabled", True)) and detect_overlap
    junc_thr = float(overlap_cfg.get("fused_junc_per_skel", 0.68))
    edge_thr = float(overlap_cfg.get("fused_edge_density", 0.26))
    ips_max = float(overlap_cfg.get("fused_ink_per_skel_max", 1.55))

    meta = meta or {}
    image_orig = Image.open(resolve_path(image_path)).convert("RGB")
    ow0, oh0 = image_orig.size

    try:
        from pipeline.ocr_engine import get_rapid_ocr

        engine = get_rapid_ocr()
    except ImportError:
        if allow_mock_fallback:
            from pipeline.perceive_common import mock_perceive

            plan = plan or [
                {
                    "entity_id": "number_mark",
                    "locate_query": "数字标注",
                    "fields": [{"name": "text", "parse_hint": "数字"}],
                }
            ]
            payload = mock_perceive(plan, meta)
            payload["note"] = "OCR 未安装，数字重叠检测回退 mock"
            return payload
        raise
    target_eids = {
        e.get("entity_id")
        for e in (plan or [])
        if e.get("entity_id") in {"number_mark", "annotation", "annotations"}
    }
    entity_id = "number_mark" if not target_eids or "number_mark" in target_eids else next(iter(target_eids))

    # 1) 先在原图 OCR（小裁剪放大后 RapidOCR 常失效）
    ink_orig = _full_ink_mask(image_orig)
    ocr_insts: list[dict[str, Any]] = []
    for ang in angles:
        ocr_insts.extend(
            _ocr_once(
                engine,
                image_orig,
                ink_orig,
                conf_th=conf_th,
                min_side=min_side,
                img_min_side=float(min(ow0, oh0)),
                entity_id=entity_id,
                angle=ang,
                orig_size=(ow0, oh0),
            )
        )

    # 1b) 大图分块 OCR：补全图漏掉的局部叠印/竖排
    tile_enabled = bool(overlap_cfg.get("tile_ocr_enabled", True))
    if tile_enabled and max(ow0, oh0) >= int(overlap_cfg.get("tile_ocr_min_side", 900)):
        tile_size = int(overlap_cfg.get("tile_size", 768))
        tile_overlap = float(overlap_cfg.get("tile_overlap", 0.25))
        tiled = _ocr_tiled(
            engine,
            image_orig,
            ink_orig,
            conf_th=max(0.30, conf_th - 0.05),
            min_side=min_side,
            entity_id=entity_id,
            angles=angles,
            tile_size=tile_size,
            overlap=tile_overlap,
        )
        ocr_insts.extend(tiled)
        notes_tile = len(tiled)
    else:
        notes_tile = 0

    # 1c) 紧凑墨迹窗补 OCR（竖排叠印常被大分块漏掉）
    local_extra = _local_ocr_ink_windows(
        engine,
        image_orig,
        ink_orig,
        conf_th=conf_th,
        min_side=min_side,
        entity_id=entity_id,
        angles=angles,
        page_h=oh0,
        existing=ocr_insts,
    )
    ocr_insts.extend(local_extra)
    notes_local = len(local_extra)

    # 2) 小图再补放大 OCR，坐标映射回原图
    scale = 1.0
    image = image_orig
    if max(ow0, oh0) < 280:
        scale = 320.0 / max(ow0, oh0)
        image = image_orig.resize(
            (max(1, int(ow0 * scale)), max(1, int(oh0 * scale))),
            Image.Resampling.LANCZOS,
        )
        ink_up = _full_ink_mask(image)
        uw, uh = image.size
        extra: list[dict[str, Any]] = []
        for ang in angles:
            extra.extend(
                _ocr_once(
                    engine,
                    image,
                    ink_up,
                    conf_th=conf_th,
                    min_side=max(6, min_side),
                    img_min_side=float(min(uw, uh)),
                    entity_id=entity_id,
                    angle=ang,
                    orig_size=(uw, uh),
                )
            )
        inv = 1.0 / scale
        for e in extra:
            x1, y1, x2, y2 = e["bbox"]
            e["bbox"] = [int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)]
        ocr_insts.extend(extra)

    # 2b) 按已检出文本角自适应补旋转 OCR（倾斜尺寸）
    adapt_extra = 0
    if angle_adapt:
        extra_angs = suggest_extra_page_angles(
            (get_instance_angle(i) for i in ocr_insts),
            existing=angles,
            step=float(dim_cfg.get("ocr_angle_adapt_step", 15)),
            min_count=int(dim_cfg.get("ocr_angle_adapt_min_count", 2)),
            max_extra=int(dim_cfg.get("ocr_angle_adapt_max_extra", 4)),
        )
        for ang in extra_angs:
            more = _ocr_once(
                engine,
                image_orig,
                ink_orig,
                conf_th=conf_th,
                min_side=min_side,
                img_min_side=float(min(ow0, oh0)),
                entity_id=entity_id,
                angle=ang,
                orig_size=(ow0, oh0),
            )
            ocr_insts.extend(more)
            adapt_extra += len(more)
        if extra_angs:
            angles = list(angles) + list(extra_angs)

    ocr_insts = _dedupe_boxes(ocr_insts, iou_thr=0.75)
    ocr_insts = _cluster_near_duplicates(ocr_insts, iou_thr=0.28)
    # 符号与数字拆框合并 + OCR 符号变体归一（Ø/°）
    ocr_insts = _merge_symbol_number_boxes(ocr_insts)
    ocr_insts = [_normalize_ocr_instance_text(i) for i in ocr_insts]
    sym_merged = sum(
        1 for i in ocr_insts if (i.get("fields") or {}).get("symbol_merged")
    )

    # 2c) 按文本角 deskew 裁剪重读，提高倾斜/竖排解析准确率
    deskew_hits = 0
    if deskew_reread and ocr_insts:
        refined: list[dict[str, Any]] = []
        for inst in ocr_insts:
            new_inst = _deskew_reread_instance(
                engine,
                image_orig,
                inst,
                conf_th=max(0.28, conf_th - 0.05),
                min_abs_angle=deskew_min_angle,
            )
            if new_inst.get("fields", {}).get("ocr_deskew"):
                deskew_hits += 1
            refined.append(_ensure_angle_fields(new_inst))
        ocr_insts = refined
    else:
        ocr_insts = [_ensure_angle_fields(i) for i in ocr_insts]

    # 非表格尺寸：剔除落在表格定位框内的 OCR 数字
    excluded_table = 0
    if exclude_bboxes:
        ocr_insts, excluded_table = filter_instances_outside_bboxes(
            ocr_insts,
            exclude_bboxes,
            pad=float(exclude_pad),
            entity_ids=None,  # 本路全是数字标注
        )
    excluded_table_text = 0
    if exclude_table_texts:
        ocr_insts, excluded_table_text = filter_instances_matching_table_values(
            ocr_insts,
            exclude_table_texts,
            entity_ids=None,
        )
    notes = [
        f"ocr={len(ocr_insts)}",
        f"tile_ocr={notes_tile}",
        f"local_ocr={notes_local}",
        f"angle_adapt_extra={adapt_extra}",
        f"deskew_reread={deskew_hits}",
        f"symbol_merged={sym_merged}",
        f"detect_overlap={detect_overlap}",
        f"dimension_marks={parse_dims}",
        f"excluded_table_region={excluded_table}",
        f"excluded_table_value={excluded_table_text}",
    ]

    pair_hits = 0
    within_hits = 0
    fused_hits = 0
    used: set[int] = set()
    collected: list[dict[str, Any]] = []

    if not detect_overlap:
        notes.append("overlap_skipped=dimension_marks_parse_only")
        for idx, inst in enumerate(ocr_insts):
            item = dict(inst)
            item["parent_id"] = f"ocr{idx}"
            ink_px = float(_ink_in_bbox(ink_orig, inst["bbox"]).sum())
            item["fields"] = {**(item.get("fields") or {}), "ink_pixels": ink_px}
            collected.append(item)
    else:
        # overlap analysis at original resolution
        analysis = image_orig
        ink = ink_orig
        masks = [_ink_in_bbox(ink_orig, inst["bbox"]) for inst in ocr_insts]

        for i in range(len(ocr_insts)):
            for j in range(i + 1, len(ocr_insts)):
                ba, bb = ocr_insts[i]["bbox"], ocr_insts[j]["bbox"]
                if not _bboxes_may_touch(ba, bb, pad=2):
                    continue
                # 跳过标题栏矮框、牌号类
                ta = str(ocr_insts[i].get("raw_text") or "")
                tb = str(ocr_insts[j].get("raw_text") or "")
                if _is_titleish_or_tiny_text(ta, ba, page_h=oh0) or _is_titleish_or_tiny_text(tb, bb, page_h=oh0):
                    continue
                if re.search(r"[A-Za-z]{3,}", ta) or re.search(r"[A-Za-z]{3,}", tb):
                    continue
                stats = _ink_overlap_stats(masks[i], masks[j])
                # 成对墨迹相交要足够「真压盖」
                if stats["ink_inter"] < max(min_ink_pixels, 64) or stats["ink_ratio"] < max(min_ink_ratio, 0.40):
                    continue
                # 墨迹对：两侧都像叠印乱码，或相交比例极高
                ga, gb = _ocr_looks_garbled_overlap(ta), _ocr_looks_garbled_overlap(tb)
                if not (ga and gb) and stats["ink_ratio"] < 0.62:
                    continue
                # 半径/普通尺寸相邻蹭墨
                if (re.match(r"^[Rr]", ta) or re.match(r"^[Rr]", tb)) and stats["ink_ratio"] < 0.58:
                    continue
                cxa, cya = (ba[0] + ba[2]) / 2.0, (ba[1] + ba[3]) / 2.0
                cxb, cyb = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
                dist = ((cxa - cxb) ** 2 + (cya - cyb) ** 2) ** 0.5
                ref = 0.5 * (max(ba[2] - ba[0], ba[3] - ba[1]) + max(bb[2] - bb[0], bb[3] - bb[1]))
                if dist > 0.85 * ref and stats["ink_ratio"] < 0.55:
                    continue
                pair_hits += 1
                used.add(i)
                used.add(j)
                score = max(float(ocr_insts[i].get("confidence") or 0), float(ocr_insts[j].get("confidence") or 0))
                reason = f"ink_inter={stats['ink_inter']:.0f},ratio={stats['ink_ratio']:.3f}"
                collected.extend(
                    [
                        {
                            "entity_id": entity_id,
                            "label": "墨迹重叠A",
                            "bbox": list(ocr_insts[i]["bbox"]),
                            "fields": {
                                "text": str(ocr_insts[i].get("raw_text") or ""),
                                "angle": get_instance_angle(ocr_insts[i]),
                                "overlap_reason": reason,
                                **stats,
                            },
                            "raw_text": str(ocr_insts[i].get("raw_text") or ""),
                            "confidence": score,
                            "needs_review": True,
                            "parent_id": f"ink{i}_{j}",
                            "keep_pair": True,
                            "angle": get_instance_angle(ocr_insts[i]),
                        },
                        {
                            "entity_id": entity_id,
                            "label": "墨迹重叠B",
                            "bbox": list(ocr_insts[j]["bbox"]),
                            "fields": {
                                "text": str(ocr_insts[j].get("raw_text") or ""),
                                "angle": get_instance_angle(ocr_insts[j]),
                                "overlap_reason": reason,
                                **stats,
                            },
                            "raw_text": str(ocr_insts[j].get("raw_text") or ""),
                            "confidence": score,
                            "needs_review": True,
                            "parent_id": f"ink{i}_{j}_twin",
                            "keep_pair": True,
                            "angle": get_instance_angle(ocr_insts[j]),
                        },
                    ]
                )

        # B) 框内笔画组 AND
        for idx, inst in enumerate(ocr_insts):
            if idx in used:
                continue
            stats = _within_box_ink_group_overlap(
                ink_orig,
                inst["bbox"],
                min_ink_pixels=min_ink_pixels,
                min_ink_ratio=min_ink_ratio,
            )
            if stats is None:
                continue
            within_hits += 1
            used.add(idx)
            collected.extend(
                _emit_overlap_pair_from_bbox(
                    entity_id,
                    inst["bbox"],
                    text=str(inst.get("raw_text") or ""),
                    score=float(inst.get("confidence") or 0),
                    parent_id=f"within{idx}",
                    stats=stats,
                    reason=f"ink_inter={stats['ink_inter']:.0f},ratio={stats['ink_ratio']:.3f}",
                    angle=get_instance_angle(inst),
                )
            )

        # C) 粘连叠印特征（原分辨率）
        if fused_enabled:
            for idx, inst in enumerate(ocr_insts):
                if idx in used:
                    continue
                text = str(inst.get("raw_text") or "")
                if _is_titleish_or_tiny_text(text, inst["bbox"], page_h=oh0):
                    continue
                # 图框坐标/噪声 OCR
                if re.search(r"[!|]", text) or inst["bbox"][2] < 0.06 * ow0:
                    continue
                metrics = _fused_ink_metrics(analysis, inst["bbox"])
                bw = max(1, inst["bbox"][2] - inst["bbox"][0])
                bh = max(1, inst["bbox"][3] - inst["bbox"][1])
                vertical = bh >= 1.35 * bw
                jt = junc_thr - (0.04 if vertical else 0.0)
                et = edge_thr - (0.05 if vertical else 0.0)
                imax = ips_max + (0.08 if vertical else 0.0)
                hit = _is_fused_overlap(metrics, junc_thr=jt, edge_thr=et, ips_max=imax)
                # 干净短数字即使骨架特征偏高也不报；乱码 OCR 可略放宽
                if not hit and _ocr_looks_garbled_overlap(text):
                    hit = _is_fused_overlap(
                        metrics,
                        junc_thr=max(0.60, jt - 0.06),
                        edge_thr=max(0.22, et - 0.03),
                        ips_max=imax + 0.15,
                    )
                if not hit:
                    continue
                # 竖排且边缘不够高时，必须 OCR 乱码才报（抑制正常竖排公差）
                if vertical and metrics.get("edge", 0) < 0.28 and not _ocr_looks_garbled_overlap(text):
                    continue
                fused_hits += 1
                used.add(idx)
                reason = f"fused_j={metrics['junc_ps']:.2f},e={metrics['edge']:.2f},ips={metrics['ips']:.2f}"
                collected.extend(
                    _emit_overlap_pair_from_bbox(
                        entity_id,
                        inst["bbox"],
                        text=text,
                        score=max(float(inst.get("confidence") or 0), min(1.0, 0.45 + metrics["junc_ps"] * 0.5)),
                        parent_id=f"fused{idx}",
                        stats=metrics,
                        reason=reason,
                        angle=get_instance_angle(inst),
                    )
                )

            # OCR 漏检区域：墨迹团兜底（大图也启用，但避开已有 keep_pair）
            if bool(overlap_cfg.get("fused_blob_scan", True)):
                existing_pairs = [c for c in collected if c.get("keep_pair")]
                blobs = _scan_fused_blobs(
                    analysis,
                    entity_id,
                    junc_thr=junc_thr,
                    edge_thr=edge_thr,
                    ips_max=ips_max,
                    min_side=min_side,
                    page_h=oh0,
                )
                extra_blob = 0
                i = 0
                while i < len(blobs):
                    b = blobs[i]
                    bb = b["bbox"]
                    # 成对输出：A 与 twin 连续
                    twin = blobs[i + 1] if i + 1 < len(blobs) and str(blobs[i + 1].get("parent_id") or "").endswith("_twin") else None
                    if _is_titleish_or_tiny_text(str(b.get("raw_text") or ""), bb, page_h=oh0):
                        i += 2 if twin is not None else 1
                        continue
                    if not str(b.get("raw_text") or "").strip():
                        if bb[1] > 0.70 * oh0 or bb[0] < 0.08 * ow0:
                            i += 2 if twin is not None else 1
                            continue
                        j = float((b.get("fields") or {}).get("junc_ps") or b.get("junc_ps") or 0)
                        e = float((b.get("fields") or {}).get("edge") or b.get("edge") or 0)
                        bw = max(1, bb[2] - bb[0])
                        bh = max(1, bb[3] - bb[1])
                        vertical = bh >= 1.4 * bw
                        if vertical and j >= 0.80 and e >= 0.25:
                            pass
                        elif j >= 0.76 and e >= 0.32:
                            pass
                        else:
                            i += 2 if twin is not None else 1
                            continue
                    if any(iou_xyxy(bb, p["bbox"]) > 0.25 for p in existing_pairs):
                        i += 2 if twin is not None else 1
                        continue
                    collected.append(b)
                    existing_pairs.append(b)
                    if twin is not None:
                        collected.append(twin)
                        existing_pairs.append(twin)
                    extra_blob += 1
                    i += 2 if twin is not None else 1
                fused_hits += extra_blob
                notes.append(f"fused_blobs={extra_blob}")
            else:
                notes.append("fused_blobs=0")
    if detect_overlap:
        for idx, inst in enumerate(ocr_insts):
            if idx in used:
                continue
            item = dict(inst)
            item["parent_id"] = f"ocr{idx}"
            item["fields"] = {**(item.get("fields") or {}), "ink_pixels": float(masks[idx].sum())}
            collected.append(item)

    notes.append(f"ink_pairs={pair_hits}")
    notes.append(f"ink_within={within_hits}")
    notes.append(f"fused_hits={int(fused_hits)}")

    # 小图允许更小框
    min_keep = 4 if max(ow0, oh0) < 200 else max(6, int(min_side * 0.5))
    collected = [c for c in collected if _box_side(c["bbox"]) >= min_keep]

    normal = [c for c in collected if not c.get("keep_pair")]
    pairs = [c for c in collected if c.get("keep_pair")]
    normal = _dedupe_boxes(normal, iou_thr=float(overlap_cfg.get("dedupe_iou", 0.88)))
    filtered_normal = []
    for n in normal:
        if any(iou_xyxy(n["bbox"], p["bbox"]) > 0.55 for p in pairs):
            continue
        filtered_normal.append(n)
    collected = filtered_normal + pairs

    meta.setdefault("width", ow0)
    meta.setdefault("height", oh0)
    meta["overlap_scale"] = 1.0

    instances = [_ensure_angle_fields(i) for i in reindex_instances(collected)]
    if parse_dims:
        apply_dimension_parse_to_instances(instances)
        # 尺寸解析后仍保留 angle
        instances = [_ensure_angle_fields(i) for i in instances]
    notes.append(f"final={len(instances)}")
    return {"backend": "number_overlap", "instances": instances, "notes": notes}
