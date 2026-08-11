"""文本相对水平线的朝向角估计与 OCR 角度自适应。"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


def normalize_text_angle(deg: float) -> float:
    """归一化到 (-90, 90]：水平≈0，竖排≈±90。"""
    a = float(deg)
    while a <= -90.0:
        a += 180.0
    while a > 90.0:
        a -= 180.0
    return round(a, 1)


def text_angle_from_quad(box: Any) -> float:
    """由 OCR 四边形最长边估计文本相对水平线的角度（度）。"""
    try:
        pts = [(float(p[0]), float(p[1])) for p in box]
    except (TypeError, ValueError, IndexError):
        return 0.0
    if len(pts) < 2:
        return 0.0
    best_len2 = -1.0
    best_ang = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        len2 = dx * dx + dy * dy
        if len2 > best_len2:
            best_len2 = len2
            best_ang = math.degrees(math.atan2(dy, dx))
    return normalize_text_angle(best_ang)


def text_angle_in_original(quad_angle: float, page_rotate: float) -> float:
    """页图旋转 page_rotate（CCW）后检测到的四边形角 → 原图文本角。"""
    return normalize_text_angle(float(quad_angle) - float(page_rotate))


def text_angle_from_bbox(bbox: list[Any] | None) -> float:
    """无四边形时，用外接框长宽比粗估：竖条→90，否则→0。"""
    if not bbox or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return 0.0
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if h >= 1.35 * w:
        return 90.0
    if w >= 1.35 * h:
        return 0.0
    return 0.0


def get_instance_angle(inst: dict[str, Any] | None) -> float:
    if not isinstance(inst, dict):
        return 0.0
    fields = inst.get("fields") or {}
    if isinstance(fields, dict) and fields.get("angle") is not None:
        try:
            return normalize_text_angle(float(fields["angle"]))
        except (TypeError, ValueError):
            pass
    if inst.get("angle") is not None:
        try:
            return normalize_text_angle(float(inst["angle"]))
        except (TypeError, ValueError):
            pass
    return text_angle_from_bbox(inst.get("bbox"))


def suggest_extra_page_angles(
    text_angles: Iterable[float],
    *,
    existing: Iterable[float] | None = None,
    step: float = 15.0,
    min_count: int = 2,
    max_extra: int = 4,
) -> list[float]:
    """根据已检出文本角，建议额外整页旋转角（使倾斜文本趋近水平）。

    页旋转 α 满足：原图文本角 θ + α ≈ 0 → α ≈ -θ。
    """
    have = {normalize_text_angle(a) for a in (existing or [])}
    # 也把常见竖排角算作已覆盖
    for a in (0.0, 90.0, -90.0):
        have.add(normalize_text_angle(a))

    buckets: Counter[int] = Counter()
    step = max(5.0, float(step))
    for ang in text_angles:
        a = normalize_text_angle(ang)
        if abs(a) < 12.0 or abs(a) > 78.0:
            continue
        b = int(round(a / step) * step)
        if b <= -90:
            b += 180
        if b > 90:
            b -= 180
        buckets[b] += 1

    extras: list[float] = []
    for bucket, cnt in buckets.most_common():
        if cnt < min_count:
            continue
        page = normalize_text_angle(-float(bucket))
        # 与已有角太近则跳过
        if any(abs(page - h) < step * 0.5 for h in have):
            continue
        extras.append(page)
        have.add(page)
        if len(extras) >= max_extra:
            break
    return extras
