"""倾斜矩形（OBB）：墨迹精修、旋转 IoU、仿射裁剪。"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _clamp_bbox(bbox: list[int], page_w: int, page_h: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(page_w - 1, x1))
    y1 = max(0, min(page_h - 1, y1))
    x2 = max(0, min(page_w, x2))
    y2 = max(0, min(page_h, y2))
    if x2 <= x1:
        x2 = min(page_w, x1 + 1)
    if y2 <= y1:
        y2 = min(page_h, y1 + 1)
    return [x1, y1, x2, y2]


def aabb_of_quad(quad: list[list[float]] | np.ndarray) -> list[int]:
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x1 = int(np.floor(pts[:, 0].min()))
    y1 = int(np.floor(pts[:, 1].min()))
    x2 = int(np.ceil(pts[:, 0].max()))
    y2 = int(np.ceil(pts[:, 1].max()))
    return [x1, y1, x2, y2]


def order_quad_clockwise(quad: list[list[float]] | np.ndarray) -> list[list[float]]:
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    ordered = pts[order]
    # 从左上开始（y 小优先，再 x 小）
    start = int(np.argmin(ordered[:, 1] + 0.01 * ordered[:, 0]))
    ordered = np.roll(ordered, -start, axis=0)
    return [[float(x), float(y)] for x, y in ordered]


def refine_aabb_to_obb(
    image: Image.Image,
    bbox: list[int],
    *,
    ink_threshold: int = 245,
    pad: int = 2,
    min_ink_pixels: int = 20,
) -> dict[str, Any] | None:
    """在 AABB 内用墨迹拟合最小面积旋转矩形。

    返回 {bbox, quad, angle}；失败返回 None。
    angle：矩形长边相对水平的朝向角（度，约 -90~90）。
    """
    import cv2

    page_w, page_h = image.size
    x1, y1, x2, y2 = _clamp_bbox(list(bbox), page_w, page_h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    gray = np.asarray(image.crop((x1, y1, x2, y2)).convert("L"), dtype=np.uint8)
    ink = (gray < int(ink_threshold)).astype(np.uint8) * 255
    if int((ink > 0).sum()) < int(min_ink_pixels):
        return None
    # 取最大连通域，避免框内噪声把 OBB 撑大
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if nlab <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    mask = (labels == best).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), ang = rect
    if rw < 1 or rh < 1:
        return None
    # OpenCV angle：使 width 边的朝向；统一成长边近水平的角度
    if rw < rh:
        rw, rh = rh, rw
        ang = ang + 90.0
    while ang <= -90.0:
        ang += 180.0
    while ang > 90.0:
        ang -= 180.0
    box = cv2.boxPoints(((cx, cy), (rw + 2 * pad, rh + 2 * pad), ang))
    # 映射回全图坐标
    box[:, 0] += float(x1)
    box[:, 1] += float(y1)
    quad = order_quad_clockwise(box)
    aabb = _clamp_bbox(aabb_of_quad(quad), page_w, page_h)
    return {"bbox": aabb, "quad": quad, "angle": float(ang)}


def iou_quad(a: list[list[float]], b: list[list[float]]) -> float:
    """两凸四边形 IoU（Skew-IoU）。"""
    import cv2

    pa = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    pb = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    if pa.shape[0] < 3 or pb.shape[0] < 3:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    if inter_area is None:
        inter = 0.0
    else:
        inter = float(inter_area)
    area_a = float(abs(cv2.contourArea(pa)))
    area_b = float(abs(cv2.contourArea(pb)))
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def warp_quad_crop(
    image: Image.Image,
    quad: list[list[float]],
    *,
    pad: int = 4,
    out_height: int | None = None,
) -> Image.Image:
    """将倾斜四边形仿射拉正为水平矩形图，供 Pass2 识读。"""
    import cv2

    pts = np.asarray(order_quad_clockwise(quad), dtype=np.float32)
    # 边长：取相邻边估计宽高
    w1 = float(np.linalg.norm(pts[1] - pts[0]))
    w2 = float(np.linalg.norm(pts[2] - pts[3]))
    h1 = float(np.linalg.norm(pts[3] - pts[0]))
    h2 = float(np.linalg.norm(pts[2] - pts[1]))
    width = max(8, int(round(max(w1, w2))))
    height = max(8, int(round(max(h1, h2))))
    if out_height is not None and height < int(out_height):
        scale = float(out_height) / float(height)
        width = max(8, int(round(width * scale)))
        height = int(out_height)
    dst = np.array(
        [[pad, pad], [width + pad, pad], [width + pad, height + pad], [pad, height + pad]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    arr = np.asarray(image.convert("RGB"))
    warped = cv2.warpPerspective(
        arr,
        M,
        (width + 2 * pad, height + 2 * pad),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(warped)


def apply_obb_to_instances(
    image: Image.Image,
    instances: list[dict[str, Any]],
    *,
    entity_ids: set[str] | None = None,
    ink_threshold: int = 245,
    pad: int = 2,
) -> list[dict[str, Any]]:
    """对指定实体做 OBB 精修；写回 bbox/quad/angle。"""
    eids = entity_ids or {"number_mark"}
    out: list[dict[str, Any]] = []
    for inst in instances or []:
        item = dict(inst)
        if str(item.get("entity_id") or "") not in eids:
            out.append(item)
            continue
        bb = item.get("bbox")
        if not bb or len(bb) != 4:
            out.append(item)
            continue
        refined = refine_aabb_to_obb(
            image,
            list(bb),
            ink_threshold=ink_threshold,
            pad=pad,
        )
        if refined is None:
            out.append(item)
            continue
        item["bbox"] = refined["bbox"]
        item["quad"] = refined["quad"]
        item["angle"] = refined["angle"]
        fields = dict(item.get("fields") or {})
        if fields.get("angle") in (None, ""):
            fields["angle"] = refined["angle"]
        item["fields"] = fields
        out.append(item)
    return out
