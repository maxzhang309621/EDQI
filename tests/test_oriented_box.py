"""倾斜框 OBB / Skew-IoU / NMS 邻接斜字场景。"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from pipeline.oriented_box import (
    apply_obb_to_instances,
    iou_quad,
    refine_aabb_to_obb,
    warp_quad_crop,
)
from pipeline.perceive_utils import filter_dimension_accuracy, nms_instances


def _draw_tilted_text_blob(img: Image.Image, cx: float, cy: float, angle_deg: float) -> None:
    """在空白图上画一段倾斜墨迹（模拟斜字）。"""
    arr = np.asarray(img.convert("L")).copy()
    h, w = arr.shape
    # 在局部坐标系画细长矩形再旋转
    length, thickness = 48, 10
    yy, xx = np.mgrid[-thickness // 2 : thickness // 2 + 1, -length // 2 : length // 2 + 1]
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    xr = xx * cos_a - yy * sin_a + cx
    yr = xx * sin_a + yy * cos_a + cy
    for x, y in zip(xr.ravel(), yr.ravel()):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            arr[yi, xi] = 0
    out = Image.fromarray(arr, mode="L").convert("RGB")
    img.paste(out)


def test_refine_aabb_to_obb_tilted_ink():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    _draw_tilted_text_blob(img, 100, 100, 35.0)
    # 故意给一个偏大的 AABB（模拟倾斜字外接轴对齐框膨胀）
    loose = [60, 60, 140, 140]
    refined = refine_aabb_to_obb(img, loose, ink_threshold=245, pad=1)
    assert refined is not None
    assert refined.get("quad") and len(refined["quad"]) == 4
    aabb = refined["bbox"]
    loose_area = (loose[2] - loose[0]) * (loose[3] - loose[1])
    tight_area = (aabb[2] - aabb[0]) * (aabb[3] - aabb[1])
    assert tight_area < loose_area
    assert abs(float(refined["angle"])) > 10.0


def test_iou_quad_identical_and_disjoint():
    q = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
    assert iou_quad(q, q) > 0.99
    q2 = [[100.0, 100.0], [110.0, 100.0], [110.0, 105.0], [100.0, 105.0]]
    assert iou_quad(q, q2) < 0.01


def test_nms_skew_keeps_adjacent_tilted_neighbors():
    """两斜字 AABB 重叠高，但 OBB 几乎不交 → Skew-NMS 应都保留。"""
    # 两个倾斜细长框，中心接近、轴对齐外接矩形高度重叠
    a_quad = [[10, 50], [70, 10], [76, 18], [16, 58]]
    b_quad = [[40, 55], [100, 15], [106, 23], [46, 63]]
    a_bb = [10, 10, 76, 58]
    b_bb = [40, 15, 106, 63]
    from pipeline.perceive_utils import iou_xyxy

    assert iou_xyxy(a_bb, b_bb) > 0.25  # AABB 会互相压制
    insts = [
        {
            "entity_id": "number_mark",
            "instance_id": "a",
            "bbox": a_bb,
            "quad": a_quad,
            "confidence": 0.9,
            "fields": {"text": "12"},
        },
        {
            "entity_id": "number_mark",
            "instance_id": "b",
            "bbox": b_bb,
            "quad": b_quad,
            "confidence": 0.85,
            "fields": {"text": "34"},
        },
    ]
    aabb_nms = nms_instances(insts, 0.25, prefer_smaller=True, use_quad=False)
    skew_nms = nms_instances(insts, 0.25, prefer_smaller=True, use_quad=True)
    assert len(aabb_nms) == 1
    assert len(skew_nms) == 2


def test_filter_dimension_accuracy_dedupe_and_weak():
    insts = [
        {
            "entity_id": "number_mark",
            "bbox": [0, 0, 20, 10],
            "fields": {"text": "Ø10"},
            "confidence": 0.9,
        },
        {
            "entity_id": "number_mark",
            "bbox": [5, 2, 25, 12],
            "fields": {"text": "ø10"},
            "confidence": 0.8,
        },
        {
            "entity_id": "number_mark",
            "bbox": [100, 100, 120, 110],
            "fields": {"text": "abc"},
            "confidence": 0.5,
        },
    ]
    out, notes = filter_dimension_accuracy(insts, drop_weak=True, dedupe_same_text=True, center_dist_thr=28)
    assert len(out) == 1
    assert any("dedupe" in n or "weak" in n for n in notes)


def test_warp_quad_crop_and_apply_obb():
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 水平墨迹
    draw.rectangle([40, 70, 120, 90], fill=(0, 0, 0))
    refined = refine_aabb_to_obb(img, [30, 60, 130, 100])
    assert refined is not None
    crop = warp_quad_crop(img, refined["quad"], pad=2)
    assert crop.size[0] >= 8 and crop.size[1] >= 8
    insts = apply_obb_to_instances(
        img,
        [{"entity_id": "number_mark", "bbox": [30, 60, 130, 100], "fields": {}}],
    )
    assert insts[0].get("quad")
