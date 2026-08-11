"""文本相对水平线角度估计与 OCR 自适应角。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.build_facts import build_facts
from pipeline.dimension_parse import merge_dimension_fields
from pipeline.text_angle import (
    get_instance_angle,
    normalize_text_angle,
    suggest_extra_page_angles,
    text_angle_from_bbox,
    text_angle_from_quad,
    text_angle_in_original,
)


def test_normalize_and_quad_horizontal():
    box = [[10, 10], [50, 10], [50, 24], [10, 24]]
    assert abs(text_angle_from_quad(box)) < 1.0


def test_quad_vertical():
    box = [[10, 10], [22, 10], [22, 60], [10, 60]]
    ang = text_angle_from_quad(box)
    assert abs(abs(ang) - 90.0) < 1.0 or abs(ang) < 1.0  # 长边竖或横视点序
    # 强制长边为竖：交换使竖直边更长
    box_v = [[10, 10], [18, 10], [18, 80], [10, 80]]
    # 最长边是左右短边? w=8 h=70 → vertical edges longer
    # edges: (10,10)-(18,10) len=8; (18,10)-(18,80) len=70 → 90°
    assert abs(abs(text_angle_from_quad(box_v)) - 90.0) < 1.0


def test_quad_tilted_45():
    # 长边约 45°
    box = [[0, 0], [40, 40], [34, 46], [-6, 6]]
    ang = text_angle_from_quad(box)
    assert 30.0 <= abs(ang) <= 60.0


def test_page_rotate_composition():
    # 页旋 +90 后四边形角为 0 → 原图竖排约 -90 或等价
    assert abs(abs(text_angle_in_original(0.0, 90.0)) - 90.0) < 1e-6


def test_suggest_extra_page_angles():
    extras = suggest_extra_page_angles([45, 44, 46, 0, 90], existing=[0, 90, -90], step=15)
    assert any(abs(a + 45) < 1 or abs(a - (-45)) < 1 for a in extras)


def test_bbox_vertical_estimate():
    assert text_angle_from_bbox([10, 10, 20, 80]) == 90.0
    assert text_angle_from_bbox([10, 10, 80, 20]) == 0.0


def test_angle_preserved_in_dimension_merge_and_facts():
    fields = merge_dimension_fields({"text": "12±0.1", "angle": -90.0}, "12±0.1")
    assert fields["angle"] == -90.0
    assert fields["dim_kind"] == "length"
    payload = {
        "backend": "number_overlap",
        "instances": [
            {
                "entity_id": "number_mark",
                "instance_id": "number_mark#0",
                "label": "数字标注",
                "bbox": [1, 2, 3, 40],
                "fields": {"text": "12±0.1", "angle": 90.0, "dim_kind": "length"},
                "raw_text": "12±0.1",
                "confidence": 0.9,
                "needs_review": False,
                "angle": 90.0,
            }
        ],
    }
    facts = build_facts(payload, {"width": 100, "height": 100, "drawing_id": "ang"}, validate=False)
    ann = facts["annotations"][0]
    assert ann["angle"] == 90.0
    assert get_instance_angle({"fields": ann}) == 90.0


def test_normalize_wrap():
    assert normalize_text_angle(135) == -45.0
    assert normalize_text_angle(-135) == 45.0
