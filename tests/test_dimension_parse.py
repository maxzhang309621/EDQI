"""尺寸标注类型与公差解析单测。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.dimension_parse import (
    apply_dimension_parse_to_instances,
    merge_dimension_fields,
    parse_dimension_text,
)

# 使用 Unicode 转义，避免 Windows 下源文件编码把 °/±/Ø 弄坏
DEG = "\u00b0"
PM = "\u00b1"
DIA = "\u00d8"
DIA2 = "\u2300"
PHI = "\u03a6"
O_SLASH = "\u00f8"


def test_diameter_with_tolerance():
    out = parse_dimension_text(f"{DIA}12{PM}0.1")
    assert out["dim_kind"] == "diameter"
    assert out["basic_size"] == "12"
    assert out["tolerance"] == "0.1"
    assert out["has_tolerance"] is True


def test_diameter_ocr_variants():
    for text in (f"{DIA2}10", f"{PHI}8.5", f"{O_SLASH}20"):
        out = parse_dimension_text(text)
        assert out["dim_kind"] == "diameter", text
        assert out["has_tolerance"] is False
        assert out["tolerance"] is None
        assert out["basic_size"] is not None


def test_radius_no_tolerance():
    out = parse_dimension_text("R5")
    assert out == {
        "dim_kind": "radius",
        "basic_size": "5",
        "tolerance": None,
        "has_tolerance": False,
    }


def test_radius_with_plus_minus_slash():
    out = parse_dimension_text("R10+/-0.2")
    assert out["dim_kind"] == "radius"
    assert out["basic_size"] == "10"
    assert out["tolerance"] == "0.2"
    assert out["has_tolerance"] is True


def test_length_plain():
    out = parse_dimension_text("25.5")
    assert out["dim_kind"] == "length"
    assert out["basic_size"] == "25.5"
    assert out["tolerance"] is None
    assert out["has_tolerance"] is False


def test_length_with_tolerance():
    out = parse_dimension_text(f"100{PM}0.05")
    assert out["dim_kind"] == "length"
    assert out["basic_size"] == "100"
    assert out["tolerance"] == "0.05"
    assert out["has_tolerance"] is True


def test_angle():
    out = parse_dimension_text(f"30{DEG}")
    assert out["dim_kind"] == "angle"
    assert out["basic_size"] == "30"
    assert out["has_tolerance"] is False


def test_ocr_symbol_misreads_diameter_and_angle():
    """OCR 常把 Ø 读成 O/Q/D，把 ° 读成 o。"""
    assert parse_dimension_text("O12")["dim_kind"] == "diameter"
    assert parse_dimension_text("Q8.5")["basic_size"] == "8.5"
    assert parse_dimension_text("45o")["dim_kind"] == "angle"
    assert parse_dimension_text("30O")["dim_kind"] == "angle"


def test_dedupe_prefers_specific_kind():
    from pipeline.dimension_parse import dedupe_dimension_attribute_instances

    a = {
        "entity_id": "number_mark",
        "bbox": [10, 10, 40, 30],
        "confidence": 0.95,
        "raw_text": "12",
        "fields": {"text": "12", "dim_kind": "length", "basic_size": "12"},
    }
    b = {
        "entity_id": "number_mark",
        "bbox": [12, 11, 42, 31],
        "confidence": 0.8,
        "raw_text": "O12",
        "fields": {"text": "O12", "dim_kind": "diameter", "basic_size": "12", "vlm_filtered": True},
    }
    out, n = dedupe_dimension_attribute_instances([a, b])
    marks = [i for i in out if i.get("entity_id") == "number_mark"]
    assert n == 1
    assert len(marks) == 1
    assert marks[0]["fields"]["dim_kind"] == "diameter"


def test_overlap_instances_skip_dimension_parse():
    """重叠 keep_pair 只标跳过，不做 dim_kind/basic_size 等属性解析。"""
    text_ok = f"{DIA}12{PM}0.1"
    text_ov = f"{DIA}8{PM}0.1"
    instances = [
        {
            "keep_pair": False,
            "raw_text": text_ok,
            "fields": {"text": text_ok},
        },
        {
            "keep_pair": True,
            "raw_text": text_ov,
            "fields": {"text": text_ov, "dim_kind": "diameter"},
        },
    ]
    apply_dimension_parse_to_instances(instances)
    assert instances[0]["fields"]["dim_kind"] == "diameter"
    assert instances[0]["fields"]["has_tolerance"] is True
    assert "dimension_parse_skipped" not in instances[0]["fields"]
    assert instances[1]["fields"].get("dimension_parse_skipped") == "overlap"
    assert "dim_kind" not in instances[1]["fields"]
    assert "basic_size" not in instances[1]["fields"]
    assert instances[1]["fields"]["text"] == text_ov


def test_angle_with_tolerance():
    out = parse_dimension_text(f"45{DEG}{PM}1")
    assert out["dim_kind"] == "angle"
    assert out["basic_size"] == "45"
    assert out["tolerance"] == "1"
    assert out["has_tolerance"] is True


def test_merge_preserves_existing_keys():
    text = f"{DIA}12{PM}0.1"
    fields = merge_dimension_fields({"text": text, "angle": 0.0}, text)
    assert fields["text"] == text
    assert fields["angle"] == 0.0
    assert fields["dim_kind"] == "diameter"
    assert fields["has_tolerance"] is True


def test_empty_and_non_numeric():
    assert parse_dimension_text("")["dim_kind"] is None
    assert parse_dimension_text("abc")["dim_kind"] is None
