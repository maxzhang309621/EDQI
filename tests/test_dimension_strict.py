"""尺寸属性严格过滤：只保留配置允许的合法尺寸。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.dimension_parse import (
    bbox_geometry_ok,
    clip_fields_to_schema,
    is_valid_dimension_mark,
)
from pipeline.perceive_qwen_vl import _finalize_dimension_instances


def test_clip_fields_to_schema():
    out = clip_fields_to_schema(
        {"text": "12", "dim_kind": "length", "foo": 1, "basic_size": "12"},
        ["text", "dim_kind", "basic_size"],
    )
    assert "foo" not in out
    assert out["dim_kind"] == "length"


def test_reject_non_dimension_and_part_no():
    assert is_valid_dimension_mark({"text": "12±0.1", "dim_kind": "length", "basic_size": "12"})
    assert not is_valid_dimension_mark({"text": "25001745002A", "dim_kind": "length", "basic_size": "25001745002"})
    assert not is_valid_dimension_mark({"text": "Siemens 2025", "dim_kind": None, "basic_size": None})
    assert not is_valid_dimension_mark({"text": "abc", "dim_kind": None})


def test_reject_wide_bbox():
    assert bbox_geometry_ok([10, 10, 40, 30], page_w=2048, page_h=1448)
    assert not bbox_geometry_ok([1024, 100, 2048, 140], page_w=2048, page_h=1448)


def test_finalize_drops_invalid():
    plan = [
        {
            "entity_id": "number_mark",
            "parse_kind": "dimension_marks",
            "strict_fields_only": True,
            "require_dim_kind": True,
            "require_basic_size": True,
            "exclude_table_regions": True,
            "max_bbox_width_ratio": 0.28,
            "max_aspect_ratio": 8.0,
            "fields": [
                {"name": "text"},
                {"name": "dim_kind"},
                {"name": "basic_size"},
                {"name": "angle"},
            ],
        },
        {"entity_id": "main_table", "parse_kind": "table", "fields": []},
    ]
    instances = [
        {
            "entity_id": "number_mark",
            "bbox": [100, 100, 140, 130],
            "fields": {"text": "R5", "dim_kind": "radius", "basic_size": "5", "extra": 1},
            "raw_text": "R5",
        },
        {
            "entity_id": "number_mark",
            "bbox": [1024, 200, 2048, 240],
            "fields": {"text": "11±0.1", "dim_kind": "length", "basic_size": "11"},
            "raw_text": "11±0.1",
        },
        {
            "entity_id": "number_mark",
            "bbox": [50, 50, 80, 70],
            "fields": {"text": "noise", "dim_kind": None},
            "raw_text": "noise",
        },
        {
            "entity_id": "main_table",
            "bbox": [1000, 1000, 1800, 1400],
            "fields": {"page": "1/1"},
        },
    ]
    notes: list[str] = []
    out = _finalize_dimension_instances(instances, plan, page_w=2048, page_h=1448, notes=notes)
    eids = [i["entity_id"] for i in out]
    assert eids.count("number_mark") == 1
    assert "main_table" in eids
    mark = next(i for i in out if i["entity_id"] == "number_mark")
    assert "extra" not in (mark.get("fields") or {})
    assert any(n.startswith("dimension_strict_dropped=") for n in notes)
