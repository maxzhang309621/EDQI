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
    assert not is_valid_dimension_mark({"text": "Rz5", "dim_kind": "radius", "basic_size": "5"})
    assert not is_valid_dimension_mark({"text": "A", "dim_kind": "diameter", "basic_size": "A"})
    assert not is_valid_dimension_mark({"text": "7295.9", "dim_kind": "length", "basic_size": "7295.9"})
    assert not is_valid_dimension_mark({"text": "(5)", "dim_kind": "length", "basic_size": "5"})
    assert not is_valid_dimension_mark({"text": "+0.2", "dim_kind": "length", "basic_size": "0.2"})
    assert not is_valid_dimension_mark({"text": "Max. 3", "dim_kind": "length", "basic_size": "3"})
    assert not is_valid_dimension_mark({"text": "max.3", "dim_kind": "length", "basic_size": "3"})
    assert not is_valid_dimension_mark({"text": "TYP 5", "dim_kind": "length", "basic_size": "5"})
    assert not is_valid_dimension_mark({"text": "MIN 0.2", "dim_kind": "length", "basic_size": "0.2"})
    assert not is_valid_dimension_mark({"text": "5:1", "dim_kind": "length", "basic_size": "5"})


def test_reject_wide_bbox():
    assert bbox_geometry_ok([10, 10, 40, 30], page_w=2048, page_h=1448)
    assert not bbox_geometry_ok([1024, 100, 2048, 140], page_w=2048, page_h=1448)
    # 分块级半页假框
    assert not bbox_geometry_ok([1024, 0, 2048, 1169], page_w=2048, page_h=1448)
    assert not bbox_geometry_ok([0, 1024, 1280, 1169], page_w=2048, page_h=1448)


def test_finalize_enriches_without_dropping():
    """与 main 对齐：finalize 只补字段，不因严格规则丢弃候选。"""
    plan = [
        {
            "entity_id": "number_mark",
            "parse_kind": "dimension_marks",
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
            "fields": {"text": "R5"},
            "raw_text": "R5",
        },
        {
            "entity_id": "number_mark",
            "bbox": [1024, 200, 2048, 240],
            "fields": {"text": "11±0.1"},
            "raw_text": "11±0.1",
        },
        {
            "entity_id": "main_table",
            "bbox": [1000, 1000, 1800, 1400],
            "fields": {"page": "1/1"},
        },
    ]
    out = _finalize_dimension_instances(instances, plan, page_w=2048, page_h=1448)
    eids = [i["entity_id"] for i in out]
    assert eids.count("number_mark") == 2
    assert "main_table" in eids
    mark = next(i for i in out if (i.get("fields") or {}).get("text") == "R5")
    assert mark["fields"].get("dim_kind") == "radius"
    assert mark["fields"].get("basic_size") == "5"
