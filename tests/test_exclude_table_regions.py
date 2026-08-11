"""非表格尺寸：排除表格框内 OCR 数字。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.perceive_common import (
    build_table_exclude_regions,
    collect_table_bboxes,
    collect_table_value_tokens,
    filter_instances_matching_table_values,
    filter_instances_outside_bboxes,
    text_matches_table_value,
)
from pipeline.perceive_number_overlap import _looks_like_number_mark


def test_collect_table_bboxes():
    boxes = collect_table_bboxes(
        [
            {"entity_id": "material_table", "bbox": [100, 100, 400, 300]},
            {"entity_id": "number_mark", "bbox": [10, 10, 50, 40]},
            {"entity_id": "main_table", "bbox": [100, 300, 400, 500]},
            {"entity_id": "main_table", "bbox": [0, 0, 0, 0]},
        ]
    )
    assert boxes == [[100, 100, 400, 300], [100, 300, 400, 500]]


def test_filter_drops_marks_inside_table():
    table = [[200, 200, 600, 600]]
    instances = [
        {"entity_id": "number_mark", "bbox": [10, 10, 40, 40], "fields": {"text": "12"}},
        {"entity_id": "number_mark", "bbox": [300, 300, 340, 340], "fields": {"text": "1:2"}},
        {"entity_id": "number_mark", "bbox": [650, 650, 690, 690], "fields": {"text": "8"}},
    ]
    kept, dropped = filter_instances_outside_bboxes(instances, table, pad=0)
    assert dropped == 1
    texts = [i["fields"]["text"] for i in kept]
    assert texts == ["12", "8"]


def test_filter_pad_expands_exclusion():
    table = [[100, 100, 200, 200]]
    # 中心 (95, 150) 在 pad=0 外、pad=6 内
    inst = [{"entity_id": "number_mark", "bbox": [90, 145, 100, 155]}]
    kept0, d0 = filter_instances_outside_bboxes(inst, table, pad=0)
    kept6, d6 = filter_instances_outside_bboxes(inst, table, pad=6)
    assert d0 == 0 and len(kept0) == 1
    assert d6 == 1 and len(kept6) == 0


def test_material_above_cells_expand_up():
    regions = build_table_exclude_regions(
        [
            {
                "entity_id": "material_table",
                "bbox": [1000, 1000, 1800, 1100],
                "fields": {"read_mode": "above_cells"},
            }
        ],
        page_w=2000,
        page_h=1400,
        pad=0,
        expand_up_frac=0.1,
    )
    # 上扩 140px → y1=860
    assert any(r[1] <= 860 for r in regions)


def test_filter_matching_document_number():
    tokens = collect_table_value_tokens(
        [
            {
                "entity_id": "main_table",
                "bbox": [1, 1, 10, 10],
                "fields": {"document_number": "25001745002A", "page": "1/1"},
            }
        ]
    )
    assert text_matches_table_value("25001745002A", tokens)
    insts = [
        {"entity_id": "number_mark", "raw_text": "25001745002A", "bbox": [10, 10, 50, 30]},
        {"entity_id": "number_mark", "raw_text": "12±0.1", "bbox": [10, 10, 50, 30]},
    ]
    kept, dropped = filter_instances_matching_table_values(insts, tokens)
    assert dropped == 1
    assert kept[0]["raw_text"] == "12±0.1"


def test_looks_like_rejects_tableish_text():
    assert _looks_like_number_mark("12±0.1") is True
    assert _looks_like_number_mark("R5") is True
    assert _looks_like_number_mark("25001745002A") is False
    assert _looks_like_number_mark("25001743T.2") is False
    assert _looks_like_number_mark("Siemens 2025") is False
    assert _looks_like_number_mark("-sheet DINEN13599 Cu-ETP-R290") is False


def test_dimension_marks_default_exclude_flag():
    from pipeline import load_config
    from pipeline.drawing_parse_plan import build_drawing_parse_plan

    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = build_drawing_parse_plan(cfg, enabled=True)
    dim = next(e for e in plan if e.get("parse_kind") == "dimension_marks")
    assert dim.get("exclude_table_regions") is True
