"""尺寸属性 backend=vlm：路由、字段补齐、合并只留 keep_pair。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.dimension_parse import enrich_dimension_fields_from_text
from pipeline.drawing_parse_plan import (
    build_drawing_parse_plan,
    dimension_marks_backend,
    is_dimension_marks_vlm,
    merge_perception_plans,
    split_plan_for_backends,
)
from pipeline import load_config
from engines import collect_entities, load_rules
from pipeline.perceive_qwen_vl import (
    _dimension_marks_pass2_prompt,
    _is_dimension_marks_entity,
)


def test_default_backend_is_vlm():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert is_dimension_marks_vlm(cfg) is True
    assert dimension_marks_backend(cfg) == "vlm"
    plan = [
        {
            "entity_id": "number_mark",
            "parse_kind": "dimension_marks",
            "backend": "vlm",
            "locate_query": "尺寸",
            "fields": [],
        },
        {"entity_id": "main_table", "parse_kind": "table", "fields": []},
    ]
    ocr, vl = split_plan_for_backends(plan)
    assert any(e.get("parse_kind") == "dimension_marks" for e in vl)
    assert not any(e.get("parse_kind") == "dimension_marks" for e in ocr)


def test_merged_plan_splits_rule_ocr_and_vlm_dims():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = merge_perception_plans(
        collect_entities(load_rules(ROOT / "rules" / "library", only_active=True)),
        build_drawing_parse_plan(cfg),
    )
    ocr, vl = split_plan_for_backends(plan)
    # 默认 vlm：尺寸进 VL；规则 number_mark 仍可在 OCR（重叠）
    assert any(e.get("parse_kind") == "dimension_marks" for e in vl)
    assert not any(e.get("parse_kind") == "dimension_marks" for e in ocr)
    assert any(e.get("entity_id") == "number_mark" for e in vl)
    assert {e["entity_id"] for e in vl} >= {"number_mark", "material_table", "main_table"}


def test_enrich_preserves_vlm_values_and_fills_angle():
    fields = enrich_dimension_fields_from_text(
        {"text": "12±0.1", "dim_kind": "diameter", "basic_size": "12"},
        "12±0.1",
        bbox=[10, 10, 20, 80],
    )
    assert fields["dim_kind"] == "diameter"
    assert fields["basic_size"] == "12"
    assert fields["tolerance"] == "0.1"
    assert fields["has_tolerance"] is True
    assert fields["angle"] == 90.0


def test_dimension_pass2_prompt_mentions_exclusions():
    ent = {
        "parse_kind": "dimension_marks",
        "fields": [
            {"name": "text", "parse_hint": "原文"},
            {"name": "dim_kind", "parse_hint": "diameter|radius"},
        ],
    }
    assert _is_dimension_marks_entity(ent)
    prompt = _dimension_marks_pass2_prompt(ent)
    assert "dim_kind" in prompt
    assert "angle" in prompt
    assert "表格" in prompt


def test_vlm_finalize_drops_marks_inside_table():
    from pipeline.perceive_qwen_vl import _finalize_dimension_instances

    plan = [
        {
            "entity_id": "number_mark",
            "parse_kind": "dimension_marks",
            "exclude_table_regions": True,
            "exclude_table_pad": 0,
            "exclude_table_expand_up": 0.0,
            "strict_fields_only": True,
            "require_dim_kind": True,
            "require_basic_size": True,
            "fields": [
                {"name": "text"},
                {"name": "dim_kind"},
                {"name": "basic_size"},
            ],
        },
        {"entity_id": "main_table", "parse_kind": "table", "fields": []},
    ]
    instances = [
        {
            "entity_id": "main_table",
            "bbox": [200, 200, 600, 600],
            "fields": {},
        },
        {
            "entity_id": "number_mark",
            "bbox": [10, 10, 40, 28],
            "raw_text": "R5",
            "fields": {"text": "R5", "dim_kind": "radius", "basic_size": "5"},
        },
        {
            "entity_id": "number_mark",
            "bbox": [300, 300, 340, 320],
            "raw_text": "12",
            "fields": {"text": "12", "dim_kind": "length", "basic_size": "12"},
        },
    ]
    out = _finalize_dimension_instances(instances, plan, page_w=800, page_h=800)
    marks = [i for i in out if i.get("entity_id") == "number_mark"]
    assert len(marks) == 1
    assert marks[0]["raw_text"] == "R5"
    assert marks[0]["fields"].get("vlm_filtered") is True


def test_vlm_merge_keeps_only_overlap_pairs():
    """模拟 run.py：backend=vlm 时 OCR 实例只保留 keep_pair。"""
    ocr_instances = [
        {"entity_id": "number_mark", "keep_pair": True, "fields": {"text": "a"}},
        {"entity_id": "number_mark", "keep_pair": False, "fields": {"text": "noise"}},
        {"entity_id": "number_mark", "fields": {"text": "also_noise"}},
    ]
    kept = [i for i in ocr_instances if i.get("keep_pair")]
    assert len(kept) == 1
    assert kept[0]["fields"]["text"] == "a"
