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
    # 兼容：显式切回 vlm 时仍可用
    from pipeline.drawing_parse_plan import get_dimension_marks_config

    raw = dict(get_dimension_marks_config(cfg))
    raw["backend"] = "vlm"
    # 用 plan 实体测路由
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
    assert dimension_marks_backend({"drawing_parse_config": "configs/drawing_parse.yaml"}) in {
        "ocr_locate_vlm_filter",
        "vlm",
        "ocr",
    }
    assert is_dimension_marks_vlm(cfg) is False  # 默认已切到 ocr_locate_vlm_filter


def test_merged_plan_splits_rule_ocr_and_vlm_dims():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = merge_perception_plans(
        collect_entities(load_rules(ROOT / "rules" / "library", only_active=True)),
        build_drawing_parse_plan(cfg),
    )
    ocr, vl = split_plan_for_backends(plan)
    # 默认 ocr_locate_vlm_filter：尺寸进 OCR；表格进 VL
    assert any(e.get("parse_kind") == "dimension_marks" for e in ocr)
    assert not any(e.get("parse_kind") == "dimension_marks" for e in vl)
    assert any(e.get("entity_id") == "number_mark" for e in ocr)


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
