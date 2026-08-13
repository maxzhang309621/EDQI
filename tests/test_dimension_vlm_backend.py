"""尺寸属性 backend=vlm：路由、字段补齐、合并只留 keep_pair、倾斜 deskew。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.dimension_parse import enrich_dimension_fields_from_text
from pipeline.drawing_parse_plan import (
    build_drawing_parse_plan,
    dimension_marks_backend,
    get_dimension_marks_config,
    is_dimension_marks_vlm,
    merge_perception_plans,
    split_plan_for_backends,
)
from pipeline import load_config
from engines import collect_entities, load_rules
from pipeline.perceive_qwen_vl import (
    _deskew_crop_for_vlm,
    _dimension_marks_pass2_prompt,
    _dimension_pass2_angles,
    _dimension_read_looks_weak,
    _is_dimension_marks_entity,
)


def test_default_backend_is_vlm():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert dimension_marks_backend(cfg) == "vlm"
    assert is_dimension_marks_vlm(cfg) is True


def test_merged_plan_splits_rule_ocr_and_vlm_dims():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    # 规则库 NUM_TEXT 当前为 draft；测路由时用 only_active=False 取其实体
    plan = merge_perception_plans(
        collect_entities(load_rules(ROOT / "rules" / "library", only_active=False)),
        build_drawing_parse_plan(cfg),
    )
    ocr, vl = split_plan_for_backends(plan)
    assert any(e.get("entity_id") == "number_mark" and e.get("parse_kind") != "dimension_marks" for e in ocr) or any(
        e.get("entity_id") == "number_mark" for e in ocr
    )
    # 规则 number_mark 在 OCR；drawing_parse dimension_marks 在 VL
    assert any(e.get("parse_kind") == "dimension_marks" for e in vl)
    assert not any(e.get("parse_kind") == "dimension_marks" for e in ocr)


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


def test_dimension_pass2_prompt_mentions_tilt_and_exclusions():
    ent = {
        "parse_kind": "dimension_marks",
        "fields": [
            {"name": "text", "parse_hint": "原文"},
            {"name": "dim_kind", "parse_hint": "diameter|radius"},
            {"name": "angle", "parse_hint": "朝向角"},
        ],
    }
    assert _is_dimension_marks_entity(ent)
    prompt = _dimension_marks_pass2_prompt(ent)
    assert "dim_kind" in prompt
    assert "angle" in prompt
    assert "竖排" in prompt or "倾斜" in prompt


def test_vlm_deskew_config_defaults():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    block = get_dimension_marks_config(cfg)
    assert block.get("vlm_deskew_reread", True) is True
    plan = build_drawing_parse_plan(cfg)
    dim = next(e for e in plan if e.get("parse_kind") == "dimension_marks")
    assert dim.get("vlm_deskew_reread") is True
    assert dim.get("vlm_orientation_retry") is True
    assert int(dim.get("vlm_orientation_retry_max", 0)) >= 8
    assert [float(a) for a in dim.get("vlm_oblique_angles") or []] == [
        30.0,
        60.0,
        120.0,
        150.0,
        210.0,
        240.0,
        300.0,
        330.0,
    ]


def test_dimension_pass2_angles_vertical_and_square():
    ent = {
        "vlm_deskew_reread": True,
        "vlm_deskew_min_angle": 8,
        "vlm_orientation_retry": True,
        "vlm_orientation_retry_max": 8,
        "vlm_oblique_angles": [30, 60, 120, 150, 210, 240, 300, 330],
    }
    vert = _dimension_pass2_angles([10, 10, 20, 80], ent)
    assert vert[0] == 90.0
    assert -90.0 in vert
    square = _dimension_pass2_angles([10, 10, 40, 40], ent)
    assert square[0] == 0.0
    # 近水平框：配置的 8 个斜向角都应进入重试序列
    for a in (30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0):
        assert a in square
    assert len(square) == 9  # 0 + 8 oblique


def test_dimension_pass2_angles_oblique_order_for_square():
    ent = {
        "vlm_deskew_reread": True,
        "vlm_deskew_min_angle": 8,
        "vlm_orientation_retry": True,
        "vlm_orientation_retry_max": 8,
        "vlm_oblique_angles": [30, 60, 120, 150, 210, 240, 300, 330],
    }
    square = _dimension_pass2_angles([10, 10, 40, 40], ent)
    assert square[1:9] == [30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0]


def test_deskew_crop_rotates_and_upsizes():
    img = Image.new("RGB", (20, 60), (255, 255, 255))
    out = _deskew_crop_for_vlm(img, 90.0, min_side=64)
    assert max(out.size) >= 64


def test_upscale_and_merge_dimension_boxes():
    from pipeline.perceive_qwen_vl import (
        _merge_dimension_boxes,
        _scale_bbox_to_original,
        _upscale_image_to_min_side,
    )

    img = Image.new("RGB", (800, 600), (255, 255, 255))
    up, scale = _upscale_image_to_min_side(img, min_side=1536, max_scale=2.5, max_side=4096)
    assert scale > 1.0
    assert min(up.size) >= 1400
    bb = _scale_bbox_to_original([100, 100, 200, 140], scale)
    assert bb[0] < 100

    merged = _merge_dimension_boxes(
        [
            [{"bbox": [10, 10, 40, 30], "confidence": 0.9, "entity_id": "number_mark"}],
            [{"bbox": [12, 12, 38, 28], "confidence": 0.8, "entity_id": "number_mark"}],
            [{"bbox": [200, 200, 240, 230], "confidence": 0.85, "entity_id": "number_mark"}],
        ]
    )
    assert len(merged) == 2


def test_dimension_read_looks_weak():
    assert _dimension_read_looks_weak({}) is True
    assert _dimension_read_looks_weak({"text": "abc"}) is True
    assert _dimension_read_looks_weak({"text": "12±0.1", "dim_kind": "length", "basic_size": "12"}) is False


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
