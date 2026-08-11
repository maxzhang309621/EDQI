"""OCR 定位+规则初筛 → VLM crop 精筛。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import load_config
from pipeline.dimension_parse import apply_dimension_parse_to_instances, prescreen_ocr_dimension_candidates
from pipeline.drawing_parse_plan import (
    build_drawing_parse_plan,
    dimension_marks_backend,
    is_dimension_marks_ocr_locate_vlm,
    is_dimension_marks_vlm,
    merge_perception_plans,
    split_plan_for_backends,
)
from engines import collect_entities, load_rules
from pipeline.perceive_qwen_vl import filter_ocr_dimension_candidates_with_vlm


def test_default_backend_is_ocr_locate_vlm_filter():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert dimension_marks_backend(cfg) == "ocr_locate_vlm_filter"
    assert is_dimension_marks_ocr_locate_vlm(cfg) is True
    assert is_dimension_marks_vlm(cfg) is False


def test_hybrid_dims_route_to_ocr_not_vl_locate():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = merge_perception_plans(
        collect_entities(load_rules(ROOT / "rules" / "library", only_active=True)),
        build_drawing_parse_plan(cfg),
    )
    ocr, vl = split_plan_for_backends(plan)
    assert any(e.get("parse_kind") == "dimension_marks" for e in ocr)
    assert not any(e.get("parse_kind") == "dimension_marks" for e in vl)
    assert {e["entity_id"] for e in vl} >= {"material_table", "main_table"}


def test_prescreen_keeps_pairs_and_valid_dims():
    instances = apply_dimension_parse_to_instances(
        [
            {
                "entity_id": "number_mark",
                "bbox": [10, 10, 40, 28],
                "raw_text": "12±0.1",
                "fields": {"text": "12±0.1"},
                "confidence": 0.9,
            },
            {
                "entity_id": "number_mark",
                "bbox": [10, 40, 80, 58],
                "raw_text": "Siemens 2025",
                "fields": {"text": "Siemens 2025"},
                "confidence": 0.9,
            },
            {
                "entity_id": "number_mark",
                "bbox": [100, 100, 130, 120],
                "raw_text": "2±0.2",
                "fields": {"text": "2±0.2"},
                "keep_pair": True,
                "confidence": 0.95,
            },
        ]
    )
    cands, pairs, dropped = prescreen_ocr_dimension_candidates(
        instances, page_w=2048, page_h=1448
    )
    assert len(pairs) == 1
    assert pairs[0].get("keep_pair") is True
    assert len(cands) == 1
    assert cands[0]["fields"].get("dim_kind") == "length"
    assert dropped >= 1


def test_vlm_filter_accepts_and_rejects(tmp_path: Path):
    img = Image.new("RGB", (200, 120), (255, 255, 255))
    path = tmp_path / "p.png"
    img.save(path)

    candidates = [
        {
            "entity_id": "number_mark",
            "bbox": [10, 10, 60, 30],
            "raw_text": "R2",
            "fields": {"text": "R2", "dim_kind": "radius", "basic_size": "2"},
            "confidence": 0.9,
        },
        {
            "entity_id": "number_mark",
            "bbox": [10, 50, 90, 70],
            "raw_text": "noise",
            "fields": {"text": "noise", "dim_kind": "length", "basic_size": "1"},
            "confidence": 0.8,
        },
    ]
    ent = {
        "parse_kind": "dimension_marks",
        "entity_id": "number_mark",
        "strict_fields_only": True,
        "require_dim_kind": True,
        "require_basic_size": True,
        "fields": [
            {"name": "text"},
            {"name": "dim_kind"},
            {"name": "basic_size"},
            {"name": "tolerance"},
            {"name": "has_tolerance"},
            {"name": "angle"},
        ],
    }

    calls = {"n": 0}

    def gen(crop, prompt, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "fields": {
                        "text": "R2",
                        "dim_kind": "radius",
                        "basic_size": "2",
                        "tolerance": None,
                        "has_tolerance": False,
                        "angle": 0,
                    },
                    "raw_text": "R2",
                }
            )
        return json.dumps(
            {
                "fields": {
                    "text": "",
                    "dim_kind": None,
                    "basic_size": None,
                    "tolerance": None,
                    "has_tolerance": False,
                    "angle": None,
                },
                "raw_text": "",
            }
        )

    notes: list[str] = []
    kept = filter_ocr_dimension_candidates_with_vlm(
        path,
        candidates,
        ent,
        meta={"width": 200, "height": 120},
        notes=notes,
        generate_fn=gen,
    )
    assert len(kept) == 1
    assert kept[0]["fields"]["dim_kind"] == "radius"
    assert kept[0]["fields"].get("vlm_filtered") is True
    assert any(n.startswith("vlm_dim_filter=") for n in notes)
