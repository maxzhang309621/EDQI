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


def test_merge_symbol_number_boxes():
    from pipeline.dimension_parse import parse_dimension_text
    from pipeline.perceive_number_overlap import _merge_symbol_number_boxes, _looks_like_number_mark

    assert _looks_like_number_mark("\u00d8")  # Ø alone
    assert _looks_like_number_mark("\u00b0")  # ° alone
    insts = [
        {
            "entity_id": "number_mark",
            "bbox": [10, 10, 28, 30],
            "raw_text": "\u00d8",
            "fields": {"text": "\u00d8"},
            "confidence": 0.7,
        },
        {
            "entity_id": "number_mark",
            "bbox": [30, 10, 70, 30],
            "raw_text": "12.5",
            "fields": {"text": "12.5"},
            "confidence": 0.9,
        },
        {
            "entity_id": "number_mark",
            "bbox": [100, 10, 140, 30],
            "raw_text": "45",
            "fields": {"text": "45"},
            "confidence": 0.85,
        },
        {
            "entity_id": "number_mark",
            "bbox": [142, 10, 158, 30],
            "raw_text": "\u00b0",
            "fields": {"text": "\u00b0"},
            "confidence": 0.6,
        },
    ]
    out = _merge_symbol_number_boxes(insts)
    assert len(out) == 2
    assert all((i.get("fields") or {}).get("symbol_merged") for i in out)
    kinds = {parse_dimension_text(str(i.get("raw_text")))["dim_kind"] for i in out}
    assert "diameter" in kinds
    assert "angle" in kinds


def test_ocr_owned_vs_vlm_symbol_roles():
    from pipeline.dimension_parse import (
        is_bare_numeric_dimension_candidate,
        is_ocr_owned_dim_kind,
        is_vlm_owned_dim_kind,
        prescreen_ocr_dimension_candidates,
        sanitize_number_mark_instances,
    )

    assert is_ocr_owned_dim_kind("radius")
    assert is_ocr_owned_dim_kind("length")
    assert is_vlm_owned_dim_kind("diameter")
    assert is_vlm_owned_dim_kind("angle")
    assert is_bare_numeric_dimension_candidate("12.5")
    assert is_bare_numeric_dimension_candidate("12±0.1")
    assert not is_bare_numeric_dimension_candidate("R5")
    assert not is_bare_numeric_dimension_candidate("Ø10")

    instances = [
        {
            "entity_id": "number_mark",
            "bbox": [10, 10, 40, 28],
            "raw_text": "R5",
            "fields": {"text": "R5"},
            "confidence": 0.9,
        },
        {
            "entity_id": "number_mark",
            "bbox": [50, 10, 90, 28],
            "raw_text": "12.5",
            "fields": {"text": "12.5"},
            "confidence": 0.9,
        },
        {
            "entity_id": "number_mark",
            "bbox": [100, 10, 140, 28],
            "raw_text": "Ø10",
            "fields": {"text": "Ø10"},
            "confidence": 0.9,
        },
    ]
    cands, _, _ = prescreen_ocr_dimension_candidates(instances, page_w=2048, page_h=1448)
    roles = {c["raw_text"]: (c["fields"].get("ocr_role"), c["fields"].get("dim_kind")) for c in cands}
    assert roles["R5"][0] == "ocr_owned"
    assert roles["12.5"][0] == "ocr_owned"  # 长度归 OCR
    assert roles["Ø10"][0] == "vlm_symbol"  # 直径必须 VLM 确认

    # 直径未经 VLM 确认 → sanitize 丢弃
    fake = [
        {
            "entity_id": "number_mark",
            "bbox": [100, 10, 140, 28],
            "raw_text": "Ø10",
            "fields": {"text": "Ø10", "dim_kind": "diameter", "basic_size": "10"},
        },
        {
            "entity_id": "number_mark",
            "bbox": [10, 10, 40, 28],
            "raw_text": "R5",
            "fields": {"text": "R5", "dim_kind": "radius", "basic_size": "5", "vlm_filtered": True},
        },
    ]
    out, dropped = sanitize_number_mark_instances(fake, page_w=2048, page_h=1448)
    texts = [i.get("raw_text") for i in out if i.get("entity_id") == "number_mark"]
    assert "R5" in texts
    assert "Ø10" not in texts
    assert dropped >= 1


def test_vertical_bbox_geometry_relaxed():
    from pipeline.dimension_parse import bbox_geometry_ok

    # 竖排窄高框应通过
    assert bbox_geometry_ok([100, 100, 130, 220], page_w=2048, page_h=1448)
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
            {
                "entity_id": "number_mark",
                "bbox": [1024, 0, 2048, 1169],
                "raw_text": "图中无尺寸标注",
                "fields": {"text": "图中无尺寸标注"},
                "confidence": 0.5,
            },
            {
                "entity_id": "number_mark",
                "bbox": [1600, 1130, 1650, 1155],
                "raw_text": "7295.9",
                "fields": {"text": "7295.9"},
                "confidence": 0.9,
            },
        ]
    )
    cands, pairs, dropped = prescreen_ocr_dimension_candidates(
        instances,
        page_w=2048,
        page_h=1448,
        exclude_bboxes=[[1108, 1090, 2009, 1430]],
    )
    assert len(pairs) == 1
    assert pairs[0].get("keep_pair") is True
    assert len(cands) == 1
    assert cands[0]["fields"].get("dim_kind") == "length"
    assert dropped >= 3


def test_vlm_filter_rejects_null_without_ocr_fallback(tmp_path: Path):
    img = Image.new("RGB", (200, 80), (255, 255, 255))
    path = tmp_path / "p2.png"
    img.save(path)
    candidates = [
        {
            "entity_id": "number_mark",
            "bbox": [10, 10, 50, 30],
            "raw_text": "12±0.1",
            "fields": {"text": "12±0.1", "dim_kind": "length", "basic_size": "12"},
            "confidence": 0.9,
        }
    ]
    ent = {
        "parse_kind": "dimension_marks",
        "strict_fields_only": True,
        "require_dim_kind": True,
        "require_basic_size": True,
        "fields": [{"name": "text"}, {"name": "dim_kind"}, {"name": "basic_size"}],
    }

    def gen_null(crop, prompt, max_tokens):
        return json.dumps(
            {
                "fields": {
                    "text": None,
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
        path, candidates, ent, meta={"width": 200, "height": 80}, notes=notes, generate_fn=gen_null
    )
    assert kept == []
    assert any("vlm_dim_rejected=" in n for n in notes)


def test_sanitize_drops_huge_and_table_marks():
    from pipeline.dimension_parse import sanitize_number_mark_instances

    instances = [
        {
            "entity_id": "number_mark",
            "bbox": [1024, 0, 2048, 1169],
            "fields": {"text": "x", "dim_kind": "length", "basic_size": "1"},
            "raw_text": "x",
        },
        {
            "entity_id": "number_mark",
            "bbox": [1200, 1120, 1250, 1145],
            "fields": {"text": "65.1", "dim_kind": "length", "basic_size": "65.1"},
            "raw_text": "65.1",
        },
        {
            "entity_id": "number_mark",
            "bbox": [100, 100, 160, 130],
            "fields": {"text": "11±0.1", "dim_kind": "length", "basic_size": "11", "tolerance": "0.1"},
            "raw_text": "11±0.1",
        },
        {"entity_id": "main_table", "bbox": [1100, 1100, 2000, 1400], "fields": {}},
    ]
    out, dropped = sanitize_number_mark_instances(
        instances,
        page_w=2048,
        page_h=1448,
        allowed_field_names=["text", "dim_kind", "basic_size", "tolerance", "has_tolerance", "angle"],
        exclude_bboxes=[[1100, 1090, 2000, 1400]],
    )
    assert dropped >= 2
    marks = [i for i in out if i.get("entity_id") == "number_mark"]
    assert len(marks) == 1
    assert marks[0]["raw_text"] == "11±0.1"
    assert any(i.get("entity_id") == "main_table" for i in out)


def test_vlm_filter_accepts_and_rejects(tmp_path: Path):
    img = Image.new("RGB", (800, 400), (255, 255, 255))
    path = tmp_path / "p.png"
    img.save(path)

    candidates = [
        {
            "entity_id": "number_mark",
            "bbox": [20, 20, 70, 45],
            "raw_text": "R2",
            "fields": {"text": "R2", "dim_kind": "radius", "basic_size": "2"},
            "confidence": 0.9,
        },
        {
            "entity_id": "number_mark",
            "bbox": [20, 80, 90, 105],
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
        "max_bbox_width_ratio": 0.22,
        "max_bbox_height_ratio": 0.12,
        "max_bbox_area_ratio": 0.035,
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
        meta={"width": 800, "height": 400},
        notes=notes,
        generate_fn=gen,
    )
    assert len(kept) == 1
    assert kept[0]["fields"]["dim_kind"] == "radius"
    assert kept[0]["fields"].get("vlm_filtered") is True
    assert any(n.startswith("vlm_dim_filter=") for n in notes)
