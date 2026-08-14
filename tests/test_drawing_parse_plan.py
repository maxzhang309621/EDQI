"""drawing_parse 配置 → 感知计划。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import load_config
from pipeline.build_facts import build_facts
from pipeline.drawing_parse_plan import (
    build_drawing_parse_plan,
    merge_perception_plans,
    split_plan_for_backends,
)
from pipeline.perceive_common import mock_perceive
from engines import collect_entities, load_rules

MATERIAL_FIELDS = {
    "article_no",
    "material_designation",
    "additional_material_information",
    "volume",
    "surface",
    "weight",
}
MAIN_FIELDS = {
    "component_name",
    "restricted_substances",
    "tolerance_1",
    "tolerance_2",
    "surface_treatment",
    "surface_finish_1",
    "surface_finish_2",
    "notes",
    "created_by",
    "approved_by",
    "date",
    "revision",
    "revision_state",
    "document_number",
    "document_subclass",
    "page",
    "scale",
}


def test_drawing_parse_split_material_and_main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = build_drawing_parse_plan(cfg, enabled=True)
    by_id = {e["entity_id"]: e for e in plan}
    assert "material_table" in by_id and "main_table" in by_id

    mat = by_id["material_table"]
    assert mat["read_mode"] == "above_cells"
    assert mat["value_as_array"] is True
    assert {f["name"] for f in mat["fields"]} == MATERIAL_FIELDS

    main = by_id["main_table"]
    assert main["read_mode"] == "cell_content"
    assert main["value_as_array"] is False
    assert {f["name"] for f in main["fields"]} == MAIN_FIELDS
    cn = next(f for f in main["fields"] if f["name"] == "component_name")
    assert cn.get("locate_by") == "size_position"


def test_dimension_marks_in_drawing_parse_plan():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = build_drawing_parse_plan(cfg, enabled=True)
    dims = [e for e in plan if e.get("parse_kind") == "dimension_marks"]
    assert len(dims) == 1
    ent = dims[0]
    assert ent["entity_id"] == "number_mark"
    assert ent.get("detect_overlap") is False
    assert ent.get("backend") == "vlm"
    names = {f["name"] for f in ent["fields"]}
    assert names >= {"text", "angle", "dim_kind", "basic_size", "tolerance", "has_tolerance"}
    ocr, vl = split_plan_for_backends(plan)
    # backend=vlm：尺寸属性进 VL，不再进重叠 OCR
    assert not any(e.get("parse_kind") == "dimension_marks" for e in ocr)
    assert {e["entity_id"] for e in vl} == {"aux_table", "material_table", "main_table", "number_mark"}
    assert any(e.get("parse_kind") == "dimension_marks" for e in vl)


def test_tables_only_default_false():
    from pipeline.drawing_parse_plan import is_tables_only

    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert is_tables_only(cfg) is False
    # 默认仅 drawing_parse（重叠规则 draft）；VL 非空，OCR 可为空
    plan = merge_perception_plans(
        collect_entities(load_rules(ROOT / "rules" / "library", only_active=True)),
        build_drawing_parse_plan(cfg),
    )
    ocr, vl = split_plan_for_backends(plan)
    assert vl
    assert any(e.get("parse_kind") == "dimension_marks" for e in vl)
    assert not ocr

    # 含 draft 规则实体时仍可拆出 OCR + VL
    plan2 = merge_perception_plans(
        collect_entities(load_rules(ROOT / "rules" / "library", only_active=False)),
        build_drawing_parse_plan(cfg),
    )
    ocr2, vl2 = split_plan_for_backends(plan2)
    assert ocr2 and vl2


def test_tables_only_keeps_dimension_marks_from_drawing_parse():
    """tables_only 跳过规则库重叠实体合并，但仍可包含 objects 中的 dimension_marks（进 VL）。"""
    from pipeline.drawing_parse_plan import is_tables_only

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = {**cfg, "drawing_parse": {**(cfg.get("drawing_parse") or {}), "tables_only": True}}
    assert is_tables_only(cfg) is True
    plan = build_drawing_parse_plan(cfg)
    ocr, vl = split_plan_for_backends(plan)
    assert not ocr
    assert {e["entity_id"] for e in vl} == {"aux_table", "material_table", "main_table", "number_mark"}
    assert any(e.get("parse_kind") == "dimension_marks" for e in vl)


def test_dimension_marks_backend_ocr_routes_to_ocr():
    plan = [
        {
            "entity_id": "number_mark",
            "parse_kind": "dimension_marks",
            "backend": "ocr",
            "locate_query": "尺寸",
            "fields": [],
        },
        {"entity_id": "main_table", "parse_kind": "table", "fields": []},
    ]
    ocr, vl = split_plan_for_backends(plan)
    assert len(ocr) == 1 and ocr[0].get("backend") == "ocr"
    assert {e["entity_id"] for e in vl} == {"main_table"}


def test_mock_split_tables_to_facts():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    plan = [e for e in build_drawing_parse_plan(cfg) if e.get("parse_kind") == "table"]
    payload = mock_perceive(plan, {"width": 800, "height": 600, "drawing_id": "parse_demo"})
    facts = build_facts(payload, {"width": 800, "height": 600, "drawing_id": "parse_demo"})
    assert len(facts["tables"]) == 3
    by_section = {t.get("section"): t for t in facts["tables"]}
    assert "aux" in by_section and "material" in by_section and "main" in by_section
    assert by_section["aux"].get("read_mode") == "bbox_only"
    assert by_section["aux"].get("bbox")
    mat = by_section["material"]
    assert isinstance(mat.get("volume"), list)
    assert mat["volume"] == ["7295.9"]
    assert mat.get("pairs") == []
    main = by_section["main"]
    assert main.get("document_number") == "25001745002A"
    assert main.get("component_name") == "Contact lever (short)"
    assert main.get("page") == "1/1"
    assert main.get("pairs") == []
