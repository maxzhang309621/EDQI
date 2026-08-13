"""规则引擎单测（不依赖 GPU）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.rule_engine import evaluate, eval_logic, get_by_path
from engines import validate_rule, collect_entities, load_rules


def test_get_by_path():
    facts = {
        "title_block": {"part_no": "A-1", "bbox": [1, 2, 3, 4]},
        "components": [{"id": "R1", "value": "10k"}, {"id": "R2", "value": "20k"}],
    }
    assert get_by_path(facts, "title_block.part_no") == "A-1"
    assert get_by_path(facts, "components.value") == ["10k", "20k"]
    assert get_by_path(facts, "components[0].id") == "R1"


def test_not_empty_pass_fail():
    facts = {"title_block": {"part_no": "A-1001"}}
    ok, actual, _ = eval_logic({"op": "not_empty", "path": "title_block.part_no"}, facts)
    assert ok and actual == "A-1001"
    ok2, actual2, _ = eval_logic({"op": "not_empty", "path": "title_block.material"}, facts)
    assert not ok2 and actual2 is None


def test_evaluate_findings():
    facts = {
        "drawing_id": "demo",
        "meta": {"width": 100, "height": 80},
        "title_block": {"part_no": None, "bbox": [10, 10, 50, 40]},
        "components": [],
    }
    rules = [
        {
            "rule_id": "TB_PART_NO_REQUIRED",
            "severity": "error",
            "status": "active",
            "logic": {"op": "not_empty", "path": "title_block.part_no"},
            "message": "标题栏缺少图号",
        }
    ]
    result = evaluate(facts, rules, rule_set_id="t")
    assert result["passed"] is False
    assert result["findings"][0]["rule_id"] == "TB_PART_NO_REQUIRED"
    assert result["findings"][0]["evidence_bboxes"] == [[10, 10, 50, 40]]


def test_and_or_ops():
    facts = {"title_block": {"part_no": "A", "material": ""}}
    logic = {
        "op": "and",
        "args": [
            {"op": "not_empty", "path": "title_block.part_no"},
            {"op": "not_empty", "path": "title_block.material"},
        ],
    }
    ok, _, _ = eval_logic(logic, facts)
    assert ok is False


def test_validate_and_load_library():
    rules = load_rules(ROOT / "rules" / "library", only_active=True)
    # 调试阶段 NUM_TEXT_NO_OVERLAP 置为 draft，active 可为空
    for r in rules:
        errs = validate_rule(r)
        assert errs == [], errs
    all_rules = load_rules(ROOT / "rules" / "library", only_active=False)
    assert any(r.get("rule_id") == "NUM_TEXT_NO_OVERLAP" for r in all_rules)
    draft = next(r for r in all_rules if r.get("rule_id") == "NUM_TEXT_NO_OVERLAP")
    assert draft.get("status") == "draft"
    plan = collect_entities(all_rules)
    assert any(e["entity_id"] == "number_mark" for e in plan)
    assert not any(e["entity_id"] == "title_block" for e in plan)


def test_unknown_op_raises():
    import pytest
    from engines.rule_engine import RuleEngineError

    with pytest.raises(RuleEngineError):
        eval_logic({"op": "foo", "path": "a"}, {})
