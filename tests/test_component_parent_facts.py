"""部件与属性 parent_id 关联进入 facts。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.build_facts import build_facts


def test_build_facts_component_and_parent_id():
    payload = {
        "backend": "qwen_vl",
        "instances": [
            {
                "entity_id": "number_mark",
                "instance_id": "number_mark#0",
                "bbox": [10, 10, 40, 30],
                "parent_id": "component#0",
                "fields": {"text": "12", "dim_kind": "length"},
                "confidence": 0.9,
            },
            {
                "entity_id": "component",
                "instance_id": "component#0",
                "label": "front",
                "bbox": [5, 5, 200, 180],
                "bbox_expanded": [0, 0, 220, 200],
                "fields": {"label": "front"},
                "confidence": 0.85,
            },
        ],
    }
    facts = build_facts(payload, {"width": 400, "height": 300, "drawing_id": "t"}, validate=False)
    assert len(facts["components"]) == 1
    assert facts["components"][0]["label"] == "front"
    assert facts["components"][0]["bbox_expanded"] == [0, 0, 220, 200]
    assert facts["annotations"][0]["parent_id"] == "component#0"
    assert facts["annotations"][0]["text"] == "12"
