"""模型输出 JSON 容错解析（截断 / 未闭合 fence）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.perceive_common import extract_json_payload, extract_json_payload_safe


def test_unclosed_fence_partial_array():
    text = """```json
[
    {"bbox_2d": [545, 764, 974, 821], "label": "material_table", "entity_id": "material_table", "fields": {}},
    {"bbox_2d": [545, 820, 974, 980], "label": "main_table", "entity_id": "main_table", "fields": {
"""
    out = extract_json_payload(text)
    assert isinstance(out, list)
    assert len(out) >= 1
    assert out[0]["entity_id"] == "material_table"


def test_closed_fence_ok():
    text = '```json\n[{"entity_id":"main_table","bbox_2d":[1,2,3,4],"fields":{}}]\n```'
    out = extract_json_payload(text)
    assert out[0]["entity_id"] == "main_table"


def test_safe_empty_default():
    assert extract_json_payload_safe("not json", default=[]) == []
