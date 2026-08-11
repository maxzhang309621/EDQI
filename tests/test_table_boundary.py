"""物料表/主表硬分界。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.perceive_qwen_vl import enforce_material_main_boundary


def test_raise_main_below_material():
    boxes = [
        {"entity_id": "material_table", "bbox": [100, 100, 500, 200]},
        {"entity_id": "main_table", "bbox": [100, 100, 500, 400]},  # 同顶，需上抬
    ]
    plan = [
        {"entity_id": "material_table", "section": "material"},
        {"entity_id": "main_table", "section": "main"},
    ]
    out, notes = enforce_material_main_boundary(
        boxes, page_w=800, page_h=600, plan=plan, gap=2, min_main_height=80
    )
    mat = next(b for b in out if b["entity_id"] == "material_table")
    main = next(b for b in out if b["entity_id"] == "main_table")
    assert main["bbox"][1] >= mat["bbox"][3] + 2
    assert main["bbox"][3] >= 400  # 下沿不收缩
    assert any("raise_main" in n for n in notes)
    assert main.get("_y1_floor") == mat["bbox"][3] + 2


def test_trim_thick_material_keeps_main_height():
    boxes = [
        {"entity_id": "material_table", "bbox": [100, 100, 500, 350]},  # 过厚
        {"entity_id": "main_table", "bbox": [100, 200, 500, 420]},
    ]
    plan = [
        {"entity_id": "material_table", "section": "material"},
        {"entity_id": "main_table", "section": "main"},
    ]
    out, notes = enforce_material_main_boundary(
        boxes, page_w=800, page_h=600, plan=plan, gap=2, min_main_height=100
    )
    mat = next(b for b in out if b["entity_id"] == "material_table")
    main = next(b for b in out if b["entity_id"] == "main_table")
    assert main["bbox"][1] >= mat["bbox"][3] + 2
    assert main["bbox"][3] - main["bbox"][1] >= 100
    assert any("trim_material" in n or "raise_main" in n for n in notes)
