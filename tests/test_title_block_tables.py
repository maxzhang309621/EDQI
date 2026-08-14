"""标题栏附属表分离与 bbox_only 契约。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.drawing_parse_plan import _normalize_table_block
from pipeline.title_block_tables import (
    detect_bordered_table_candidates,
    enforce_aux_above_material,
    separate_aux_tables,
    split_bbox_by_horizontal_gaps,
)
from pipeline.perceive_qwen_vl import _is_bbox_only_table, refine_material_main_boxes
from unittest.mock import patch


def test_normalize_bbox_only_aux():
    ent = _normalize_table_block(
        {
            "enabled": True,
            "entity_id": "aux_table",
            "section": "aux",
            "locate_query": "附属表",
            "read_mode": "bbox_only",
            "fields": [],
        }
    )
    assert ent is not None
    assert ent["read_mode"] == "bbox_only"
    assert ent["section"] == "aux"
    assert ent["fields"] == []


def test_normalize_rejects_unknown_read_mode_fallback():
    ent = _normalize_table_block(
        {
            "enabled": True,
            "entity_id": "main_table",
            "section": "main",
            "locate_query": "主表",
            "read_mode": "weird_mode",
            "fields": [{"name": "page", "parse_hint": "Page"}],
        }
    )
    assert ent is not None
    assert ent["read_mode"] == "cell_content"


def _draw_grid_table(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, rows: int = 3):
    draw.rectangle([x1, y1, x2, y2], outline="black", width=3)
    for i in range(1, rows):
        y = y1 + (y2 - y1) * i // rows
        draw.line([(x1, y), (x2, y)], fill="black", width=2)
    mid = (x1 + x2) // 2
    draw.line([(mid, y1), (mid, y2)], fill="black", width=2)


def test_detect_bordered_table_candidates_two_stacks():
    img = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid_table(draw, 50, 40, 350, 140, rows=2)  # aux
    _draw_grid_table(draw, 50, 160, 350, 280, rows=3)  # material-ish
    cands = detect_bordered_table_candidates(img, [20, 20, 380, 300], min_area=400, min_side=20)
    assert len(cands) >= 2


def test_split_bbox_by_horizontal_gaps():
    img = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 80], outline="black", width=2)
    draw.rectangle([20, 160, 180, 260], outline="black", width=2)
    strips = split_bbox_by_horizontal_gaps(img, [10, 10, 190, 270], min_gap=20, min_strip_h=20)
    assert len(strips) >= 2
    assert strips[0][3] <= strips[1][1] + 5


def test_separate_aux_from_tall_material_via_gaps():
    img = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid_table(draw, 40, 30, 360, 120, rows=2)
    _draw_grid_table(draw, 40, 180, 360, 300, rows=3)
    boxes = [
        {"entity_id": "material_table", "bbox": [40, 30, 360, 300]},
        {"entity_id": "main_table", "bbox": [40, 320, 360, 460]},
    ]
    plan = [
        {"entity_id": "aux_table", "section": "aux", "read_mode": "bbox_only"},
        {"entity_id": "material_table", "section": "material"},
        {"entity_id": "main_table", "section": "main"},
    ]
    with patch("pipeline.title_block_tables.run_ocr_boxes", return_value=[]):
        out, notes = separate_aux_tables(
            img, boxes, plan, page_w=400, page_h=500, search_up_pad=20
        )
    aux = [b for b in out if str(b.get("entity_id") or "").startswith("aux_table")]
    mat = next(b for b in out if b["entity_id"] == "material_table")
    assert aux, notes
    assert mat["bbox"][1] >= aux[0]["bbox"][3] - 2 or any("trim_material" in n for n in notes)
    out2, _ = enforce_aux_above_material(out, plan, gap=2)
    mat2 = next(b for b in out2 if b["entity_id"] == "material_table")
    for a in out2:
        if str(a.get("entity_id") or "").startswith("aux_table"):
            assert a["bbox"][3] <= mat2["bbox"][1]


def test_refine_keeps_two_table_without_aux():
    img = Image.new("RGB", (400, 400), "white")
    boxes = [
        {"entity_id": "material_table", "bbox": [50, 80, 350, 160]},
        {"entity_id": "main_table", "bbox": [50, 160, 350, 340]},
    ]
    plan = [
        {"entity_id": "material_table", "section": "material"},
        {"entity_id": "main_table", "section": "main"},
    ]
    layout = [{"text": "Product", "score": 0.99, "bbox": [30, 70, 100, 90]}]
    with patch("pipeline.table_layout_ocr.run_ocr_boxes", return_value=layout):
        with patch("pipeline.title_block_tables.run_ocr_boxes", return_value=layout):
            with patch(
                "pipeline.title_block_tables.detect_bordered_table_candidates",
                return_value=[],
            ):
                with patch(
                    "pipeline.title_block_tables.split_bbox_by_horizontal_gaps",
                    side_effect=lambda *a, **k: [a[1]],
                ):
                    out, notes = refine_material_main_boxes(
                        img, boxes, page_w=400, page_h=400, plan=plan, config={}
                    )
    assert any("product_split:row=" in n or "boundary_source" in n for n in notes)
    aux = [b for b in out if str(b.get("entity_id") or "").startswith("aux_table")]
    assert aux == []
    mat = next(b for b in out if b["entity_id"] == "material_table")
    main = next(b for b in out if b["entity_id"] == "main_table")
    assert mat["bbox"][3] <= main["bbox"][1] + 2


def test_is_bbox_only_table():
    assert _is_bbox_only_table({"read_mode": "bbox_only", "section": "aux"}, None)
    assert _is_bbox_only_table(None, {"entity_id": "aux_table_1", "fields": {}})
    assert not _is_bbox_only_table(
        {"entity_id": "material_table", "read_mode": "above_cells", "section": "material"},
        {"entity_id": "material_table"},
    )
