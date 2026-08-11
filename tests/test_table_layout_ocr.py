"""Product 行分界 + 物料表标签上方 OCR。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.table_layout_ocr import (
    apply_product_split_to_boxes,
    extract_fields_above_labels,
    split_title_block_by_product,
)


def _items_from_layout(layout: list[tuple[str, list[int]]]):
    return [{"text": t, "score": 0.99, "bbox": b} for t, b in layout]


def test_split_by_product_row():
    img = Image.new("RGB", (400, 300), "white")
    seeds = [[50, 20, 350, 280]]
    # crop-local coords（pad=12 → origin 38,8）
    layout = [
        ("Article no.", [20, 40, 80, 55]),
        ("21940", [100, 20, 160, 35]),
        ("Product", [20, 120, 90, 140]),
        ("Document number", [20, 160, 140, 180]),
    ]
    with patch("pipeline.table_layout_ocr.run_ocr_boxes", return_value=_items_from_layout(layout)):
        mat, main, notes = split_title_block_by_product(img, seeds, pad=12)
    assert mat is not None and main is not None
    assert any(n.startswith("product_split:row=") for n in notes)
    assert mat[3] <= main[1]
    assert any("Product" in n for n in notes)


def test_extract_above_cells_null_and_value():
    img = Image.new("RGB", (500, 200), "white")
    bbox = [0, 0, 500, 200]
    # 标签同行；article 上方空；AMI 上方 21940；Volume 上方 12.3（不同列）
    layout = [
        ("Article no.", [10, 140, 90, 160]),
        ("Additional material information", [200, 140, 420, 160]),
        ("Volume", [100, 140, 160, 160]),
        ("21940", [220, 40, 280, 60]),
        ("12.3", [110, 80, 150, 100]),
    ]
    fields = [
        {"name": "article_no", "parse_hint": "Article no."},
        {
            "name": "additional_material_information",
            "parse_hint": "Additional material information",
        },
        {"name": "volume", "parse_hint": "Volume"},
    ]
    with patch("pipeline.table_layout_ocr.run_ocr_boxes", return_value=_items_from_layout(layout)):
        out, _raw, notes = extract_fields_above_labels(img, bbox, fields)
    assert out["article_no"] is None
    assert out["additional_material_information"] == ["21940"]
    assert out["volume"] == ["12.3"]
    assert any("empty_above:article_no" in n for n in notes)


def test_apply_product_split_rewrites_boxes():
    img = Image.new("RGB", (400, 300), "white")
    boxes = [
        {"entity_id": "material_table", "bbox": [40, 10, 360, 200]},
        {"entity_id": "main_table", "bbox": [40, 10, 360, 280]},
    ]
    plan = [
        {"entity_id": "material_table", "section": "material"},
        {"entity_id": "main_table", "section": "main"},
    ]
    layout = [("Product", [30, 100, 100, 120])]
    with patch("pipeline.table_layout_ocr.run_ocr_boxes", return_value=_items_from_layout(layout)):
        out, notes = apply_product_split_to_boxes(img, boxes, plan)
    mat = next(b for b in out if b["entity_id"] == "material_table")
    main = next(b for b in out if b["entity_id"] == "main_table")
    assert mat["bbox"][3] <= main["bbox"][1]
    assert main.get("_y1_floor") == main["bbox"][1]
    assert any("product_split:row=" in n for n in notes)
