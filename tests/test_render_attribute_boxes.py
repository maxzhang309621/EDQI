"""show_attribute_boxes 可视化已识别尺寸属性。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.render_report import (
    _attribute_rows_from_facts,
    _draw_attribute_boxes,
    _font,
    _format_attribute_label,
    render_report,
)


def test_format_attribute_label():
    label = _format_attribute_label(
        {
            "text": "12±0.1",
            "dim_kind": "diameter",
            "basic_size": "12",
            "tolerance": "0.1",
            "angle": 90.0,
        }
    )
    assert "12±0.1" in label
    assert "diameter" in label
    assert "s=12" in label
    assert "±0.1" in label
    assert "∠90" in label


def test_attribute_rows_from_annotations():
    facts = {
        "annotations": [
            {"bbox": [10, 10, 40, 30], "text": "R5", "dim_kind": "radius", "angle": 0},
            {"bbox": None, "text": "skip"},
        ]
    }
    rows = _attribute_rows_from_facts(facts)
    assert len(rows) == 1
    assert rows[0]["dim_kind"] == "radius"


def test_draw_attribute_boxes_and_flag(tmp_path):
    img = Image.new("RGB", (200, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    facts = {
        "annotations": [
            {
                "bbox": [20, 40, 80, 70],
                "text": "10",
                "dim_kind": "length",
                "basic_size": "10",
                "angle": 0,
            }
        ]
    }
    legend = _draw_attribute_boxes(
        draw,
        facts,
        colors={"attribute": [14, 165, 233], "attribute_length": [59, 130, 246]},
        width=2,
        show_label=False,
        font=_font(12),
    )
    assert img.getpixel((80, 55)) == (59, 130, 246)
    assert legend and "dimension" in legend[0][1][0]

    img_path = tmp_path / "page.png"
    Image.new("RGB", (200, 160), (255, 255, 255)).save(img_path)
    findings = {"drawing_id": "a", "passed": True, "findings": []}
    base = {
        "work_dirs": {"vis": str(tmp_path / "vis"), "reports": str(tmp_path / "reports")},
        "render": {
            "show_table_boxes": False,
            "attribute_label": False,
            "attribute_box_width": 2,
            "colors": {"attribute": [14, 165, 233]},
        },
    }
    off = {**base, "render": {**base["render"], "show_attribute_boxes": False}}
    on = {**base, "render": {**base["render"], "show_attribute_boxes": True}}
    out_off = render_report(img_path, findings, facts=facts, config=off)
    out_on = render_report(img_path, findings, facts=facts, config=on)
    assert out_off["vis"].exists() and out_on["vis"].exists()
    assert out_on["vis"].stat().st_size >= out_off["vis"].stat().st_size
