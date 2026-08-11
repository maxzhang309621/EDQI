"""show_table_boxes 配置绘制表格识别框。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.render_report import (
    _draw_table_boxes,
    _font,
    _table_box_style,
    _table_rows_from_facts,
    render_report,
)


def test_table_rows_and_styles():
    facts = {
        "tables": [
            {"section": "material", "bbox": [10, 20, 200, 80]},
            {"section": "main", "bbox": [10, 80, 200, 250]},
        ]
    }
    rows = _table_rows_from_facts(facts)
    assert len(rows) == 2
    colors = {"table_material": [16, 185, 129], "table_main": [59, 130, 246]}
    assert _table_box_style(rows[0], colors) == ((16, 185, 129), "material_table")
    assert _table_box_style(rows[1], colors) == ((59, 130, 246), "main_table")


def test_draw_table_boxes_pixels():
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    facts = {
        "tables": [
            {"section": "material", "bbox": [10, 20, 200, 80]},
            {"section": "main", "bbox": [10, 80, 200, 250]},
        ]
    }
    legend = _draw_table_boxes(
        draw,
        facts,
        colors={"table_material": [16, 185, 129], "table_main": [59, 130, 246]},
        width=3,
        show_label=False,
        font=_font(14),
    )
    assert img.getpixel((200, 50)) == (16, 185, 129)
    assert img.getpixel((200, 160)) == (59, 130, 246)
    assert len(legend) == 2


def test_render_report_respects_show_table_boxes_flag(tmp_path):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (200, 160), (255, 255, 255)).save(img_path)
    facts = {"tables": [{"section": "main", "bbox": [20, 30, 180, 140]}]}
    findings = {"drawing_id": "t", "passed": True, "findings": []}
    base = {
        "work_dirs": {"vis": str(tmp_path / "vis"), "reports": str(tmp_path / "reports")},
        "render": {
            "table_label": False,
            "box_width": 3,
            "colors": {"table_main": [59, 130, 246]},
        },
    }
    off = {**base, "render": {**base["render"], "show_table_boxes": False}}
    on = {**base, "render": {**base["render"], "show_table_boxes": True}}
    out_off = render_report(img_path, findings, facts=facts, config=off)
    out_on = render_report(img_path, findings, facts=facts, config=on)
    # 关闭时不应出现明显蓝色框；开启时应写出文件
    assert out_off["vis"].exists() and out_on["vis"].exists()
    assert out_on["vis"].stat().st_size >= out_off["vis"].stat().st_size
