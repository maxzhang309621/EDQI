"""部件/视图框可视化（原始框 + 扩展框）。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.render_report import _draw_view_region_boxes, _font, render_report


def test_draw_view_region_boxes(tmp_path):
    img = Image.new("RGB", (240, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    facts = {
        "meta": {"width": 240, "height": 200},
        "components": [
            {
                "instance_id": "component#0",
                "label": "front",
                "bbox": [40, 40, 100, 100],
                "bbox_expanded": [20, 20, 120, 120],
            }
        ],
    }
    legend = _draw_view_region_boxes(
        draw,
        facts,
        colors={"view_raw": [234, 179, 8], "view_expanded": [239, 68, 68]},
        width_raw=3,
        width_expanded=2,
        show_label=False,
        font=_font(12),
    )
    assert img.getpixel((20, 60)) == (239, 68, 68)
    assert img.getpixel((40, 70)) == (234, 179, 8)
    assert len(legend) == 2

    img_path = tmp_path / "page.png"
    Image.new("RGB", (240, 200), (255, 255, 255)).save(img_path)
    findings = {"drawing_id": "v", "passed": True, "findings": []}
    cfg = {
        "work_dirs": {"vis": str(tmp_path / "vis"), "reports": str(tmp_path / "reports")},
        "render": {
            "show_view_regions": True,
            "show_table_boxes": False,
            "show_attribute_boxes": False,
            "view_region_label": False,
            "colors": {"view_raw": [234, 179, 8], "view_expanded": [239, 68, 68]},
        },
    }
    out = render_report(img_path, findings, facts=facts, config=cfg)
    assert out["vis"].exists()
