"""部件/视图分区：外扩、扣表、回退网格、部件关联。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.view_regions import (
    build_view_regions,
    expand_view_bbox,
    resolve_dimension_regions,
    resolve_part_region_specs,
    shrink_region_away_from_tables,
)


def test_expand_view_bbox():
    out = expand_view_bbox([100, 100, 200, 200], 1000, 1000, expand_ratio=0.2)
    assert out[0] < 100 and out[1] < 100
    assert out[2] > 200 and out[3] > 200


def test_tighten_bbox_to_ink():
    from PIL import Image, ImageDraw

    from pipeline.view_regions import tighten_bbox_to_ink

    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 120, 120], outline=(0, 0, 0), width=2)
    loose = [40, 40, 160, 160]
    tight = tighten_bbox_to_ink(img, loose, ink_threshold=245, pad=2)
    assert tight[0] >= loose[0] and tight[1] >= loose[1]
    assert tight[2] <= loose[2] and tight[3] <= loose[3]
    assert tight[0] <= 82 and tight[1] <= 82
    assert tight[2] >= 118 and tight[3] >= 118


def test_tighten_thick_outline_ignores_thin_peripheral_lines():
    """粗矩形边框 + 框外细线：粗轮廓收紧应贴粗框，不被细线撑大。"""
    from PIL import Image, ImageDraw

    from pipeline.view_regions import tighten_bbox_to_ink, tighten_bbox_to_thick_outline

    img = Image.new("RGB", (300, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 粗轮廓（约 5px）
    draw.rectangle([100, 100, 200, 200], outline=(0, 0, 0), width=5)
    # 外围细尺寸线（1px），会撑大全墨迹框
    draw.line([(60, 150), (90, 150)], fill=(0, 0, 0), width=1)
    draw.line([(210, 150), (250, 150)], fill=(0, 0, 0), width=1)
    loose = [40, 40, 260, 260]
    thick = tighten_bbox_to_thick_outline(
        img, loose, ink_threshold=245, pad=2, thick_min_width=3
    )
    ink = tighten_bbox_to_ink(img, loose, ink_threshold=245, pad=2)
    # 粗轮廓结果应明显小于含细线的全墨迹框
    thick_w = thick[2] - thick[0]
    ink_w = ink[2] - ink[0]
    assert thick_w < ink_w - 10
    assert thick[0] >= 95 and thick[1] >= 95
    assert thick[2] <= 205 and thick[3] <= 205


def test_expand_ratio_tighter_default_behavior():
    loose = expand_view_bbox([100, 100, 200, 200], 1000, 1000, expand_ratio=0.28)
    tight = expand_view_bbox([100, 100, 200, 200], 1000, 1000, expand_ratio=0.18)
    assert tight[0] > loose[0] and tight[1] > loose[1]
    assert tight[2] < loose[2] and tight[3] < loose[3]


def test_shrink_drops_center_in_table():
    region = [1100, 1100, 1300, 1200]
    tables = [[1000, 1050, 2000, 1400]]
    assert shrink_region_away_from_tables(region, tables) is None


def test_shrink_clips_above_table():
    region = [1200, 900, 1400, 1150]
    tables = [[1100, 1093, 2007, 1170]]
    out = shrink_region_away_from_tables(region, tables)
    assert out is not None
    assert out[3] <= 1093


def test_build_view_regions_and_fallback():
    page_w, page_h = 2048, 1448
    tables = [[1100, 1093, 2007, 1400]]
    views = [[200, 200, 600, 700], [700, 150, 1000, 500]]
    regions = build_view_regions(
        views,
        page_w=page_w,
        page_h=page_h,
        table_bboxes=tables,
        expand_ratio=0.2,
        min_side=32,
        max_area_frac=0.5,
    )
    assert len(regions) >= 1
    for r in regions:
        cx = (r[0] + r[2]) / 2
        cy = (r[1] + r[3]) / 2
        assert not (1100 <= cx <= 2007 and 1093 <= cy <= 1400)

    empty, src = resolve_dimension_regions(
        [],
        page_w=page_w,
        page_h=page_h,
        table_bboxes=tables,
        tile_size=1280,
        fallback_grid=True,
    )
    assert src in {"grid", "full_page"}
    assert len(empty) >= 1


def test_resolve_part_specs_with_labels_and_parent():
    specs, src = resolve_part_region_specs(
        [{"bbox": [100, 100, 400, 400], "label": "front"}],
        page_w=2000,
        page_h=1500,
        table_bboxes=[[1500, 1200, 1900, 1450]],
        expand_ratio=0.15,
        fallback_grid=True,
    )
    assert src == "view_regions"
    assert len(specs) == 1
    assert specs[0]["part_id"] == "component#0"
    assert specs[0]["label"] == "front"
    assert specs[0]["bbox_raw"] == [100, 100, 400, 400]
    assert specs[0]["bbox_expanded"][0] <= 100
    assert len(specs[0]["regions"]) >= 1


def test_empty_views_fallback_grid():
    specs, src = resolve_part_region_specs(
        None,
        page_w=2400,
        page_h=1800,
        fallback_grid=True,
        tile_size=1280,
    )
    assert src == "grid"
    assert specs[0]["part_id"] is None
    assert len(specs[0]["regions"]) >= 2


def test_build_view_regions_splits_by_tile_side():
    # 面积未超 max_area_frac，但边长 > tile_size → 仍应再切
    page_w, page_h = 3000, 3000
    regions = build_view_regions(
        [[100, 100, 2000, 900]],
        page_w=page_w,
        page_h=page_h,
        table_bboxes=None,
        expand_ratio=0.0,
        min_side=32,
        max_area_frac=0.9,
        tile_size=1280,
        tile_overlap=0.2,
    )
    assert len(regions) >= 2


def test_uncovered_grid_regions_skips_covered():
    from pipeline.view_regions import uncovered_grid_regions

    covered = [[0, 0, 2000, 1500]]
    extra = uncovered_grid_regions(
        covered,
        page_w=2000,
        page_h=1500,
        tile_size=1280,
        min_cover_frac=0.55,
    )
    assert extra == []


def test_assign_parent_by_components():
    from pipeline.view_regions import assign_parent_by_components

    comps = [
        {
            "instance_id": "component#0",
            "bbox": [100, 100, 400, 400],
            "bbox_expanded": [50, 50, 450, 450],
        }
    ]
    insts = [
        {"entity_id": "number_mark", "bbox": [200, 200, 220, 230]},
        {"entity_id": "number_mark", "bbox": [900, 900, 920, 930], "parent_id": "component#9"},
    ]
    out = assign_parent_by_components(insts, comps)
    assert out[0]["parent_id"] == "component#0"
    assert out[1]["parent_id"] == "component#9"