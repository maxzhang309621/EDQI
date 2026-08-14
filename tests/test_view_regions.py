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


def test_extract_thick_strokes_suppresses_thin_lines():
    from PIL import Image, ImageDraw

    from pipeline.view_regions import extract_thick_strokes

    img = Image.new("RGB", (300, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 220, 220], outline=(0, 0, 0), width=5)
    draw.line([(20, 150), (70, 150)], fill=(0, 0, 0), width=1)
    draw.line([(230, 150), (280, 150)], fill=(0, 0, 0), width=1)
    thick = extract_thick_strokes(img, ink_threshold=245, thick_min_width=3)
    assert thick[150, 40] == 0  # 左侧细线应被抑制
    assert thick[150, 80] > 0 or thick[80, 150] > 0  # 粗边保留


def test_propose_two_thick_rects_exact_boxes():
    """两分离粗矩形 → 恰好 2 个贴边框。"""
    from PIL import Image, ImageDraw

    from pipeline.view_regions import locate_views_from_thick_boundaries

    img = Image.new("RGB", (600, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 40, 250, 200], outline=(0, 0, 0), width=5)
    draw.rectangle([320, 260, 520, 420], outline=(0, 0, 0), width=5)
    draw.line([(10, 10), (100, 10)], fill=(0, 0, 0), width=1)

    views, notes = locate_views_from_thick_boundaries(
        img,
        thick_min_width=3,
        pad=2,
        min_side=40,
        max_area_frac=0.5,
        min_side_evidence=3,
        min_ink_density=0.001,
        box_nms_iou=0.45,
    )
    assert len(views) == 2, notes
    boxes = sorted(views, key=lambda v: v["bbox"][0])
    a, b = boxes[0]["bbox"], boxes[1]["bbox"]
    assert abs(a[0] - 50) <= 6 and abs(a[1] - 40) <= 6
    assert abs(a[2] - 250) <= 6 and abs(a[3] - 200) <= 6
    assert abs(b[0] - 320) <= 6 and abs(b[1] - 260) <= 6
    assert abs(b[2] - 520) <= 6 and abs(b[3] - 420) <= 6


def test_filter_ghost_rejects_empty_and_nms_and_table():
    import numpy as np

    from pipeline.view_regions import filter_ghost_boxes

    h, w = 400, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:105, 100:250] = 255
    mask[245:250, 100:250] = 255
    mask[100:250, 100:105] = 255
    mask[100:250, 245:250] = 255
    good = {"bbox": [100, 100, 250, 250], "score": 0.1}
    ghost = {"bbox": [10, 10, 80, 80], "score": 0.9}
    dup = {"bbox": [102, 102, 248, 248], "score": 0.05}
    table_hit = {"bbox": [300, 300, 380, 380], "score": 0.2}
    mask[300:305, 300:380] = 255
    mask[375:380, 300:380] = 255
    mask[300:380, 300:305] = 255
    mask[300:380, 375:380] = 255

    out = filter_ghost_boxes(
        [ghost, good, dup, table_hit],
        mask,
        page_w=w,
        page_h=h,
        table_bboxes=[[290, 290, 390, 390]],
        min_side=20,
        max_area_frac=0.5,
        min_side_evidence=3,
        side_band=5,
        side_min_pixels=5,
        min_ink_density=0.001,
        box_nms_iou=0.45,
        exclude_tables=True,
    )
    assert len(out) == 1
    assert out[0]["bbox"] == [100, 100, 250, 250]


def test_view_regions_cache_fp_v6_fields():
    from pipeline.perceive_utils import _view_regions_cache_fp

    fp = _view_regions_cache_fp(
        {
            "propose_mode": "thick_boundary",
            "min_side_evidence": 3,
            "min_ink_density": 0.002,
            "box_nms_iou": 0.45,
            "exclude_tables": True,
            "fallback_vlm": True,
        }
    )
    assert fp["v"] == 6
    assert fp["propose_mode"] == "thick_boundary"
    assert fp["min_side_evidence"] == 3
    assert "box_nms_iou" in fp
