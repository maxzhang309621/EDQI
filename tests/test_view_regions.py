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


def test_drop_oversized_and_reindex_preserves_component_parent():
    from pipeline.perceive_utils import _drop_oversized_dimension_boxes, reindex_instances

    insts = [
        {"entity_id": "number_mark", "bbox": [10, 10, 50, 40], "confidence": 0.9, "parent_id": "component#0"},
        {"entity_id": "number_mark", "bbox": [0, 0, 2000, 1000], "confidence": 0.99, "parent_id": "component#0"},
        {
            "entity_id": "component",
            "instance_id": "component#0",
            "bbox": [100, 100, 400, 400],
            "label": "front",
        },
    ]
    kept = _drop_oversized_dimension_boxes(insts[:2], page_w=2000, page_h=1500, max_area_frac=0.35)
    assert len(kept) == 1
    out = reindex_instances(kept + [insts[2]])
    assert out[0]["parent_id"] == "component#0"
    assert out[1]["instance_id"] == "component#0"
