"""部件/视图感知分区：由视图框生成尺寸识别区（外扩、扣表格、过大过滤）。"""
from __future__ import annotations

from typing import Any


def _area(bbox: list[int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _clamp_bbox(bbox: list[int], page_w: int, page_h: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(page_w - 1, x1))
    y1 = max(0, min(page_h - 1, y1))
    x2 = max(0, min(page_w, x2))
    y2 = max(0, min(page_h, y2))
    if x2 <= x1:
        x2 = min(page_w, x1 + 1)
    if y2 <= y1:
        y2 = min(page_h, y1 + 1)
    return [x1, y1, x2, y2]


def expand_view_bbox(
    bbox: list[int],
    page_w: int,
    page_h: int,
    *,
    expand_ratio: float = 0.2,
) -> list[int]:
    """按比例外扩视图框，覆盖周围尺寸标注。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    dx = int(bw * float(expand_ratio))
    dy = int(bh * float(expand_ratio))
    return _clamp_bbox([x1 - dx, y1 - dy, x2 + dx, y2 + dy], page_w, page_h)


def tighten_bbox_to_ink(
    image: Any,
    bbox: list[int],
    *,
    ink_threshold: int = 245,
    pad: int = 2,
    min_ink_pixels: int = 40,
    max_shrink_frac: float = 0.85,
) -> list[int]:
    """在 VLM 粗框内按墨迹收紧到零件几何外接矩形（贴边）。

    若墨迹过少或收紧幅度异常，返回原框。
    """
    import numpy as np

    page_w, page_h = image.size
    x1, y1, x2, y2 = _clamp_bbox(list(bbox), page_w, page_h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return [x1, y1, x2, y2]
    crop = image.crop((x1, y1, x2, y2)).convert("L")
    arr = np.asarray(crop, dtype=np.uint8)
    ink = arr < int(ink_threshold)
    if int(ink.sum()) < int(min_ink_pixels):
        return [x1, y1, x2, y2]
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return [x1, y1, x2, y2]
    tx1 = x1 + int(xs.min()) - int(pad)
    ty1 = y1 + int(ys.min()) - int(pad)
    tx2 = x1 + int(xs.max()) + 1 + int(pad)
    ty2 = y1 + int(ys.max()) + 1 + int(pad)
    tight = _clamp_bbox([tx1, ty1, tx2, ty2], page_w, page_h)
    # 防止噪声导致过度塌缩（面积缩到原框 <15% 则放弃）
    if _area(tight) < (1.0 - float(max_shrink_frac)) * _area([x1, y1, x2, y2]):
        return [x1, y1, x2, y2]
    if tight[2] - tight[0] < 8 or tight[3] - tight[1] < 8:
        return [x1, y1, x2, y2]
    return tight


def tighten_bbox_to_thick_outline(
    image: Any,
    bbox: list[int],
    *,
    ink_threshold: int = 245,
    pad: int = 2,
    thick_min_width: int = 3,
    min_ink_pixels: int = 40,
    max_shrink_frac: float = 0.85,
    min_component_area_frac: float = 0.002,
) -> list[int]:
    """按粗轮廓线收紧部件框：形态学 OPEN 抑制细线后取粗笔画外接框。

    工程图零件边框多为较粗实线；尺寸线/剖面线较细。OPEN(核≈粗线宽)
    可保留粗笔画。失败时回退 ``tighten_bbox_to_ink``。
    """
    import numpy as np

    try:
        import cv2
    except Exception:
        return tighten_bbox_to_ink(
            image,
            bbox,
            ink_threshold=ink_threshold,
            pad=pad,
            min_ink_pixels=min_ink_pixels,
            max_shrink_frac=max_shrink_frac,
        )

    page_w, page_h = image.size
    x1, y1, x2, y2 = _clamp_bbox(list(bbox), page_w, page_h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 16 or bh < 16:
        return [x1, y1, x2, y2]

    gray = np.asarray(image.crop((x1, y1, x2, y2)).convert("L"), dtype=np.uint8)
    ink = ((gray < int(ink_threshold)).astype(np.uint8)) * 255
    if int((ink > 0).sum()) < int(min_ink_pixels):
        return [x1, y1, x2, y2]

    k = max(2, int(thick_min_width))
    # 奇数核更稳；略小于宣称线宽以免打断略细的粗线
    k_open = k if k % 2 == 1 else k - 1
    k_open = max(3, k_open)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
    # 补缝：粗轮廓可能因扫描断裂
    k_close = max(3, k_open)
    close_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, close_ker, iterations=1)

    if int((thick > 0).sum()) < int(min_ink_pixels):
        return tighten_bbox_to_ink(
            image,
            bbox,
            ink_threshold=ink_threshold,
            pad=pad,
            min_ink_pixels=min_ink_pixels,
            max_shrink_frac=max_shrink_frac,
        )

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    if nlab <= 1:
        return tighten_bbox_to_ink(
            image,
            bbox,
            ink_threshold=ink_threshold,
            pad=pad,
            min_ink_pixels=min_ink_pixels,
            max_shrink_frac=max_shrink_frac,
        )

    crop_area = float(max(1, bw * bh))
    min_area = max(float(min_ink_pixels), float(min_component_area_frac) * crop_area)
    # 合并所有足够大的粗线连通域外接框（零件轮廓常非单连通）
    ux1, uy1, ux2, uy2 = bw, bh, 0, 0
    kept = 0
    for i in range(1, nlab):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        lx = int(stats[i, cv2.CC_STAT_LEFT])
        ly = int(stats[i, cv2.CC_STAT_TOP])
        lw = int(stats[i, cv2.CC_STAT_WIDTH])
        lh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ux1 = min(ux1, lx)
        uy1 = min(uy1, ly)
        ux2 = max(ux2, lx + lw)
        uy2 = max(uy2, ly + lh)
        kept += 1

    if kept == 0 or ux2 <= ux1 or uy2 <= uy1:
        return tighten_bbox_to_ink(
            image,
            bbox,
            ink_threshold=ink_threshold,
            pad=pad,
            min_ink_pixels=min_ink_pixels,
            max_shrink_frac=max_shrink_frac,
        )

    tight = _clamp_bbox(
        [
            x1 + ux1 - int(pad),
            y1 + uy1 - int(pad),
            x1 + ux2 + int(pad),
            y1 + uy2 + int(pad),
        ],
        page_w,
        page_h,
    )
    if _area(tight) < (1.0 - float(max_shrink_frac)) * _area([x1, y1, x2, y2]):
        return [x1, y1, x2, y2]
    if tight[2] - tight[0] < 8 or tight[3] - tight[1] < 8:
        return [x1, y1, x2, y2]
    return tight


def tighten_view_bbox(
    image: Any,
    bbox: list[int],
    *,
    mode: str = "thick_outline",
    ink_threshold: int = 245,
    pad: int = 2,
    thick_min_width: int = 3,
    min_ink_pixels: int = 40,
    max_shrink_frac: float = 0.85,
) -> list[int]:
    """部件框收紧入口：thick_outline（默认）或 ink（全墨迹）。"""
    m = str(mode or "thick_outline").strip().lower()
    if m in {"ink", "all_ink", "full_ink"}:
        return tighten_bbox_to_ink(
            image,
            bbox,
            ink_threshold=ink_threshold,
            pad=pad,
            min_ink_pixels=min_ink_pixels,
            max_shrink_frac=max_shrink_frac,
        )
    return tighten_bbox_to_thick_outline(
        image,
        bbox,
        ink_threshold=ink_threshold,
        pad=pad,
        thick_min_width=thick_min_width,
        min_ink_pixels=min_ink_pixels,
        max_shrink_frac=max_shrink_frac,
    )


def _overlap_xyxy(a: list[int], b: list[int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def shrink_region_away_from_tables(
    region: list[int],
    table_bboxes: list[list[int]] | None,
) -> list[int] | None:
    """识别区避开表格：中心落表内则丢弃；与表重叠则单边裁切取最大剩余。"""
    if not table_bboxes:
        return list(region)
    x1, y1, x2, y2 = [int(v) for v in region]
    for tb in table_bboxes:
        if not tb or len(tb) != 4:
            continue
        tx1, ty1, tx2, ty2 = [int(v) for v in tb]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            return None
        cur = [x1, y1, x2, y2]
        if not _overlap_xyxy(cur, [tx1, ty1, tx2, ty2]):
            continue
        candidates = [
            [x1, y1, x2, min(y2, ty1)],
            [x1, max(y1, ty2), x2, y2],
            [x1, y1, min(x2, tx1), y2],
            [max(x1, tx2), y1, x2, y2],
        ]
        best: list[int] | None = None
        best_area = -1.0
        for c in candidates:
            if c[2] - c[0] < 8 or c[3] - c[1] < 8:
                continue
            if _overlap_xyxy(c, [tx1, ty1, tx2, ty2]):
                continue
            a = _area(c)
            if a > best_area:
                best_area = a
                best = c
        if best is None:
            return None
        x1, y1, x2, y2 = best
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return [x1, y1, x2, y2]


def is_region_too_small(bbox: list[int], *, min_side: int = 64) -> bool:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) < min_side or (y2 - y1) < min_side


def is_region_too_large(
    bbox: list[int],
    page_w: int,
    page_h: int,
    *,
    max_area_frac: float = 0.35,
) -> bool:
    page_area = max(1, int(page_w) * int(page_h))
    return _area(bbox) / float(page_area) > float(max_area_frac)


def split_large_region(
    bbox: list[int],
    *,
    tile_size: int = 1280,
    overlap: float = 0.2,
) -> list[list[int]]:
    """过大识别区按局部网格拆分（坐标仍为全图）。"""
    from pipeline.perceive_utils import make_tiles

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if max(w, h) <= tile_size:
        return [list(bbox)]
    local = make_tiles(w, h, tile_size=tile_size, overlap=overlap)
    return [[x1 + a, y1 + b, x1 + c, y1 + d] for a, b, c, d in local]


def build_view_regions(
    view_bboxes: list[list[int]] | None,
    *,
    page_w: int,
    page_h: int,
    table_bboxes: list[list[int]] | None = None,
    expand_ratio: float = 0.2,
    min_side: int = 64,
    max_area_frac: float = 0.35,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
) -> list[list[int]]:
    """视图框 → 尺寸识别区列表（已外扩、扣表；过大/超边长则按 tile 再切，对齐旧网格窗口）。"""
    regions: list[list[int]] = []
    for vb in view_bboxes or []:
        if not vb or len(vb) != 4:
            continue
        expanded = expand_view_bbox(vb, page_w, page_h, expand_ratio=expand_ratio)
        shrunk = shrink_region_away_from_tables(expanded, table_bboxes)
        if shrunk is None:
            continue
        shrunk = _clamp_bbox(shrunk, page_w, page_h)
        if is_region_too_small(shrunk, min_side=min_side):
            continue
        # 与架构前 make_tiles 对齐：边长或面积过大都再切，避免「整块松散大 crop」导致 Pass1 漏尺寸
        need_split = is_region_too_large(
            shrunk, page_w, page_h, max_area_frac=max_area_frac
        ) or max(shrunk[2] - shrunk[0], shrunk[3] - shrunk[1]) > int(tile_size)
        parts = (
            split_large_region(shrunk, tile_size=tile_size, overlap=tile_overlap)
            if need_split
            else [shrunk]
        )
        for p in parts:
            p = _clamp_bbox(p, page_w, page_h)
            if is_region_too_small(p, min_side=min_side):
                continue
            p2 = shrink_region_away_from_tables(p, table_bboxes)
            if p2 is None or is_region_too_small(p2, min_side=min_side):
                continue
            regions.append(_clamp_bbox(p2, page_w, page_h))
    return regions


def uncovered_grid_regions(
    covered: list[list[int]] | None,
    *,
    page_w: int,
    page_h: int,
    table_bboxes: list[list[int]] | None = None,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
    min_cover_frac: float = 0.55,
) -> list[list[int]]:
    """整页网格中与已有部件区覆盖不足的 tile，用于补回部件外尺寸（对齐旧全图分块覆盖）。"""
    from pipeline.perceive_utils import make_tiles

    if max(page_w, page_h) <= tile_size:
        grid = [[0, 0, page_w, page_h]]
    else:
        grid = [list(t) for t in make_tiles(page_w, page_h, tile_size=tile_size, overlap=tile_overlap)]
    covered = [c for c in (covered or []) if c and len(c) == 4]
    out: list[list[int]] = []
    for tile in grid:
        tx1, ty1, tx2, ty2 = tile
        tarea = max(1.0, _area(tile))
        cover = 0.0
        for c in covered:
            ix1 = max(tx1, c[0])
            iy1 = max(ty1, c[1])
            ix2 = min(tx2, c[2])
            iy2 = min(ty2, c[3])
            if ix2 > ix1 and iy2 > iy1:
                cover += float((ix2 - ix1) * (iy2 - iy1))
        if cover / tarea >= float(min_cover_frac):
            continue
        shrunk = shrink_region_away_from_tables(tile, table_bboxes)
        if shrunk is None:
            continue
        shrunk = _clamp_bbox(shrunk, page_w, page_h)
        if is_region_too_small(shrunk, min_side=32):
            continue
        out.append(shrunk)
    return out


def assign_parent_by_components(
    instances: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将尚无 parent_id 的尺寸框，按中心落入的部件扩展框关联。"""
    comps = [c for c in components or [] if c.get("bbox") or c.get("bbox_expanded")]
    if not comps:
        return instances
    out: list[dict[str, Any]] = []
    for inst in instances or []:
        item = dict(inst)
        if item.get("parent_id"):
            out.append(item)
            continue
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            out.append(item)
            continue
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        best_id = None
        best_area = None
        for c in comps:
            box = c.get("bbox_expanded") or c.get("bbox")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                a = (x2 - x1) * (y2 - y1)
                if best_area is None or a < best_area:
                    best_area = a
                    best_id = c.get("instance_id") or c.get("part_id")
        if best_id:
            item["parent_id"] = best_id
        out.append(item)
    return out


def build_part_region_specs(
    views: list[dict[str, Any]] | None,
    *,
    page_w: int,
    page_h: int,
    table_bboxes: list[list[int]] | None = None,
    expand_ratio: float = 0.2,
    min_side: int = 64,
    max_area_frac: float = 0.35,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
) -> tuple[list[dict[str, Any]], str]:
    """由带 label 的视图生成识别规格；无有效视图时回退网格。

    返回 (specs, source)。spec 含:
      part_id, label, bbox_raw, bbox_expanded, regions[]
    source: view_regions | grid | full_page
    """
    specs: list[dict[str, Any]] = []
    for i, view in enumerate(views or []):
        if not isinstance(view, dict):
            continue
        bbox = view.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        raw = _clamp_bbox(list(bbox), page_w, page_h)
        label = str(view.get("label") or f"view#{i}").strip() or f"view#{i}"
        part_id = f"component#{i}"
        expanded = expand_view_bbox(raw, page_w, page_h, expand_ratio=expand_ratio)
        regions = build_view_regions(
            [raw],
            page_w=page_w,
            page_h=page_h,
            table_bboxes=table_bboxes,
            expand_ratio=expand_ratio,
            min_side=min_side,
            max_area_frac=max_area_frac,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        if not regions:
            continue
        specs.append(
            {
                "part_id": part_id,
                "label": label,
                "bbox_raw": raw,
                "bbox_expanded": expanded,
                "regions": regions,
            }
        )
    if specs:
        return specs, "view_regions"
    return [], "empty"


def resolve_dimension_regions(
    view_bboxes: list[list[int]] | None,
    *,
    page_w: int,
    page_h: int,
    table_bboxes: list[list[int]] | None = None,
    expand_ratio: float = 0.2,
    min_side: int = 64,
    max_area_frac: float = 0.35,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
    fallback_grid: bool = True,
) -> tuple[list[list[int]], str]:
    """生成尺寸识别区；无有效视图时可选回退全图网格。

    返回 (regions, source) source 为 view_regions | grid | full_page。
    """
    regions = build_view_regions(
        view_bboxes,
        page_w=page_w,
        page_h=page_h,
        table_bboxes=table_bboxes,
        expand_ratio=expand_ratio,
        min_side=min_side,
        max_area_frac=max_area_frac,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if regions:
        return regions, "view_regions"
    if not fallback_grid:
        return [[0, 0, page_w, page_h]], "full_page"
    from pipeline.perceive_utils import make_tiles

    if max(page_w, page_h) <= tile_size:
        return [[0, 0, page_w, page_h]], "full_page"
    tiles = make_tiles(page_w, page_h, tile_size=tile_size, overlap=tile_overlap)
    return [list(t) for t in tiles], "grid"


def resolve_part_region_specs(
    views: list[dict[str, Any]] | None,
    *,
    page_w: int,
    page_h: int,
    table_bboxes: list[list[int]] | None = None,
    expand_ratio: float = 0.2,
    min_side: int = 64,
    max_area_frac: float = 0.35,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
    fallback_grid: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """部件规格列表；失败时回退为无 parent 的网格规格。"""
    specs, src = build_part_region_specs(
        views,
        page_w=page_w,
        page_h=page_h,
        table_bboxes=table_bboxes,
        expand_ratio=expand_ratio,
        min_side=min_side,
        max_area_frac=max_area_frac,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if specs:
        return specs, src
    regions, grid_src = resolve_dimension_regions(
        [],
        page_w=page_w,
        page_h=page_h,
        table_bboxes=table_bboxes,
        expand_ratio=expand_ratio,
        min_side=min_side,
        max_area_frac=max_area_frac,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        fallback_grid=fallback_grid,
    )
    # 网格回退：无部件关联
    return (
        [
            {
                "part_id": None,
                "label": "grid",
                "bbox_raw": None,
                "bbox_expanded": None,
                "regions": regions,
            }
        ],
        grid_src,
    )
