"""标题栏多表定位：形态学线网候选 + 角色分类（aux / material / main）。"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from pipeline.table_layout_ocr import (
    apply_product_split_to_boxes,
    run_ocr_boxes,
    union_bbox,
)


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _x_overlap_ratio(a: list[int], b: list[int]) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    if inter <= 0:
        return 0.0
    return inter / float(max(1, min(ax2 - ax1, bx2 - bx1)))


def detect_bordered_table_candidates(
    page_image: Image.Image,
    roi: list[int],
    *,
    ink_threshold: int = 245,
    min_area: int = 800,
    min_side: int = 24,
) -> list[list[int]]:
    """在 ROI 内用横竖线形态学提取有线表候选外接框（页坐标 xyxy）。"""
    try:
        import cv2
    except ImportError:
        return []

    x1, y1, x2, y2 = [int(v) for v in roi]
    if x2 - x1 < 40 or y2 - y1 < 40:
        return []
    crop = page_image.crop((x1, y1, x2, y2)).convert("L")
    gray = np.asarray(crop)
    binary = (gray < int(ink_threshold)).astype(np.uint8) * 255
    h, w = binary.shape[:2]
    h_len = max(15, w // 18)
    v_len = max(15, h // 18)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[int]] = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < min_side or bh < min_side:
            continue
        if bw * bh < min_area:
            continue
        # 过扁/过窄的单条线丢弃
        if bw > 8 * max(1, bh) or bh > 8 * max(1, bw):
            continue
        out.append([x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh])
    out.sort(key=lambda b: (b[1], b[0]))
    return out


def split_bbox_by_horizontal_gaps(
    page_image: Image.Image,
    bbox: list[int],
    *,
    ink_threshold: int = 245,
    min_gap: int = 10,
    min_strip_h: int = 28,
) -> list[list[int]]:
    """若单框内存在明显水平空白带，切成多条（页坐标）。无线网时的兜底。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if y2 - y1 < min_strip_h * 2:
        return [bbox]
    crop = page_image.crop((x1, y1, x2, y2)).convert("L")
    gray = np.asarray(crop)
    ink = gray < int(ink_threshold)
    row_ink = ink.any(axis=1)
    gaps: list[tuple[int, int]] = []
    i = 0
    n = len(row_ink)
    while i < n:
        if not row_ink[i]:
            j = i
            while j < n and not row_ink[j]:
                j += 1
            if j - i >= min_gap:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    if not gaps:
        return [bbox]
    cuts = [0]
    for gs, ge in gaps:
        mid = (gs + ge) // 2
        if mid - cuts[-1] >= min_strip_h and n - mid >= min_strip_h:
            cuts.append(mid)
    cuts.append(n)
    strips: list[list[int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < min_strip_h:
            continue
        strips.append([x1, y1 + a, x2, y1 + b])
    return strips if len(strips) >= 2 else [bbox]


def _find_product_y(
    page_image: Image.Image,
    search_bbox: list[int],
    *,
    conf_th: float = 0.35,
) -> tuple[int | None, list[str]]:
    notes: list[str] = []
    x1, y1, x2, y2 = [int(v) for v in search_bbox]
    if x2 <= x1 or y2 <= y1:
        return None, ["aux_stack:product_search_empty"]
    crop = page_image.crop((x1, y1, x2, y2))
    items = run_ocr_boxes(crop, conf_th=conf_th)
    pat = re.compile(r"\bproduct\b", re.I)
    cands = [it for it in items if pat.search(it["text"]) and len(it["text"]) <= 48]
    if not cands:
        notes.append("aux_stack:product_not_found")
        return None, notes
    cands.sort(key=lambda it: it["bbox"][1])
    chosen = cands[-1]
    prod_y = y1 + int(chosen["bbox"][1])
    notes.append(f"aux_stack:product_y={prod_y}")
    return prod_y, notes


def _section_of_box(box: dict[str, Any], plan: list[dict[str, Any]]) -> str:
    eid = str(box.get("entity_id") or "")
    if eid.startswith("aux_table") or eid == "aux_table":
        return "aux"
    if eid == "material_table" or eid.endswith("material_table"):
        return "material"
    if eid == "main_table" or eid.endswith("main_table"):
        return "main"
    for e in plan:
        if e.get("entity_id") == eid:
            return str(e.get("section") or "")
    fields = box.get("fields") if isinstance(box.get("fields"), dict) else {}
    return str(fields.get("section") or "")


def _make_aux_instance(bbox: list[int], idx: int) -> dict[str, Any]:
    return {
        "entity_id": "aux_table" if idx == 0 else f"aux_table_{idx}",
        "instance_id": f"aux_table#{idx}",
        "label": "aux_table",
        "bbox": [int(v) for v in bbox],
        "fields": {
            "section": "aux",
            "read_mode": "bbox_only",
            "pairs": [],
        },
        "raw_text": "",
        "confidence": 0.75,
        "needs_review": False,
        "parse_kind": "table",
    }


def _shrink_material_away_from_aux(
    mat_bb: list[int],
    aux_bbs: list[list[int]],
    *,
    gap: int = 2,
) -> list[int]:
    """物料框若与上方附属表重叠，上抬 y1 到最下附属表之下。"""
    x1, y1, x2, y2 = [int(v) for v in mat_bb]
    for ab in aux_bbs:
        if _x_overlap_ratio(mat_bb, ab) < 0.35:
            continue
        ay2 = int(ab[3])
        if ay2 > y1 and ay2 < y2 - 8:
            y1 = max(y1, ay2 + gap)
    if y1 >= y2 - 4:
        y1 = max(0, y2 - 20)
    return [x1, y1, x2, y2]


def separate_aux_tables(
    page_image: Image.Image,
    boxes: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    gap: int = 2,
    search_up_pad: int = 80,
    ink_threshold: int = 245,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """在 material 上方检出/拆出附属表；material/main 字段路径不改。"""
    notes: list[str] = []
    if not enabled:
        notes.append("aux_split:disabled")
        return boxes, notes

    out = [dict(b) for b in boxes]
    mat_idxs = [
        i
        for i, b in enumerate(out)
        if _section_of_box(b, plan) == "material" or b.get("entity_id") == "material_table"
    ]
    main_idxs = [
        i
        for i, b in enumerate(out)
        if _section_of_box(b, plan) == "main" or b.get("entity_id") == "main_table"
    ]
    existing_aux = [
        i
        for i, b in enumerate(out)
        if _section_of_box(b, plan) == "aux" or str(b.get("entity_id") or "").startswith("aux_table")
    ]

    seeds = []
    for i in mat_idxs + main_idxs + existing_aux:
        bb = out[i].get("bbox")
        if bb and len(bb) == 4:
            seeds.append(list(bb))
    uni = union_bbox(seeds)
    if uni is None:
        notes.append("aux_split:no_seed")
        return out, notes

    ux1 = max(0, uni[0] - 8)
    ux2 = min(page_w, uni[2] + 8)
    uy2 = min(page_h, uni[3] + 8)
    uy1 = max(0, uni[1] - search_up_pad)
    roi = [ux1, uy1, ux2, uy2]

    # 1) 线网候选
    candidates = detect_bordered_table_candidates(
        page_image, roi, ink_threshold=ink_threshold
    )
    notes.append(f"aux_split:line_cands={len(candidates)}")

    # 2) 若 material 框内可切多段，加入切条
    for mi in mat_idxs:
        mb = list(out[mi]["bbox"])
        strips = split_bbox_by_horizontal_gaps(
            page_image, mb, ink_threshold=ink_threshold
        )
        if len(strips) >= 2:
            notes.append(f"aux_split:material_strips={len(strips)}")
            # 最下条保留为 material，上方为候选 aux
            out[mi]["bbox"] = strips[-1]
            candidates.extend(strips[:-1])

    prod_y, pn = _find_product_y(page_image, [ux1, uy1, ux2, uy2])
    notes.extend(pn)

    mat_bb = None
    if mat_idxs:
        mat_bb = list(out[mat_idxs[0]]["bbox"])
    elif prod_y is not None:
        mat_bb = [ux1, max(uy1, prod_y - 120), ux2, prod_y]

    aux_bbs: list[list[int]] = []
    # 已有 VLM aux 框
    for i in existing_aux:
        bb = out[i].get("bbox")
        if bb:
            aux_bbs.append(list(bb))

    for cand in candidates:
        if mat_bb is None:
            # 无 material 时：Product 以上全部可作 aux（保守：仅保留明显在上的）
            if prod_y is not None and cand[3] <= prod_y + 2:
                if all(_iou_xyxy(cand, a) < 0.5 for a in aux_bbs):
                    aux_bbs.append(cand)
            continue
        # 必须大体在 material 上方，且与 material x 有重叠
        if _x_overlap_ratio(cand, mat_bb) < 0.3:
            continue
        if cand[3] > mat_bb[1] + 12 and cand[1] >= mat_bb[1] - 4:
            # 与 material 同层或偏下 → 不是上方附属表
            continue
        # 主要部分在 material 顶边之上
        if cand[3] > mat_bb[1] + gap:
            # 裁切到 material 之上
            cand = [cand[0], cand[1], cand[2], min(cand[3], mat_bb[1] - gap)]
        if cand[3] - cand[1] < 20:
            continue
        if prod_y is not None and cand[1] >= prod_y:
            continue
        # 勿与 main 混淆
        skip = False
        for ni in main_idxs:
            nb = out[ni].get("bbox")
            if nb and _iou_xyxy(cand, list(nb)) > 0.4:
                skip = True
                break
        if skip:
            continue
        if any(_iou_xyxy(cand, a) > 0.55 for a in aux_bbs):
            continue
        # 若候选几乎就是 material 本身，跳过
        if mat_bb and _iou_xyxy(cand, mat_bb) > 0.55:
            continue
        aux_bbs.append(cand)

    # 按 y 排序，去重叠
    aux_bbs.sort(key=lambda b: (b[1], b[0]))
    cleaned: list[list[int]] = []
    for b in aux_bbs:
        if any(_iou_xyxy(b, c) > 0.55 for c in cleaned):
            continue
        cleaned.append(b)
    aux_bbs = cleaned

    if mat_idxs and aux_bbs:
        new_mat = _shrink_material_away_from_aux(
            list(out[mat_idxs[0]]["bbox"]), aux_bbs, gap=gap
        )
        if new_mat != list(out[mat_idxs[0]]["bbox"]):
            notes.append(
                f"aux_split:trim_material_y1:{out[mat_idxs[0]]['bbox'][1]}->{new_mat[1]}"
            )
            out[mat_idxs[0]]["bbox"] = new_mat
            # 同步其它 material 实例
            for mi in mat_idxs[1:]:
                out[mi]["bbox"] = list(new_mat)

    # 写入 / 更新 aux 实例
    if existing_aux and aux_bbs:
        # 用检出框覆盖已有 aux（取前 N 个）
        for j, i in enumerate(existing_aux):
            if j < len(aux_bbs):
                out[i]["bbox"] = aux_bbs[j]
                fields = out[i].get("fields") if isinstance(out[i].get("fields"), dict) else {}
                fields = dict(fields)
                fields["section"] = "aux"
                fields["read_mode"] = "bbox_only"
                fields.setdefault("pairs", [])
                out[i]["fields"] = fields
                out[i]["parse_kind"] = "table"
        start = len(existing_aux)
        for j, bb in enumerate(aux_bbs[start:]):
            out.append(_make_aux_instance(bb, start + j))
    elif aux_bbs:
        for j, bb in enumerate(aux_bbs):
            out.append(_make_aux_instance(bb, j))

    notes.append(f"aux_split:aux_n={len(aux_bbs)}")
    return out, notes


def enforce_aux_above_material(
    boxes: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    *,
    gap: int = 2,
) -> tuple[list[dict[str, Any]], list[str]]:
    """硬约束：aux.y2 ≤ material.y1 - gap。"""
    notes: list[str] = []
    out = [dict(b) for b in boxes]
    mat_idxs = [
        i
        for i, b in enumerate(out)
        if _section_of_box(b, plan) == "material" or b.get("entity_id") == "material_table"
    ]
    aux_idxs = [
        i
        for i, b in enumerate(out)
        if _section_of_box(b, plan) == "aux" or str(b.get("entity_id") or "").startswith("aux_table")
    ]
    if not mat_idxs or not aux_idxs:
        return out, notes
    mat_y1 = min(int(out[i]["bbox"][1]) for i in mat_idxs if out[i].get("bbox"))
    for i in aux_idxs:
        bb = list(out[i]["bbox"])
        if bb[3] > mat_y1 - gap:
            new_y2 = mat_y1 - gap
            if new_y2 - bb[1] < 16:
                notes.append(f"aux_enforce:drop_thin:{out[i].get('entity_id')}")
                out[i]["_drop"] = True
            else:
                notes.append(f"aux_enforce:trim_y2:{bb[3]}->{new_y2}")
                bb[3] = new_y2
                out[i]["bbox"] = bb
    kept = [b for b in out if not b.pop("_drop", False)]
    return kept, notes


def refine_title_block_tables(
    page_image: Image.Image,
    boxes: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    plan: list[dict[str, Any]] | None = None,
    aux_enabled: bool = True,
    ink_threshold: int = 245,
    search_up_pad: int = 80,
    material_main_refiner=None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Product/几何分界（material/main）后做附属表分离。

    material_main_refiner: 可选回调 (image, boxes, page_w, page_h, plan) → (boxes, notes)
    未提供时仅做 Product OCR 分界。
    """
    plan = plan or []
    notes: list[str] = []
    if material_main_refiner is not None:
        cur, ref_notes = material_main_refiner(
            page_image, boxes, page_w=page_w, page_h=page_h, plan=plan
        )
        notes.extend(ref_notes)
    else:
        cur, split_notes = apply_product_split_to_boxes(page_image, boxes, plan)
        notes.extend(split_notes)
        if any(n.startswith("product_split:row=") for n in split_notes):
            notes.append("boundary_source:product_row_ocr")
        else:
            notes.append("boundary_source:product_split_miss")

    cur, aux_notes = separate_aux_tables(
        page_image,
        cur,
        plan,
        page_w=page_w,
        page_h=page_h,
        search_up_pad=search_up_pad,
        ink_threshold=ink_threshold,
        enabled=aux_enabled,
    )
    notes.extend(aux_notes)
    cur, enf_notes = enforce_aux_above_material(cur, plan, gap=2)
    notes.extend(enf_notes)
    return cur, notes
