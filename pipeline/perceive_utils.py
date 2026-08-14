"""感知缓存、分块、IoU 与模型就绪检测辅助。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from pipeline import dump_json, load_json, resolve_path, model_weights_ready


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_instances(
    instances: list[dict[str, Any]],
    iou_thr: float = 0.5,
    *,
    prefer_smaller: bool = False,
    use_quad: bool = False,
) -> list[dict[str, Any]]:
    """同类 entity NMS；use_quad 时对倾斜框用旋转 IoU（Skew-NMS）。"""
    by_eid: dict[str, list[dict[str, Any]]] = {}
    for inst in instances:
        by_eid.setdefault(inst.get("entity_id", "unknown"), []).append(inst)
    kept: list[dict[str, Any]] = []

    def _area_of(x: dict[str, Any]) -> float:
        if use_quad and x.get("quad"):
            try:
                import cv2
                import numpy as np

                pts = np.asarray(x["quad"], dtype=np.float32).reshape(-1, 2)
                return float(abs(cv2.contourArea(pts)))
            except Exception:
                pass
        bbox = x.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            return 0.0
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
            0.0, float(bbox[3]) - float(bbox[1])
        )

    def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        if use_quad and a.get("quad") and b.get("quad"):
            from pipeline.oriented_box import iou_quad

            return float(iou_quad(a["quad"], b["quad"]))
        if a.get("bbox") and b.get("bbox"):
            return float(iou_xyxy(a["bbox"], b["bbox"]))
        return 0.0

    for group in by_eid.values():
        def _sort_key(x: dict[str, Any]) -> tuple:
            conf = float(x.get("confidence") or 0)
            area = _area_of(x)
            return (-conf, area if prefer_smaller else -area)

        group = sorted(group, key=_sort_key)
        selected: list[dict[str, Any]] = []
        for cand in group:
            if all(_iou(cand, s) < iou_thr for s in selected):
                selected.append(cand)
        kept.extend(selected)
    return kept


def filter_dimension_accuracy(
    instances: list[dict[str, Any]],
    *,
    drop_weak: bool = True,
    dedupe_same_text: bool = True,
    center_dist_thr: float = 28.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """尺寸准确率后处理：弱结果剔除 + 同文近距去重。"""
    notes: list[str] = []
    kept: list[dict[str, Any]] = []
    n_weak = 0
    for inst in instances or []:
        item = dict(inst)
        if str(item.get("entity_id") or "") != "number_mark":
            kept.append(item)
            continue
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        text = str(fields.get("text") or item.get("raw_text") or "").strip()
        weak = False
        if drop_weak:
            if not text:
                weak = True
            elif not any(ch.isdigit() for ch in text) and fields.get("dim_kind") in (None, ""):
                weak = True
        if weak:
            n_weak += 1
            continue
        kept.append(item)
    if n_weak:
        notes.append(f"dim_accuracy_drop_weak={n_weak}")

    if not dedupe_same_text:
        return kept, notes

    def _center(b: list[Any]) -> tuple[float, float]:
        return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)

    def _norm_text(t: str) -> str:
        return "".join(ch for ch in t.lower() if ch.isalnum() or ch in "±.+-°ø⌀φ")

    out: list[dict[str, Any]] = []
    n_dup = 0
    for inst in kept:
        if str(inst.get("entity_id") or "") != "number_mark":
            out.append(inst)
            continue
        fields = inst.get("fields") if isinstance(inst.get("fields"), dict) else {}
        text = _norm_text(str(fields.get("text") or inst.get("raw_text") or ""))
        bb = inst.get("bbox") or []
        if not text or len(bb) != 4:
            out.append(inst)
            continue
        cx, cy = _center(bb)
        dup = False
        for prev in out:
            if str(prev.get("entity_id") or "") != "number_mark":
                continue
            pf = prev.get("fields") if isinstance(prev.get("fields"), dict) else {}
            pt = _norm_text(str(pf.get("text") or prev.get("raw_text") or ""))
            if pt != text:
                continue
            pb = prev.get("bbox") or []
            if len(pb) != 4:
                continue
            px, py = _center(pb)
            if (cx - px) ** 2 + (cy - py) ** 2 <= float(center_dist_thr) ** 2:
                # 同 parent 或都无 parent 才去重
                if (inst.get("parent_id") or None) == (prev.get("parent_id") or None):
                    dup = True
                    break
        if dup:
            n_dup += 1
            continue
        out.append(inst)
    if n_dup:
        notes.append(f"dim_accuracy_dedupe_text={n_dup}")
    return out, notes


class PerceptionCache:
    """按 (image_sig, backend, locate_query, fields) 缓存 instances。"""

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.dir = resolve_path(cache_dir)
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def image_signature(image_path: Path) -> str:
        """页图签名：优先用 ingest_sig（不因重复保存改 mtime 而失效）。"""
        image_path = Path(image_path)
        meta_path = image_path.with_name(f"{image_path.stem}_meta.json")
        if meta_path.exists():
            try:
                meta = load_json(meta_path)
                ingest_sig = meta.get("ingest_sig") if isinstance(meta, dict) else None
                if ingest_sig:
                    return hashlib.sha1(str(ingest_sig).encode("utf-8")).hexdigest()[:16]
            except Exception:
                pass
        # 回退：内容哈希（比 mtime 稳定）
        h = hashlib.sha1()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    @staticmethod
    def plan_key(ent: dict[str, Any]) -> str:
        fields = sorted(
            (f.get("name"), f.get("parse_hint"), bool(f.get("required")))
            for f in ent.get("fields") or []
        )
        payload = {
            "entity_id": ent.get("entity_id"),
            "locate_query": ent.get("locate_query"),
            "fields": fields,
            "read_mode": ent.get("read_mode"),
            "section": ent.get("section"),
            # 物料表 OCR 增强变更时必须失效缓存，避免沿用空结果
            "ocr_enhance": ent.get("ocr_enhance") or {},
            # 部件分区配置变更时失效旧网格分块缓存
            "_perception_view_regions": ent.get("_perception_view_regions"),
            # 尺寸多尺度/小字放大配置变更时失效缓存
            "_perception_dim_vlm": ent.get("_perception_dim_vlm"),
            # OBB / 准确率过滤变更时失效缓存
            "_perception_dim_obb": ent.get("_perception_dim_obb"),
            "_perception_dim_accuracy": ent.get("_perception_dim_accuracy"),
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

    def _path(self, backend: str, img_sig: str, ent_key: str) -> Path:
        return self.dir / backend / img_sig / f"{ent_key}.json"

    def get(self, backend: str, img_sig: str, ent: dict[str, Any]) -> list[dict[str, Any]] | None:
        if not self.enabled:
            return None
        path = self._path(backend, img_sig, self.plan_key(ent))
        if not path.exists():
            return None
        try:
            data = load_json(path)
            return data.get("instances")
        except Exception:
            return None

    def set(self, backend: str, img_sig: str, ent: dict[str, Any], instances: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        path = self._path(backend, img_sig, self.plan_key(ent))
        dump_json(
            {
                "entity_id": ent.get("entity_id"),
                "locate_query": ent.get("locate_query"),
                "saved_at": time.time(),
                "instances": instances,
            },
            path,
        )


def make_tiles(
    width: int,
    height: int,
    *,
    tile_size: int = 1280,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """返回原图像素坐标下的 tile 框列表 [x1,y1,x2,y2]。"""
    if max(width, height) <= tile_size:
        return [(0, 0, width, height)]
    step = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []
    y = 0
    while y < height:
        x = 0
        y2 = min(height, y + tile_size)
        while x < width:
            x2 = min(width, x + tile_size)
            tiles.append((x, y, x2, y2))
            if x2 >= width:
                break
            x += step
        if y2 >= height:
            break
        y += step
    return tiles


def shift_instances(instances: list[dict[str, Any]], ox: int, oy: int) -> list[dict[str, Any]]:
    out = []
    for inst in instances:
        item = dict(inst)
        bbox = list(inst.get("bbox") or [0, 0, 0, 0])
        if len(bbox) == 4:
            item["bbox"] = [bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy]
        quad = inst.get("quad")
        if isinstance(quad, list) and quad:
            item["quad"] = [
                [float(p[0]) + ox, float(p[1]) + oy] for p in quad if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
        out.append(item)
    return out


def reindex_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    out = []
    for inst in instances:
        eid = inst.get("entity_id", "unknown")
        item = dict(inst)
        existing = str(inst.get("instance_id") or "")
        # 部件实例保留预设 id，避免破坏属性 parent_id 关联
        if eid == "component" and existing.startswith("component#"):
            item["instance_id"] = existing
            try:
                counters[eid] = max(counters.get(eid, 0), int(existing.split("#", 1)[1]) + 1)
            except ValueError:
                counters[eid] = counters.get(eid, 0) + 1
        else:
            idx = counters.get(eid, 0)
            counters[eid] = idx + 1
            item["instance_id"] = f"{eid}#{idx}"
        out.append(item)
    return out


def _is_table_plan_entity(ent: dict[str, Any]) -> bool:
    if ent.get("parse_kind") == "table":
        return True
    eid = str(ent.get("entity_id") or "")
    return eid in {"info_table", "material_table", "main_table"} or eid.endswith("_table")


def _is_dimension_plan_entity(ent: dict[str, Any]) -> bool:
    if str(ent.get("parse_kind") or "").lower() == "dimension_marks":
        return True
    return str(ent.get("entity_id") or "") == "number_mark"


def _drop_oversized_dimension_boxes(
    instances: list[dict[str, Any]],
    *,
    page_w: int,
    page_h: int,
    max_area_frac: float = 0.35,
) -> list[dict[str, Any]]:
    """丢弃半页级尺寸框，避免融框压制小标注。"""
    page_area = max(1, int(page_w) * int(page_h))
    thr = float(max_area_frac) * float(page_area)
    kept: list[dict[str, Any]] = []
    for inst in instances or []:
        bbox = inst.get("bbox") or []
        if len(bbox) != 4:
            kept.append(inst)
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if max(0.0, x2 - x1) * max(0.0, y2 - y1) > thr:
            continue
        kept.append(inst)
    return kept


def _view_regions_cache_fp(vr_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": 6,
        "enabled": bool(vr_cfg.get("enabled", True)),
        "expand_ratio": float(vr_cfg.get("expand_ratio", 0.18)),
        "min_side": int(vr_cfg.get("min_side", 64)),
        "max_area_frac": float(vr_cfg.get("max_area_frac", 0.35)),
        "fallback_grid": bool(vr_cfg.get("fallback_grid", True)),
        "tile_size": int(vr_cfg.get("tile_size", 1280)),
        "tile_overlap": float(vr_cfg.get("tile_overlap", 0.2)),
        "propose_mode": str(vr_cfg.get("propose_mode", "thick_boundary")),
        "fallback_vlm": bool(vr_cfg.get("fallback_vlm", True)),
        "min_side_evidence": int(vr_cfg.get("min_side_evidence", 3)),
        "min_ink_density": float(vr_cfg.get("min_ink_density", 0.002)),
        "box_nms_iou": float(vr_cfg.get("box_nms_iou", 0.45)),
        "exclude_tables": bool(vr_cfg.get("exclude_tables", True)),
        "tighten_to_ink": bool(vr_cfg.get("tighten_to_ink", True)),
        "tighten_mode": str(vr_cfg.get("tighten_mode", "thick_outline")),
        "thick_min_width": int(vr_cfg.get("thick_min_width", 3)),
        "ink_threshold": int(vr_cfg.get("ink_threshold", 245)),
        "tighten_pad": int(vr_cfg.get("tighten_pad", 2)),
        "fill_uncovered_grid": bool(vr_cfg.get("fill_uncovered_grid", True)),
        "drop_oversized_dims": bool(vr_cfg.get("drop_oversized_dims", False)),
    }


def _run_perceive_on_regions(
    *,
    image: Any,
    regions: list[list[int]],
    ent: dict[str, Any],
    meta: dict[str, Any],
    config: dict[str, Any],
    perceive_fn: Callable[..., dict[str, Any]],
    tile_dir: Path,
    notes: list[str],
    parent_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """在给定全图像素区域内调用 perceive_fn，并还原坐标；可选写入 parent_id。"""
    collected: list[dict[str, Any]] = []
    timing_acc: dict[str, Any] | None = None
    tile_dir.mkdir(parents=True, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(regions):
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
        tile_path = tile_dir / f"region_{i:02d}.png"
        crop.save(tile_path)
        tile_meta = {
            **meta,
            "width": crop.size[0],
            "height": crop.size[1],
            "tile_origin": [int(x1), int(y1)],
        }
        payload = perceive_fn(tile_path, [ent], tile_meta, config, allow_mock_fallback=True)
        shifted = shift_instances(payload.get("instances") or [], int(x1), int(y1))
        if parent_id:
            for item in shifted:
                item["parent_id"] = parent_id
        collected.extend(shifted)
        for n in payload.get("notes") or []:
            notes.append(str(n))
        if payload.get("note"):
            notes.append(str(payload["note"]))
        if isinstance(payload.get("timing"), dict):
            timing_acc = dict(payload["timing"])
    return collected, timing_acc


def _make_component_instance(spec: dict[str, Any]) -> dict[str, Any] | None:
    part_id = spec.get("part_id")
    bbox = spec.get("bbox_raw")
    if not part_id or not bbox or len(bbox) != 4:
        return None
    label = str(spec.get("label") or part_id)
    return {
        "entity_id": "component",
        "instance_id": str(part_id),
        "label": label,
        "bbox": list(bbox),
        "bbox_expanded": list(spec["bbox_expanded"]) if spec.get("bbox_expanded") else None,
        "fields": {"label": label},
        "raw_text": label,
        "confidence": 0.85,
        "needs_review": False,
    }


def perceive_with_cache_and_tiles(
    *,
    backend_name: str,
    image_path: Path,
    plan: list[dict[str, Any]],
    meta: dict[str, Any],
    config: dict[str, Any],
    perceive_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """
    对每个 entity 查缓存；未命中则整图或分块调用 perceive_fn。
    表格实体（material/main）合并一次调用，以便 Pass1 后做硬分界。
    尺寸属性（dimension_marks）优先「部件/视图分区」，失败回退网格；属性经 parent_id 关联部件。
    """
    perc_cfg = config.get("perception", {}) or {}
    cache = PerceptionCache(
        perc_cfg.get("cache_dir", "work_dirs/cache/perception"),
        enabled=bool(perc_cfg.get("cache_enabled", True)),
    )
    img_sig = PerceptionCache.image_signature(Path(image_path))
    tile_enabled = bool(perc_cfg.get("tile_enabled", True))
    tile_size = int(perc_cfg.get("tile_size", 1280))
    overlap = float(perc_cfg.get("tile_overlap", 0.2))
    nms_thr = float(perc_cfg.get("nms_iou", 0.5))
    vr_cfg = perc_cfg.get("view_regions") or {}
    view_regions_enabled = bool(vr_cfg.get("enabled", True))

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    all_instances: list[dict[str, Any]] = []
    notes: list[str] = []
    timing_acc: dict[str, Any] | None = None

    table_plan = [e for e in plan if _is_table_plan_entity(e)]
    other_plan = [e for e in plan if not _is_table_plan_entity(e)]

    def _finalize_entity(
        ent: dict[str, Any],
        collected: list[dict[str, Any]],
        *,
        cache_ent: dict[str, Any] | None = None,
        drop_oversized: bool = False,
        prefer_smaller: bool = False,
        use_quad: bool = False,
        apply_accuracy: bool = False,
    ) -> list[dict[str, Any]]:
        if drop_oversized:
            collected = _drop_oversized_dimension_boxes(
                collected,
                page_w=width,
                page_h=height,
                max_area_frac=float(vr_cfg.get("max_area_frac", 0.35)),
            )
        if apply_accuracy:
            acc_cfg = (config.get("perception") or {}).get("dimension_accuracy") or {}
            collected, acc_notes = filter_dimension_accuracy(
                collected,
                drop_weak=bool(acc_cfg.get("drop_weak", True)),
                dedupe_same_text=bool(acc_cfg.get("dedupe_same_text", True)),
                center_dist_thr=float(acc_cfg.get("center_dist_thr", 28.0)),
            )
            notes.extend(acc_notes)
        collected = nms_instances(
            collected,
            nms_thr,
            prefer_smaller=prefer_smaller,
            use_quad=use_quad,
        )
        fixed = []
        for c in collected or []:
            item = dict(c)
            item["entity_id"] = ent["entity_id"]
            if not item.get("label"):
                item["label"] = ent.get("locate_query", ent["entity_id"])
            fixed.append(item)
        cache.set(backend_name, img_sig, cache_ent or ent, fixed)
        return fixed

    # ---- 表格：同批感知，保证硬分界可见两框 ----
    if table_plan:
        cached_parts: list[list[dict[str, Any]]] = []
        all_hit = True
        for ent in table_plan:
            hit = cache.get(backend_name, img_sig, ent)
            if hit is None:
                all_hit = False
                break
            cached_parts.append(hit)
        if all_hit:
            for part, ent in zip(cached_parts, table_plan):
                all_instances.extend(part)
                notes.append(f"cache_hit:{ent['entity_id']}")
        else:
            payload = perceive_fn(image_path, table_plan, meta, config, allow_mock_fallback=True)
            collected = list(payload.get("instances") or [])
            if payload.get("note"):
                notes.append(str(payload["note"]))
            for n in payload.get("notes") or []:
                notes.append(str(n))
            if isinstance(payload.get("timing"), dict):
                timing_acc = dict(payload["timing"])
            notes.append(f"tables_batch:{[e.get('entity_id') for e in table_plan]}")
            for ent in table_plan:
                subset = [c for c in collected if c.get("entity_id") == ent["entity_id"]]
                all_instances.extend(_finalize_entity(ent, subset))

    # ---- 其它实体：尺寸走视图分区；其余仍按网格/整图 ----
    for ent in other_plan:
        cache_ent = dict(ent)
        is_dim = _is_dimension_plan_entity(ent)
        if view_regions_enabled and is_dim:
            cache_ent["_perception_view_regions"] = _view_regions_cache_fp(vr_cfg)
        if is_dim:
            dcfg = (config.get("perception") or {}).get("dimension_vlm") or {}
            cache_ent["_perception_dim_vlm"] = {
                "v": 1,
                "enabled": bool(dcfg.get("enabled", True)),
                "locate_min_side": int(dcfg.get("locate_min_side", 1536)),
                "locate_max_scale": float(dcfg.get("locate_max_scale", 2.5)),
                "locate_max_side": int(dcfg.get("locate_max_side", 2560)),
                "multi_scale": bool(dcfg.get("multi_scale", True)),
                "pass2_min_side": int(dcfg.get("pass2_min_side", 128)),
            }
            obb_cfg = (config.get("perception") or {}).get("dimension_obb") or {}
            cache_ent["_perception_dim_obb"] = {
                "v": 1,
                "enabled": bool(obb_cfg.get("enabled", True)),
                "ink_threshold": int(obb_cfg.get("ink_threshold", 245)),
                "pad": int(obb_cfg.get("pad", 2)),
                "warp_pad": int(obb_cfg.get("warp_pad", 4)),
            }
            acc_cfg = (config.get("perception") or {}).get("dimension_accuracy") or {}
            cache_ent["_perception_dim_accuracy"] = {
                "v": 1,
                "drop_weak": bool(acc_cfg.get("drop_weak", True)),
                "dedupe_same_text": bool(acc_cfg.get("dedupe_same_text", True)),
                "center_dist_thr": float(acc_cfg.get("center_dist_thr", 28.0)),
            }
        cached = cache.get(backend_name, img_sig, cache_ent)
        if cached is not None:
            all_instances.extend(cached)
            notes.append(f"cache_hit:{ent['entity_id']}")
            continue

        collected: list[dict[str, Any]] = []
        use_view = view_regions_enabled and is_dim
        use_tiles = tile_enabled and max(width, height) > tile_size
        tile_dir = resolve_path(perc_cfg.get("tile_dir", "work_dirs/cache/tiles")) / img_sig

        if use_view:
            from pipeline.perceive_common import collect_table_bboxes
            from pipeline.perceive_qwen_vl import locate_drawing_views
            from pipeline.view_regions import (
                assign_parent_by_components,
                resolve_part_region_specs,
                uncovered_grid_regions,
            )

            views, vnotes = locate_drawing_views(image, config, allow_mock_fallback=True)
            notes.extend(vnotes)
            table_bbs = collect_table_bboxes(all_instances)
            vr_tile = int(vr_cfg.get("tile_size", tile_size))
            vr_overlap = float(vr_cfg.get("tile_overlap", overlap))
            expand_ratio = float(vr_cfg.get("expand_ratio", 0.25))
            specs, src = resolve_part_region_specs(
                views,
                page_w=width,
                page_h=height,
                table_bboxes=table_bbs,
                expand_ratio=expand_ratio,
                min_side=int(vr_cfg.get("min_side", 64)),
                max_area_frac=float(vr_cfg.get("max_area_frac", 0.35)),
                tile_size=vr_tile,
                tile_overlap=vr_overlap,
                fallback_grid=bool(vr_cfg.get("fallback_grid", True)),
            )
            n_part_regions = sum(len(s.get("regions") or []) for s in specs)
            notes.append(f"view_regions:n={n_part_regions}")
            notes.append(f"view_regions_source={src}")
            if src in {"grid", "full_page"}:
                notes.append(f"view_regions_fallback={src}")

            component_insts: list[dict[str, Any]] = []
            covered_for_fill: list[list[int]] = []
            for si, spec in enumerate(specs):
                parent_id = spec.get("part_id")
                regions = list(spec.get("regions") or [])
                covered_for_fill.extend(regions)
                part_collected, t_acc = _run_perceive_on_regions(
                    image=image,
                    regions=regions,
                    ent=ent,
                    meta=meta,
                    config=config,
                    perceive_fn=perceive_fn,
                    tile_dir=tile_dir / "views" / f"part_{si:02d}",
                    notes=notes,
                    parent_id=str(parent_id) if parent_id else None,
                )
                collected.extend(part_collected)
                if t_acc is not None:
                    timing_acc = t_acc
                comp = _make_component_instance(spec)
                if comp is not None:
                    component_insts.append(comp)

            # 补扫部件区未覆盖的网格（恢复架构前「整页可检」能力）
            if bool(vr_cfg.get("fill_uncovered_grid", True)) and src == "view_regions":
                extra = uncovered_grid_regions(
                    covered_for_fill,
                    page_w=width,
                    page_h=height,
                    table_bboxes=table_bbs,
                    tile_size=vr_tile,
                    tile_overlap=vr_overlap,
                )
                notes.append(f"view_regions_uncovered:n={len(extra)}")
                if extra:
                    extra_collected, t_acc = _run_perceive_on_regions(
                        image=image,
                        regions=extra,
                        ent=ent,
                        meta=meta,
                        config=config,
                        perceive_fn=perceive_fn,
                        tile_dir=tile_dir / "views" / "uncovered",
                        notes=notes,
                        parent_id=None,
                    )
                    collected.extend(extra_collected)
                    if t_acc is not None:
                        timing_acc = t_acc
                    collected = assign_parent_by_components(collected, component_insts)

            notes.append(f"view_tiled:{ent['entity_id']}:{n_part_regions}")
            drop_os = bool(vr_cfg.get("drop_oversized_dims", False))
            use_obb = bool(
                ((config.get("perception") or {}).get("dimension_obb") or {}).get("enabled", True)
            )
            dim_fixed = _finalize_entity(
                ent,
                collected,
                cache_ent=cache_ent,
                drop_oversized=drop_os,
                prefer_smaller=True,
                use_quad=use_obb,
                apply_accuracy=True,
            )
            bundled = list(dim_fixed) + component_insts
            cache.set(backend_name, img_sig, cache_ent, bundled)
            all_instances.extend(bundled)
            continue

        if not use_tiles:
            payload = perceive_fn(image_path, [ent], meta, config, allow_mock_fallback=True)
            collected = list(payload.get("instances") or [])
            if payload.get("note"):
                notes.append(str(payload["note"]))
            for n in payload.get("notes") or []:
                notes.append(str(n))
            if isinstance(payload.get("timing"), dict):
                timing_acc = dict(payload["timing"])
        else:
            tiles = make_tiles(width, height, tile_size=tile_size, overlap=overlap)
            collected, t_acc = _run_perceive_on_regions(
                image=image,
                regions=[list(t) for t in tiles],
                ent=ent,
                meta=meta,
                config=config,
                perceive_fn=perceive_fn,
                tile_dir=tile_dir,
                notes=notes,
            )
            if t_acc is not None:
                timing_acc = t_acc
            notes.append(f"tiled:{ent['entity_id']}:{len(tiles)}")

        use_obb = is_dim and bool(
            ((config.get("perception") or {}).get("dimension_obb") or {}).get("enabled", True)
        )
        all_instances.extend(
            _finalize_entity(
                ent,
                collected,
                cache_ent=cache_ent,
                drop_oversized=False,
                prefer_smaller=is_dim,
                use_quad=use_obb,
                apply_accuracy=is_dim,
            )
        )

    out: dict[str, Any] = {
        "backend": backend_name,
        "instances": reindex_instances(all_instances),
        "notes": notes,
    }
    if timing_acc is not None:
        out["timing"] = timing_acc
    return out
