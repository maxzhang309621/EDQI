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


def nms_instances(instances: list[dict[str, Any]], iou_thr: float = 0.5) -> list[dict[str, Any]]:
    """同类 entity 按 confidence NMS。"""
    by_eid: dict[str, list[dict[str, Any]]] = {}
    for inst in instances:
        by_eid.setdefault(inst.get("entity_id", "unknown"), []).append(inst)
    kept: list[dict[str, Any]] = []
    for group in by_eid.values():
        group = sorted(group, key=lambda x: float(x.get("confidence") or 0), reverse=True)
        selected: list[dict[str, Any]] = []
        for cand in group:
            if all(iou_xyxy(cand["bbox"], s["bbox"]) < iou_thr for s in selected if s.get("bbox") and cand.get("bbox")):
                selected.append(cand)
        kept.extend(selected)
    return kept


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
        out.append(item)
    return out


def reindex_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    out = []
    for inst in instances:
        eid = inst.get("entity_id", "unknown")
        idx = counters.get(eid, 0)
        counters[eid] = idx + 1
        item = dict(inst)
        item["instance_id"] = f"{eid}#{idx}"
        out.append(item)
    return out


def _is_table_plan_entity(ent: dict[str, Any]) -> bool:
    if ent.get("parse_kind") == "table":
        return True
    eid = str(ent.get("entity_id") or "")
    return eid in {"info_table", "material_table", "main_table"} or eid.endswith("_table")


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

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    all_instances: list[dict[str, Any]] = []
    notes: list[str] = []
    timing_acc: dict[str, Any] | None = None

    table_plan = [e for e in plan if _is_table_plan_entity(e)]
    other_plan = [e for e in plan if not _is_table_plan_entity(e)]

    def _finalize_entity(ent: dict[str, Any], collected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        collected = nms_instances(collected, nms_thr)
        fixed = []
        for c in collected or []:
            item = dict(c)
            item["entity_id"] = ent["entity_id"]
            if not item.get("label"):
                item["label"] = ent.get("locate_query", ent["entity_id"])
            fixed.append(item)
        cache.set(backend_name, img_sig, ent, fixed)
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

    # ---- 其它实体：仍按单实体缓存/分块 ----
    for ent in other_plan:
        cached = cache.get(backend_name, img_sig, ent)
        if cached is not None:
            all_instances.extend(cached)
            notes.append(f"cache_hit:{ent['entity_id']}")
            continue

        use_tiles = (
            tile_enabled
            and max(width, height) > tile_size
            # 尺寸 number_mark 分块易出半页假框；表格已单独批次，此处禁 tile
            and str(ent.get("parse_kind") or "").lower() != "dimension_marks"
            and str(ent.get("entity_id") or "") not in {"number_mark", "annotation", "annotations"}
        )
        collected: list[dict[str, Any]] = []
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
            tile_dir = resolve_path(perc_cfg.get("tile_dir", "work_dirs/cache/tiles")) / img_sig
            tile_dir.mkdir(parents=True, exist_ok=True)
            for i, (x1, y1, x2, y2) in enumerate(tiles):
                crop = image.crop((x1, y1, x2, y2))
                tile_path = tile_dir / f"tile_{i:02d}.png"
                crop.save(tile_path)
                tile_meta = {
                    **meta,
                    "width": crop.size[0],
                    "height": crop.size[1],
                    "tile_origin": [x1, y1],
                }
                payload = perceive_fn(tile_path, [ent], tile_meta, config, allow_mock_fallback=True)
                shifted = shift_instances(payload.get("instances") or [], x1, y1)
                collected.extend(shifted)
                for n in payload.get("notes") or []:
                    notes.append(str(n))
            notes.append(f"tiled:{ent['entity_id']}:{len(tiles)}")

        all_instances.extend(_finalize_entity(ent, collected))

    out: dict[str, Any] = {
        "backend": backend_name,
        "instances": reindex_instances(all_instances),
        "notes": notes,
    }
    if timing_acc is not None:
        out["timing"] = timing_acc
    return out
