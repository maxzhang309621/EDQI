"""从用户配置构建图纸信息解析感知计划（与质检规则实体合并）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pipeline import ROOT, resolve_path

# 走数字重叠 OCR 路径的实体（与 VL 配置实体拆分）
OCR_OVERLAP_ENTITY_IDS = frozenset({"number_mark", "annotation", "annotations"})


def load_drawing_parse_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    rel = cfg.get("drawing_parse_config") or "configs/drawing_parse.yaml"
    path = resolve_path(rel)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def is_tables_only(config: dict[str, Any] | None = None) -> bool:
    """是否仅表格解析（跳过数字重叠 OCR 合并）。

    default.yaml 的 drawing_parse.tables_only 与 drawing_parse.yaml 顶层 tables_only
    任一为 true 即开启。
    """
    cfg = config or {}
    if bool((cfg.get("drawing_parse") or {}).get("tables_only", False)):
        return True
    data = load_drawing_parse_config(cfg)
    return bool(data.get("tables_only", False))


def _field_list(fields: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in fields or []:
        if not isinstance(field, dict) or not field.get("name"):
            continue
        locate_by = str(field.get("locate_by") or "label").strip().lower()
        if locate_by not in {"label", "size_position"}:
            locate_by = "label"
        item = {
            "name": str(field["name"]),
            "parse_hint": str(field.get("parse_hint") or field["name"]),
            "required": bool(field.get("required", False)),
            "locate_by": locate_by,
        }
        out.append(item)
    return out


def _normalize_table_block(block: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    if not bool(block.get("enabled", True)):
        return None
    read_mode = str(block.get("read_mode") or "cell_content").strip().lower()
    if read_mode not in {"cell_content", "above_cells"}:
        read_mode = "cell_content"
    value_as_array = bool(block.get("value_as_array", read_mode == "above_cells"))
    ocr_enhance = block.get("ocr_enhance")
    if ocr_enhance is not None and not isinstance(ocr_enhance, dict):
        ocr_enhance = None
    ent = {
        "entity_id": str(block.get("entity_id") or "info_table"),
        "locate_query": str(
            block.get("locate_query") or "图纸中的表格（参数表/明细表/数据表等）"
        ),
        "cardinality": str(block.get("cardinality") or "many"),
        "fields": _field_list(block.get("fields")),
        "parse_kind": "table",
        "section": str(block.get("section") or block.get("entity_id") or "table"),
        "read_mode": read_mode,
        "value_as_array": value_as_array,
        "extract_all_pairs": bool(block.get("extract_all_pairs", False)),
        "crop_expand": float(block.get("crop_expand", 0.25)),
        "max_new_tokens": int(block.get("max_new_tokens", 4096)),
    }
    if ocr_enhance:
        ent["ocr_enhance"] = dict(ocr_enhance)
    return ent


def get_dimension_marks_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """从 drawing_parse.yaml 的 objects 中取 parse_kind=dimension_marks 项。

    兼容旧版顶层 dimension_marks 段。
    """
    data = load_drawing_parse_config(config)
    if not isinstance(data, dict):
        return {"enabled": False}
    for obj in data.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("parse_kind") or "").lower() == "dimension_marks":
            return obj
    legacy = data.get("dimension_marks")
    if isinstance(legacy, dict):
        return legacy
    return {"enabled": False}


def is_dimension_marks_enabled(config: dict[str, Any] | None = None) -> bool:
    return bool(get_dimension_marks_config(config).get("enabled", False))


_DIM_BACKEND_ALIASES = {
    "ocr": "ocr",
    "vlm": "vlm",
    "ocr_locate_vlm_filter": "ocr_locate_vlm_filter",
    "ocr_vlm": "ocr_locate_vlm_filter",
    "ocr+vlm": "ocr_locate_vlm_filter",
    "hybrid": "ocr_locate_vlm_filter",
}


def dimension_marks_backend(config: dict[str, Any] | None = None) -> str:
    """尺寸属性后端：ocr_locate_vlm_filter（默认）| vlm | ocr。

    - ocr：OCR 定位 + 规则解析（无 VLM）
    - vlm：VLM 全图定位 + 字段（OCR 仅保留 keep_pair）
    - ocr_locate_vlm_filter：OCR 宽召回+规则初筛，再用 VLM crop 精筛
    """
    raw = str(get_dimension_marks_config(config).get("backend") or "ocr_locate_vlm_filter").strip().lower()
    return _DIM_BACKEND_ALIASES.get(raw, "ocr_locate_vlm_filter")


def is_dimension_marks_vlm(config: dict[str, Any] | None = None) -> bool:
    return is_dimension_marks_enabled(config) and dimension_marks_backend(config) == "vlm"


def is_dimension_marks_ocr_locate_vlm(config: dict[str, Any] | None = None) -> bool:
    return (
        is_dimension_marks_enabled(config)
        and dimension_marks_backend(config) == "ocr_locate_vlm_filter"
    )


def dimension_marks_uses_ocr_parse(config: dict[str, Any] | None = None) -> bool:
    """是否在 OCR 路径做尺寸文本解析/初筛。"""
    if not is_dimension_marks_enabled(config):
        return False
    return dimension_marks_backend(config) in {"ocr", "ocr_locate_vlm_filter"}


def _normalize_dimension_marks_block(block: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(block, dict) or not bool(block.get("enabled", False)):
        return None
    fields = _field_list(block.get("fields"))
    if not fields:
        fields = _field_list(
            [
                {"name": "text", "parse_hint": "尺寸标注可见原文"},
                {"name": "angle", "parse_hint": "文本相对水平线朝向角（度）"},
                {"name": "dim_kind", "parse_hint": "diameter|radius|length|angle"},
                {"name": "basic_size", "parse_hint": "基本尺寸"},
                {"name": "tolerance", "parse_hint": "公差"},
                {"name": "has_tolerance", "parse_hint": "是否标明公差"},
            ]
        )
    backend_raw = str(block.get("backend") or "ocr_locate_vlm_filter").strip().lower()
    backend = _DIM_BACKEND_ALIASES.get(backend_raw, "ocr_locate_vlm_filter")
    return {
        "entity_id": str(block.get("entity_id") or "number_mark"),
        "locate_query": str(
            block.get("locate_query")
            or "图中尺寸数字或数值标注（Ø直径 / R半径 / 长度 / °角度）；排除标题栏/图号/材料牌号"
        ),
        "cardinality": str(block.get("cardinality") or "many"),
        "fields": fields,
        "parse_kind": "dimension_marks",
        "backend": backend,
        "detect_overlap": bool(block.get("detect_overlap", False)),
        # 尺寸属性专用 OCR 增强（ocr / ocr_locate_vlm_filter；与 perception.number_overlap 独立）
        "ocr_angle_adapt": bool(block.get("ocr_angle_adapt", False)),
        "ocr_angle_adapt_step": float(block.get("ocr_angle_adapt_step", 15)),
        "ocr_angle_adapt_min_count": int(block.get("ocr_angle_adapt_min_count", 2)),
        "ocr_angle_adapt_max_extra": int(block.get("ocr_angle_adapt_max_extra", 4)),
        "ocr_deskew_reread": bool(block.get("ocr_deskew_reread", False)),
        "ocr_deskew_min_angle": float(block.get("ocr_deskew_min_angle", 8.0)),
        "ocr_angles": [float(a) for a in (block.get("ocr_angles") or [])],
        "ocr_accept_kinds": [
            str(k).strip().lower()
            for k in (block.get("ocr_accept_kinds") or ["radius", "length"])
            if str(k).strip()
        ],
        "vlm_require_kinds": [
            str(k).strip().lower()
            for k in (block.get("vlm_require_kinds") or ["diameter", "angle"])
            if str(k).strip()
        ],
        # 默认排除表格框内 OCR 数字（尺寸属性仅表格外）
        "exclude_table_regions": bool(block.get("exclude_table_regions", True)),
        "exclude_table_pad": float(block.get("exclude_table_pad", 2.0)),
        "exclude_table_expand_up": float(block.get("exclude_table_expand_up", 0.12)),
        # 严格模式：只保留配置字段 + 合法尺寸；过滤畸形框/杂讯
        "strict_fields_only": bool(block.get("strict_fields_only", True)),
        "require_dim_kind": bool(block.get("require_dim_kind", True)),
        "require_basic_size": bool(block.get("require_basic_size", True)),
        "max_bbox_width_ratio": float(block.get("max_bbox_width_ratio", 0.22)),
        "max_bbox_height_ratio": float(block.get("max_bbox_height_ratio", 0.12)),
        "max_bbox_area_ratio": float(block.get("max_bbox_area_ratio", 0.035)),
        "max_aspect_ratio": float(block.get("max_aspect_ratio", 8.0)),
        # ocr_locate_vlm_filter：送入 VLM 精筛的最大候选数
        "vlm_filter_max_candidates": int(block.get("vlm_filter_max_candidates", 48)),
        "vlm_filter_crop_expand": float(block.get("vlm_filter_crop_expand", 0.22)),
    }


def build_drawing_parse_plan(
    config: dict[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """将 drawing_parse.yaml 转为感知 plan 实体列表。"""
    if not enabled:
        return []
    cfg = config or {}
    if cfg.get("drawing_parse", {}).get("enabled") is False:
        return []

    data = load_drawing_parse_config(cfg)
    if not data:
        return []

    plan: list[dict[str, Any]] = []

    for obj in data.get("objects") or []:
        if not isinstance(obj, dict) or not bool(obj.get("enabled", True)):
            continue
        if str(obj.get("parse_kind") or "").lower() == "dimension_marks":
            ent = _normalize_dimension_marks_block(obj)
            if ent:
                plan.append(ent)
            continue
        eid = obj.get("entity_id")
        if not eid:
            continue
        plan.append(
            {
                "entity_id": str(eid),
                "locate_query": str(obj.get("locate_query") or eid),
                "cardinality": str(obj.get("cardinality") or "one"),
                "fields": _field_list(obj.get("fields")),
                "parse_kind": str(obj.get("parse_kind") or "object"),
                "crop_expand": float(obj["crop_expand"]) if obj.get("crop_expand") is not None else None,
                "max_new_tokens": int(obj["max_new_tokens"]) if obj.get("max_new_tokens") is not None else None,
            }
        )

    tables = data.get("tables")
    table_blocks: list[dict[str, Any]] = []
    if isinstance(tables, list):
        table_blocks = [t for t in tables if isinstance(t, dict)]
    elif isinstance(tables, dict):
        # 兼容旧版单表；若含 sections 列表则展开
        sections = tables.get("sections")
        if isinstance(sections, list) and sections:
            base = {k: v for k, v in tables.items() if k != "sections"}
            for sec in sections:
                if isinstance(sec, dict):
                    table_blocks.append({**base, **sec})
        else:
            table_blocks = [tables]

    for block in table_blocks:
        ent = _normalize_table_block(block)
        if ent:
            plan.append(ent)

    # 兼容旧版顶层 dimension_marks（若 objects 里尚未声明）
    if not any(e.get("parse_kind") == "dimension_marks" for e in plan):
        legacy = data.get("dimension_marks")
        dim_ent = _normalize_dimension_marks_block(legacy if isinstance(legacy, dict) else {})
        if dim_ent:
            plan.append(dim_ent)

    return plan


def merge_perception_plans(*plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 entity_id::locate_query 合并多路 plan，字段去重合并。"""
    merged: dict[str, dict[str, Any]] = {}
    for plan in plans:
        for ent in plan or []:
            eid = ent.get("entity_id")
            if not eid:
                continue
            key = f"{eid}::{ent.get('locate_query', '')}"
            if key not in merged:
                merged[key] = {
                    **{k: v for k, v in ent.items() if k != "fields"},
                    "fields": {},
                }
            dest = merged[key]
            for k, v in ent.items():
                if k == "fields":
                    continue
                if v is not None:
                    dest[k] = v
            for field in ent.get("fields") or []:
                name = field["name"]
                prev = dest["fields"].get(name, {})
                dest["fields"][name] = {
                    "name": name,
                    "parse_hint": field.get("parse_hint") or prev.get("parse_hint") or name,
                    "required": bool(field.get("required") or prev.get("required")),
                    "locate_by": field.get("locate_by") or prev.get("locate_by") or "label",
                }

    out: list[dict[str, Any]] = []
    for item in merged.values():
        item["fields"] = list(item["fields"].values())
        out.append(item)
    return out


def split_plan_for_backends(
    plan: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回 (ocr_overlap_plan, vl_plan)。

    dimension_marks:
      - vlm → 仅 VL 全图定位
      - ocr / ocr_locate_vlm_filter → OCR 定位（后者再经 VLM crop 精筛，不进 VL 全图 locate）
    """
    ocr: list[dict[str, Any]] = []
    vl: list[dict[str, Any]] = []
    for ent in plan:
        parse_kind = str(ent.get("parse_kind") or "").lower()
        backend = str(ent.get("backend") or "ocr_locate_vlm_filter").strip().lower()
        backend = _DIM_BACKEND_ALIASES.get(backend, backend)
        if parse_kind == "dimension_marks":
            if backend == "vlm":
                vl.append(ent)
            else:
                ocr.append(ent)
            continue
        if ent.get("entity_id") in OCR_OVERLAP_ENTITY_IDS:
            ocr.append(ent)
        else:
            vl.append(ent)
    return ocr, vl


def default_drawing_parse_path() -> Path:
    return ROOT / "configs" / "drawing_parse.yaml"
