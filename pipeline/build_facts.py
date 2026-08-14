"""S6：instances[] -> Facts 对象图。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from engines import ENTITY_FACTS_MAP
from pipeline import dump_json, validate_schema


def _normalize_pairs(pairs: Any) -> list[dict[str, Any]]:
    if not isinstance(pairs, list):
        return []
    out: list[dict[str, Any]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is None or str(name).strip() == "":
            continue
        content = item.get("content")
        out.append({"name": str(name).strip(), "content": content})
    return out


def build_facts(
    instances_payload: dict[str, Any],
    meta: dict[str, Any],
    *,
    drawing_id: str | None = None,
    dump_path: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    backend = instances_payload.get("backend", "unknown")
    instances = instances_payload.get("instances") or []

    meta_obj: dict[str, Any] = {
        "width": int(meta.get("width") or 1),
        "height": int(meta.get("height") or 1),
        "page": int(meta.get("page") or 1),
        "perception_backend": backend,
    }
    if meta.get("dpi") is not None:
        meta_obj["dpi"] = meta.get("dpi")
    if meta.get("scale_to_vlm") is not None:
        meta_obj["scale_to_vlm"] = meta.get("scale_to_vlm")
    elif "scale_to_vlm" in meta:
        pass
    else:
        meta_obj["scale_to_vlm"] = 1.0
    if meta.get("source_path") is not None:
        meta_obj["source_path"] = meta.get("source_path")

    facts: dict[str, Any] = {
        "drawing_id": drawing_id or meta.get("drawing_id") or "unknown",
        "meta": meta_obj,
        "title_block": None,
        "components": [],
        "features": [],
        "annotations": [],
        "datums": [],
        "tables": [],
        "_instances": deepcopy(instances),
    }

    list_keys = {"components", "features", "annotations", "datums", "tables"}
    counters: dict[str, int] = {}
    for inst in instances:
        eid = inst.get("entity_id", "unknown")
        mapping = ENTITY_FACTS_MAP.get(eid)
        if mapping is None and str(eid).startswith("aux_table"):
            mapping = {"key": "tables", "many": True}
        if mapping is None and (
            str(inst.get("parse_kind") or "").lower() == "table"
            or str(eid).endswith("_table")
        ):
            mapping = {"key": "tables", "many": True}
        if mapping is None:
            mapping = {"key": eid, "many": False}
        key = mapping["key"]
        many = mapping["many"] or inst.get("cardinality") == "many"

        fields = dict(inst.get("fields") or {})
        if key == "tables" and "pairs" in fields:
            fields["pairs"] = _normalize_pairs(fields.get("pairs"))

        obj = {
            **fields,
            "bbox": inst.get("bbox"),
            "raw_text": inst.get("raw_text"),
            "instance_id": inst.get("instance_id"),
            "label": inst.get("label"),
            "confidence": inst.get("confidence"),
            "needs_review": inst.get("needs_review", False),
        }
        if inst.get("keep_pair"):
            obj["keep_pair"] = True
        if inst.get("parent_id") is not None:
            obj["parent_id"] = inst.get("parent_id")
        if inst.get("bbox_expanded") is not None:
            obj["bbox_expanded"] = inst.get("bbox_expanded")
        if inst.get("quad") is not None:
            obj["quad"] = inst.get("quad")
        if inst.get("angle") is not None and "angle" not in obj:
            obj["angle"] = inst.get("angle")
        if "id" not in obj and fields.get("id"):
            obj["id"] = fields["id"]

        if many or key in list_keys:
            if key not in facts or not isinstance(facts[key], list):
                facts[key] = []
            facts[key].append(obj)
        else:
            # 单实例：后者覆盖前者（通常标题栏只有一个）
            facts[key] = obj

        counters[eid] = counters.get(eid, 0) + 1

    if validate:
        errors = validate_schema(facts, "facts.schema.json")
        if errors:
            raise ValueError("Facts schema 校验失败: " + "; ".join(errors))

    if dump_path:
        dump_json(facts, dump_path)
    return facts
