"""规则加载、实体计划收集、schema/算子校验。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pipeline import ROOT, load_json, resolve_path, validate_schema

ALLOWED_OPS = {
    "not_empty",
    "eq",
    "neq",
    "in",
    "regex",
    "count_gte",
    "count_lte",
    "ref_exists",
    "bbox_overlap_count_lte",
    "and",
    "or",
    "not",
}

# entity_id -> Facts 顶层键及是否多实例
ENTITY_FACTS_MAP: dict[str, dict[str, Any]] = {
    "title_block": {"key": "title_block", "many": False},
    "component": {"key": "components", "many": True},
    "components": {"key": "components", "many": True},
    "feature": {"key": "features", "many": True},
    "features": {"key": "features", "many": True},
    "annotation": {"key": "annotations", "many": True},
    "annotations": {"key": "annotations", "many": True},
    "number_mark": {"key": "annotations", "many": True},
    "datum": {"key": "datums", "many": True},
    "datums": {"key": "datums", "many": True},
    "info_table": {"key": "tables", "many": True},
    "material_table": {"key": "tables", "many": True},
    "main_table": {"key": "tables", "many": True},
    "aux_table": {"key": "tables", "many": True},
}


def load_rules(
    library_dir: str | Path,
    *,
    only_active: bool = True,
    rule_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    lib = resolve_path(library_dir)
    rules: list[dict[str, Any]] = []
    if not lib.exists():
        return rules
    for path in sorted(lib.glob("*.json")):
        rule = load_json(path)
        if only_active and rule.get("status") != "active":
            continue
        if rule_ids and rule.get("rule_id") not in rule_ids:
            continue
        rules.append(rule)
    return rules


def collect_entities(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 locate_query + entity_id 去重，合并 fields。"""
    merged: dict[str, dict[str, Any]] = {}
    for rule in rules:
        for ent in rule.get("entities", []):
            eid = ent["entity_id"]
            key = f"{eid}::{ent.get('locate_query', '')}"
            if key not in merged:
                merged[key] = {
                    "entity_id": eid,
                    "locate_query": ent.get("locate_query", eid),
                    "cardinality": ent.get("cardinality", "one"),
                    "fields": {},
                }
            for field in ent.get("fields", []):
                name = field["name"]
                prev = merged[key]["fields"].get(name, {})
                merged[key]["fields"][name] = {
                    "name": name,
                    "parse_hint": field.get("parse_hint") or prev.get("parse_hint") or name,
                    "required": bool(field.get("required") or prev.get("required")),
                }
    plan = []
    for item in merged.values():
        item["fields"] = list(item["fields"].values())
        plan.append(item)
    return plan


def _collect_paths_from_logic(logic: dict[str, Any], out: set[str]) -> None:
    op = logic.get("op")
    if op in {"and", "or"}:
        for arg in logic.get("args", []):
            _collect_paths_from_logic(arg, out)
        return
    if op == "not":
        arg = logic.get("arg")
        if arg:
            _collect_paths_from_logic(arg, out)
        return
    path = logic.get("path")
    if path:
        out.add(path)


def _entity_field_names(rule: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for ent in rule.get("entities", []):
        eid = ent["entity_id"]
        names.add(eid)
        for field in ent.get("fields", []):
            names.add(f"{eid}.{field['name']}")
            # 多实例列表字段常见写法 components.value
            mapping = ENTITY_FACTS_MAP.get(eid)
            if mapping and mapping["many"]:
                names.add(f"{mapping['key']}.{field['name']}")
            else:
                names.add(f"{mapping['key'] if mapping else eid}.{field['name']}")
    return names


def validate_rule(rule: dict[str, Any]) -> list[str]:
    errors = validate_schema(rule, "rule.schema.json")
    logic = rule.get("logic") or {}
    ops_errors = _validate_ops(logic)
    errors.extend(ops_errors)

    declared = _entity_field_names(rule)
    paths: set[str] = set()
    if isinstance(logic, dict):
        _collect_paths_from_logic(logic, paths)
    for path in paths:
        root = path.split(".")[0]
        # path 形如 title_block.part_no 或 components
        if path not in declared and root not in declared and path.split(".")[0] not in {
            m["key"] for m in ENTITY_FACTS_MAP.values()
        }:
            # 宽松：根键在 entity_id 或 facts map 中即可
            entity_ids = {e["entity_id"] for e in rule.get("entities", [])}
            facts_keys = {
                ENTITY_FACTS_MAP.get(eid, {"key": eid})["key"] for eid in entity_ids
            }
            if root not in entity_ids and root not in facts_keys:
                errors.append(f"logic.path '{path}' 未在 entities 中声明")
    return errors


def _validate_ops(logic: dict[str, Any], prefix: str = "logic") -> list[str]:
    if not isinstance(logic, dict):
        return [f"{prefix} 必须是对象"]
    op = logic.get("op")
    if op not in ALLOWED_OPS:
        return [f"{prefix}.op 非法: {op}（仅允许 {sorted(ALLOWED_OPS)}）"]
    errors: list[str] = []
    if op in {"and", "or"}:
        args = logic.get("args")
        if not isinstance(args, list) or not args:
            errors.append(f"{prefix}.args 必须为非空数组")
        else:
            for i, arg in enumerate(args):
                errors.extend(_validate_ops(arg, f"{prefix}.args[{i}]"))
    elif op == "not":
        arg = logic.get("arg")
        if not isinstance(arg, dict):
            errors.append(f"{prefix}.arg 必须为对象")
        else:
            errors.extend(_validate_ops(arg, f"{prefix}.arg"))
    elif op in {
        "not_empty",
        "eq",
        "neq",
        "in",
        "regex",
        "count_gte",
        "count_lte",
        "ref_exists",
        "bbox_overlap_count_lte",
    }:
        if not logic.get("path"):
            errors.append(f"{prefix}.path 必填")
        if op in {"eq", "neq", "in"} and "value" not in logic:
            errors.append(f"{prefix}.value 必填")
        if op == "regex" and not logic.get("pattern"):
            errors.append(f"{prefix}.pattern 必填")
        if op in {"count_gte", "count_lte", "bbox_overlap_count_lte"} and "count" not in logic:
            errors.append(f"{prefix}.count 必填")
    return errors


def path_root_entity(path: str) -> str:
    return path.split(".", 1)[0]
