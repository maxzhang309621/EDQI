"""Findings 人工复核队列（S8 之后可选业务复核）。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline import dump_json, load_config, load_json, resolve_path


def _queue_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg.get("review", {}).get("queue_dir", "work_dirs/review_queue"))


def enqueue_from_result(
    result: dict[str, Any],
    facts: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> Path | None:
    """将 error findings 与 needs_review 实例写入待复核队列。"""
    cfg = config or load_config()
    findings = result.get("findings") or []
    needs_instances = []
    if facts:
        for key in ("title_block",):
            node = facts.get(key)
            if isinstance(node, dict) and node.get("needs_review"):
                needs_instances.append(node)
        for key in ("components", "features", "annotations", "datums"):
            for item in facts.get(key) or []:
                if isinstance(item, dict) and item.get("needs_review"):
                    needs_instances.append(item)

    error_findings = [f for f in findings if f.get("severity") == "error"]
    if not error_findings and not needs_instances and result.get("passed", True):
        # 无事可复核
        if not cfg.get("review", {}).get("enqueue_warn", False):
            warn_findings = [f for f in findings if f.get("severity") == "warn"]
            if not warn_findings:
                return None
            error_findings = warn_findings

    drawing_id = result.get("drawing_id") or (facts or {}).get("drawing_id") or "unknown"
    item = {
        "drawing_id": drawing_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "passed": result.get("passed"),
        "perception_backend": result.get("perception_backend"),
        "findings": findings,
        "needs_review_instances": needs_instances,
        "artifacts": result.get("artifacts"),
        "reviewer": None,
        "decision": None,  # approve | reject | revise
        "comment": None,
    }
    qdir = _queue_dir(cfg)
    path = qdir / f"{drawing_id}_{item['created_at'].replace(':', '').replace('+', 'p')}.json"
    dump_json(item, path)
    # 同步最新指针
    dump_json(item, qdir / f"{drawing_id}_latest.json")
    return path


def list_pending(config: dict[str, Any] | None = None) -> list[Path]:
    cfg = config or load_config()
    qdir = _queue_dir(cfg)
    if not qdir.exists():
        return []
    out = []
    for p in sorted(qdir.glob("*.json")):
        if p.name.endswith("_latest.json"):
            continue
        data = load_json(p)
        if data.get("status") == "pending":
            out.append(p)
    return out


def resolve_item(
    path: str | Path,
    *,
    decision: str,
    reviewer: str = "human",
    comment: str = "",
) -> Path:
    p = resolve_path(path)
    data = load_json(p)
    if decision not in {"approve", "reject", "revise"}:
        raise ValueError("decision 必须是 approve|reject|revise")
    data["status"] = "resolved"
    data["decision"] = decision
    data["reviewer"] = reviewer
    data["comment"] = comment
    data["resolved_at"] = datetime.now(timezone.utc).isoformat()
    dump_json(data, p)
    latest = p.parent / f"{data.get('drawing_id')}_latest.json"
    dump_json(data, latest)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Findings 复核队列")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 pending")
    p_list.add_argument("--config", default="configs/default.yaml")

    p_res = sub.add_parser("resolve", help="处理一条队列")
    p_res.add_argument("path")
    p_res.add_argument("--decision", required=True, choices=["approve", "reject", "revise"])
    p_res.add_argument("--reviewer", default="human")
    p_res.add_argument("--comment", default="")
    p_res.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()
    if args.cmd == "list":
        cfg = load_config(args.config)
        items = list_pending(cfg)
        print(f"pending={len(items)}")
        for p in items:
            data = load_json(p)
            print(f" - {p.name} drawing={data.get('drawing_id')} findings={len(data.get('findings') or [])}")
    elif args.cmd == "resolve":
        path = resolve_item(args.path, decision=args.decision, reviewer=args.reviewer, comment=args.comment)
        print(f"resolved {path}")


if __name__ == "__main__":
    main()
