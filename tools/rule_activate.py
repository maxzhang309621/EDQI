"""S2：draft → active 入库（人工确认后执行）。"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from engines import validate_rule
from pipeline import dump_json, load_config, load_json, resolve_path


def activate(
    draft_path: str | Path,
    *,
    config_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    cfg = load_config(config_path)
    src = resolve_path(draft_path)
    rule = load_json(src)
    errors = validate_rule(rule)
    if errors and not force:
        raise ValueError("规则校验失败:\n- " + "\n- ".join(errors))

    if rule.get("status") not in {"draft", "active"} and not force:
        raise ValueError(f"当前 status={rule.get('status')}，请确认后再激活")

    rule["status"] = "active"
    rule["version"] = int(rule.get("version") or 1)
    rule["updated_at"] = datetime.now(timezone.utc).isoformat()
    # 清理内部备注字段
    for k in list(rule.keys()):
        if k.startswith("_"):
            rule.pop(k)

    rid = rule["rule_id"]
    dst = resolve_path(cfg["rules"]["library_dir"]) / f"{rid}.json"
    if dst.exists() and not force:
        # 版本递增
        old = load_json(dst)
        rule["version"] = int(old.get("version") or 1) + 1

    dump_json(rule, dst)
    # 备份草稿
    bak_dir = resolve_path(cfg["rules"]["drafts_dir"]) / "_activated"
    bak_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, bak_dir / src.name)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="激活规则入库")
    parser.add_argument("draft_json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dst = activate(args.draft_json, config_path=args.config, force=args.force)
    print(f"activated -> {dst}")


if __name__ == "__main__":
    main()
