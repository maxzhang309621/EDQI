"""S2：规则 schema / path / 算子校验。"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from engines import validate_rule
from pipeline import load_json, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="校验规则 JSON")
    parser.add_argument("rule_json", help="规则文件路径")
    args = parser.parse_args()
    path = resolve_path(args.rule_json)
    rule = load_json(path)
    errors = validate_rule(rule)
    if errors:
        print(f"INVALID {path}")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"OK {path} rule_id={rule.get('rule_id')} status={rule.get('status')}")


if __name__ == "__main__":
    main()
