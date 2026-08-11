"""检查本地模型权重是否就绪。"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from pipeline import load_config, model_download_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = {}
    for key, mcfg in (cfg.get("models") or {}).items():
        if not isinstance(mcfg, dict) or "path" not in mcfg:
            continue
        out[key] = model_download_status(mcfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
