"""兼容旧入口：请改用 python -m pipeline.run / from pipeline.run import run。"""
from __future__ import annotations

from pipeline.run import (  # noqa: F401
    load_run_config,
    load_run_config as load_run_check_config,
    main,
    perceive,
    resolve_run_options,
    resolve_run_options as resolve_run_check_options,
    run,
    run as run_check,
)

if __name__ == "__main__":
    main()
