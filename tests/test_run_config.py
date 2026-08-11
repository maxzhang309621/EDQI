"""run.yaml 运行参数合并。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run import load_run_config, resolve_run_options


def test_load_run_config_defaults():
    cfg = load_run_config(ROOT / "configs" / "run.yaml")
    assert cfg.get("rule_set_id") == "default"
    assert cfg.get("page") == 1
    assert cfg.get("drawing_parse") is True
    assert cfg.get("enqueue_review") is True
    assert cfg.get("tables_only") is None


def test_resolve_cli_overrides_yaml():
    run_cfg = {
        "drawing": "data/input/from_yaml.pdf",
        "backend": "mock",
        "rule_set_id": "yaml_set",
        "page": 2,
        "enqueue_review": True,
        "drawing_parse": True,
        "config": "configs/default.yaml",
    }
    opts = resolve_run_options(
        drawing="data/input/cli.pdf",
        run_config=run_cfg,
        perception_backend="qwen_vl",
        page=3,
        enqueue_review=False,
        drawing_parse=False,
        tables_only=True,
    )
    assert opts["drawing"] == "data/input/cli.pdf"
    assert opts["perception_backend"] == "qwen_vl"
    assert opts["page"] == 3
    assert opts["enqueue_review"] is False
    assert opts["drawing_parse"] is False
    assert opts["tables_only"] is True
    assert opts["rule_set_id"] == "yaml_set"


def test_resolve_requires_drawing():
    with pytest.raises(ValueError, match="图纸路径"):
        resolve_run_options(run_config={"backend": "mock"})
