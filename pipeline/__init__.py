"""项目通用工具：路径、配置加载、schema 校验。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or ROOT) / p


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_path(config_path or "configs/default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(ROOT)
    return cfg


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    return path


def load_schema(name: str) -> dict[str, Any]:
    schema_path = ROOT / "rules" / "schema" / name
    return load_json(schema_path)


def validate_schema(instance: Any, schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    return sorted(e.message for e in validator.iter_errors(instance))


def model_weights_ready(local: Path) -> bool:
    """真正可加载：有 config，且有完整权重文件（非 .incomplete）。"""
    if not local.is_dir() or not (local / "config.json").exists():
        return False
    weights = list(local.glob("*.safetensors")) + list(local.glob("*.bin"))
    weights = [p for p in weights if p.is_file() and p.stat().st_size > 0]
    if not weights:
        return False
    index = local / "model.safetensors.index.json"
    if index.exists():
        try:
            meta = load_json(index)
            shards = set((meta.get("weight_map") or {}).values())
            missing = [s for s in shards if not (local / s).exists()]
            if missing:
                return False
        except Exception:
            return False
    cache_dl = local / ".cache" / "huggingface" / "download"
    if cache_dl.exists() and any(cache_dl.rglob("*.incomplete")):
        return False
    return True


def model_path_ready(model_cfg: dict[str, Any]) -> bool:
    """本地权重是否真正可加载。"""
    local = resolve_path(model_cfg.get("path", ""))
    return model_weights_ready(local)


def model_download_status(model_cfg: dict[str, Any]) -> dict[str, Any]:
    local = resolve_path(model_cfg.get("path", ""))
    has_config = (local / "config.json").exists()
    shards = list(local.glob("*.safetensors")) + list(local.glob("*.bin"))
    incomplete: list[str] = []
    cache_dl = local / ".cache" / "huggingface" / "download"
    if cache_dl.exists():
        incomplete = [str(p.name) for p in cache_dl.rglob("*.incomplete")]
    ready = model_path_ready(model_cfg)
    return {
        "path": str(local),
        "has_config": has_config,
        "weight_files": len(shards),
        "incomplete": incomplete[:10],
        "ready": ready,
        "hint": None
        if ready
        else f"权重未就绪，请继续下载 {model_cfg.get('hf_id')} 到 {local}",
    }


def resolve_model_source(model_cfg: dict[str, Any]) -> str:
    """优先本地 path，否则返回 hf_id（用于后续下载/加载）。"""
    if model_path_ready(model_cfg):
        return str(resolve_path(model_cfg["path"]))
    return str(model_cfg.get("hf_id") or model_cfg.get("path"))
