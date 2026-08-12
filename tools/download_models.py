"""下载本地模型权重到 models/（需网络与 huggingface / modelscope）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def download_hf(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("请先 pip install huggingface_hub") from e
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def download_ms(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError("请先 pip install modelscope") from e
    print(f"Downloading (ModelScope) {repo_id} -> {local_dir}")
    snapshot_download(repo_id, local_dir=str(local_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 EDQI 本地模型权重")
    parser.add_argument(
        "which",
        choices=["qwen3_text", "qwen3_vl", "locateanything", "all"],
        help="要下载的模型",
    )
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mapping = {
        "qwen3_text": cfg["models"]["qwen3_text"],
        "qwen3_vl": cfg["models"]["qwen3_vl"],
        "locateanything": cfg["models"]["locateanything"],
    }
    targets = list(mapping.keys()) if args.which == "all" else [args.which]
    for key in targets:
        m = mapping[key]
        local = ROOT / m["path"]
        repo = m["hf_id"]
        if args.source == "hf":
            download_hf(repo, local)
        else:
            # ModelScope 上 id 可能同名；LocateAnything 以 HF 为主
            download_ms(repo, local)
        print(f"done: {key}")


if __name__ == "__main__":
    main()
