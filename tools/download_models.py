"""下载本地模型权重到 models/（需网络与 huggingface / modelscope）。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent

# 国内常用 HF 镜像（可用环境变量 HF_ENDPOINT 覆盖）
DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def download_hf(repo_id: str, local_dir: Path, *, endpoint: str | None = None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("请先 pip install huggingface_hub") from e

    if endpoint:
        # huggingface_hub 读取 HF_ENDPOINT；部分版本也认 HUGGINGFACE_HUB_ENDPOINT
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
        os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", endpoint.rstrip("/"))

    print(f"Downloading {repo_id} -> {local_dir}")
    if os.environ.get("HF_ENDPOINT"):
        print(f"  HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    except Exception as e:
        msg = str(e)
        hint = (
            "\n下载失败。可尝试：\n"
            "  1) 国内镜像: python tools/download_models.py qwen3_vl --mirror\n"
            "  2) ModelScope: pip install modelscope && "
            "python tools/download_models.py qwen3_vl --source modelscope\n"
            "  3) 手动: huggingface-cli download Qwen/Qwen3-VL-4B-Instruct "
            "--local-dir models/Qwen3-VL-4B-Instruct\n"
            "     （先设 $env:HF_ENDPOINT='https://hf-mirror.com'）\n"
        )
        if "CERTIFICATE" in msg.upper() or "SSL" in msg.upper() or "ConnectError" in msg:
            raise SystemExit(f"{e}\n{hint}") from e
        raise


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
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=f"使用 HF 镜像（默认 {DEFAULT_HF_MIRROR}，也可用环境变量 HF_ENDPOINT）",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="自定义 HF API 地址，例如 https://hf-mirror.com",
    )
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

    endpoint = args.endpoint
    if args.mirror and not endpoint:
        endpoint = os.environ.get("HF_ENDPOINT") or DEFAULT_HF_MIRROR

    for key in targets:
        m = mapping[key]
        local = ROOT / m["path"]
        repo = m["hf_id"]
        if args.source == "hf":
            download_hf(repo, local, endpoint=endpoint)
        else:
            # ModelScope 上 id 可能同名；LocateAnything 以 HF 为主
            download_ms(repo, local)
        print(f"done: {key}")


if __name__ == "__main__":
    main()
