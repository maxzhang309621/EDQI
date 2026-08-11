"""批跑图纸目录。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from pipeline import dump_json, load_config, resolve_path
from pipeline.run import run

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def main() -> None:
    parser = argparse.ArgumentParser(description="批量质检")
    parser.add_argument("input_dir", help="图纸目录")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rule-set-id", default="default")
    parser.add_argument("--glob", default="*", help="例如 *.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(args.input_dir)
    files = sorted(
        p for p in root.glob(args.glob) if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    # 同进程批跑：先预热，避免每张图重复加载权重
    if args.backend in {"qwen_vl", "hybrid"}:
        from pipeline.worker import warmup_perception

        warmup_perception(config_path=args.config, backend=args.backend)
    rows = []
    for path in files:
        try:
            result = run(
                path,
                perception_backend=args.backend,
                config_path=args.config,
                rule_set_id=args.rule_set_id,
            )
            rows.append(
                {
                    "drawing": str(path),
                    "drawing_id": result.get("drawing_id"),
                    "passed": result.get("passed"),
                    "n_findings": len(result.get("findings") or []),
                    "backend": result.get("perception_backend"),
                    "vis": (result.get("artifacts") or {}).get("vis"),
                    "error": "",
                }
            )
            print(f"OK {path.name} passed={result['passed']} findings={len(result['findings'])}")
        except Exception as e:
            rows.append(
                {
                    "drawing": str(path),
                    "drawing_id": path.stem,
                    "passed": False,
                    "n_findings": 0,
                    "backend": args.backend,
                    "vis": "",
                    "error": str(e),
                }
            )
            print(f"FAIL {path.name}: {e}")

    out_json = resolve_path(cfg["work_dirs"]["reports"]) / "batch_summary.json"
    out_csv = resolve_path(cfg["work_dirs"]["reports"]) / "batch_summary.csv"
    dump_json({"backend": args.backend, "n": len(rows), "items": rows}, out_json)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["drawing", "drawing_id", "passed", "n_findings", "backend", "vis", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print("summary:", out_json)
    print("csv:", out_csv)


if __name__ == "__main__":
    main()
