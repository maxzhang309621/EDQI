"""常驻推理进程：VLM/OCR 加载一次，多图循环跑质检（P0）。

用法:
  python -m pipeline.worker
  python -m pipeline.worker data/input/test.pdf
  python -m pipeline.worker a.pdf b.png --tables-only
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from pipeline import load_config, model_path_ready
from pipeline.ocr_engine import warmup_rapid_ocr
from pipeline.perceive_qwen_vl import warmup_qwen_vl
from pipeline.run import load_run_config, resolve_run_options, run


def warmup_perception(
    *,
    config_path: str | Path | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """预热 VLM + RapidOCR；打印耗时。"""
    cfg = load_config(config_path)
    be = backend or cfg.get("perception_backend", "qwen_vl")
    out: dict[str, Any] = {"backend": be, "ocr": False, "vlm": None}

    t0 = time.perf_counter()
    out["ocr"] = warmup_rapid_ocr()
    print(f"[warmup] rapid_ocr={'ok' if out['ocr'] else 'skip'} ({time.perf_counter() - t0:.2f}s)", flush=True)

    if be in {"qwen_vl", "hybrid"}:
        model_cfg = cfg.get("models", {}).get("qwen3_vl", {})
        if model_path_ready(model_cfg):
            t1 = time.perf_counter()
            info = warmup_qwen_vl(cfg)
            out["vlm"] = info
            print(
                f"[warmup] qwen_vl ok already={info.get('already_loaded')} "
                f"({time.perf_counter() - t1:.2f}s)",
                flush=True,
            )
        else:
            out["vlm"] = {"ok": False, "reason": "weights_not_ready"}
            print(f"[warmup] qwen_vl skip: weights not ready at {model_cfg.get('path')}", flush=True)
    else:
        print(f"[warmup] qwen_vl skip: backend={be}", flush=True)

    print("[warmup] ready — 后续同进程内 run 将复用已加载模型", flush=True)
    return out


def _print_result_brief(drawing: str | Path, result: dict[str, Any]) -> None:
    notes = list(result.get("perception_notes") or [])
    hits = [n for n in notes if str(n).startswith("cache_hit:")]
    print(
        f"[done] {drawing} passed={result.get('passed')} "
        f"findings={len(result.get('findings') or [])} "
        f"cache_hits={len(hits)} timing={result.get('timing')}",
        flush=True,
    )
    if hits:
        print(f"[cache] {', '.join(hits)}", flush=True)
    for f in result.get("findings") or []:
        print(
            f" - [{f['severity']}] {f['rule_id']}: {f['message']} (actual={f.get('actual')})",
            flush=True,
        )


def _run_one(drawing: str | Path, opts: dict[str, Any]) -> dict[str, Any]:
    result = run(
        drawing,
        rule_set_id=opts["rule_set_id"],
        perception_backend=opts["perception_backend"],
        config_path=opts["config_path"],
        page=opts["page"],
        enqueue_review=opts["enqueue_review"],
        rule_ids=opts["rule_ids"],
        drawing_parse=opts["drawing_parse"],
        tables_only=opts["tables_only"],
    )
    _print_result_brief(drawing, result)
    return result


def _interactive_loop(opts: dict[str, Any]) -> None:
    print(
        "进入交互模式：输入图纸路径后回车；空行或 quit/exit 退出。",
        flush=True,
    )
    while True:
        try:
            line = input("drawing> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            break
        if not line or line.lower() in {"q", "quit", "exit"}:
            break
        path = Path(line.strip("\"'"))
        if not path.exists():
            print(f"[error] 文件不存在: {path}", flush=True)
            continue
        try:
            _run_one(path, opts)
        except Exception as exc:
            print(f"[error] {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDQI 常驻 worker：预热模型后批跑/交互跑图纸（避免每次重载权重）"
    )
    parser.add_argument(
        "drawings",
        nargs="*",
        default=None,
        help="图纸路径列表；省略则用 run.yaml 的 drawing，或进入交互",
    )
    parser.add_argument("--run-config", default="configs/run.yaml")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--rule-set-id", default=None)
    parser.add_argument("--rule-id", action="append", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--no-review-queue", action="store_true")
    parser.add_argument("--no-drawing-parse", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过显式预热（仍会在首次推理时加载）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="强制交互模式（即使提供了 drawings）",
    )
    args = parser.parse_args()

    run_cfg = load_run_config(args.run_config)
    drawings: list[Path] = [Path(p) for p in (args.drawings or [])]
    if not drawings and run_cfg.get("drawing"):
        drawings = [Path(run_cfg["drawing"])]

    opts = resolve_run_options(
        drawing=str(drawings[0]) if drawings else None,
        run_config=run_cfg,
        perception_backend=args.backend,
        rule_set_id=args.rule_set_id,
        rule_ids=args.rule_id,
        config_path=args.config,
        page=args.page,
        enqueue_review=False if args.no_review_queue else None,
        drawing_parse=False if args.no_drawing_parse else None,
        tables_only=True if args.tables_only else None,
        allow_missing_drawing=True,
    )

    if not args.no_warmup:
        warmup_perception(
            config_path=opts["config_path"],
            backend=opts["perception_backend"],
        )

    if drawings and not args.interactive:
        for d in drawings:
            if not d.exists():
                print(f"[error] 文件不存在: {d}", flush=True)
                continue
            _run_one(d, opts)
        return

    _interactive_loop(opts)


if __name__ == "__main__":
    main()
