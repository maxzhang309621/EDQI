"""S9：金标评测 — 框 Recall@IoU、字段 Exact Match、规则级 P/R。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from pipeline import dump_json, load_config, load_json, resolve_path
from pipeline.perceive_utils import iou_xyxy
from pipeline.run import run


def match_boxes(
    preds: list[list[float]],
    gts: list[list[float]],
    iou_thr: float = 0.5,
) -> tuple[int, int, int]:
    """返回 tp, fp, fn（贪心匹配）。"""
    matched_gt = set()
    tp = 0
    for pb in preds:
        best_i, best_iou = -1, 0.0
        for gi, gb in enumerate(gts):
            if gi in matched_gt:
                continue
            v = iou_xyxy(pb, gb)
            if v > best_iou:
                best_iou, best_i = v, gi
        if best_iou >= iou_thr and best_i >= 0:
            tp += 1
            matched_gt.add(best_i)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def field_exact_match(
    pred_fields: dict[str, Any],
    gt_fields: dict[str, Any],
) -> tuple[int, int]:
    hit, total = 0, 0
    for k, gv in gt_fields.items():
        total += 1
        pv = pred_fields.get(k)
        if pv is None and gv is None:
            hit += 1
        elif pv is not None and gv is not None and str(pv).strip() == str(gv).strip():
            hit += 1
    return hit, total


def evaluate_one(
    gold: dict[str, Any],
    result: dict[str, Any],
    facts: dict[str, Any],
    *,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    # 框：按 entity_id
    gt_boxes = gold.get("boxes") or []  # [{entity_id, bbox}]
    pred_inst = facts.get("_instances") or []
    by_eid_gt: dict[str, list] = defaultdict(list)
    by_eid_pred: dict[str, list] = defaultdict(list)
    for g in gt_boxes:
        by_eid_gt[g["entity_id"]].append(g["bbox"])
    for p in pred_inst:
        if p.get("bbox"):
            by_eid_pred[p["entity_id"]].append(p["bbox"])

    box_tp = box_fp = box_fn = 0
    for eid in set(by_eid_gt) | set(by_eid_pred):
        tp, fp, fn = match_boxes(by_eid_pred.get(eid, []), by_eid_gt.get(eid, []), iou_thr)
        box_tp += tp
        box_fp += fp
        box_fn += fn

    # 字段
    field_hit = field_total = 0
    for gf in gold.get("fields") or []:
        # gf: {entity_id, instance_index?, fields:{}}
        eid = gf["entity_id"]
        gt_f = gf.get("fields") or {}
        cands = [i for i in pred_inst if i.get("entity_id") == eid]
        pred_f = (cands[0].get("fields") if cands else {}) or {}
        h, t = field_exact_match(pred_f, gt_f)
        field_hit += h
        field_total += t

    # 规则：应触发的 rule_id 集合
    expected_rules = set(gold.get("expected_rule_ids") or [])
    predicted_rules = {f["rule_id"] for f in result.get("findings") or []}
    # 规则级：以 expected 为正例
    tp_r = len(expected_rules & predicted_rules)
    fp_r = len(predicted_rules - expected_rules)
    fn_r = len(expected_rules - predicted_rules)

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return {
        "drawing_id": gold.get("drawing_id"),
        "boxes": _prf(box_tp, box_fp, box_fn),
        "fields_exact_match": field_hit / field_total if field_total else None,
        "fields_hit": field_hit,
        "fields_total": field_total,
        "rules": _prf(tp_r, fp_r, fn_r),
        "expected_rule_ids": sorted(expected_rules),
        "predicted_rule_ids": sorted(predicted_rules),
    }


def load_gold_dir(gold_dir: Path) -> list[dict[str, Any]]:
    items = []
    for p in sorted(gold_dir.glob("*.json")):
        items.append(load_json(p))
    return items


def run_eval(
    gold_dir: str | Path,
    *,
    backend: str = "mock",
    config_path: str | Path | None = None,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    gdir = resolve_path(gold_dir)
    golds = load_gold_dir(gdir)
    per_item = []
    sum_box = {"tp": 0, "fp": 0, "fn": 0}
    sum_rule = {"tp": 0, "fp": 0, "fn": 0}
    field_hit = field_total = 0

    for gold in golds:
        image = resolve_path(gold["image"])
        result = run(
            image,
            perception_backend=backend,
            config_path=config_path,
            rule_set_id=gold.get("rule_set_id", "default"),
            enqueue_review=False,
        )
        facts = load_json(result["artifacts"]["facts"])
        metrics = evaluate_one(gold, result, facts, iou_thr=iou_thr)
        per_item.append(metrics)
        for k in ("tp", "fp", "fn"):
            sum_box[k] += metrics["boxes"][k]
            sum_rule[k] += metrics["rules"][k]
        field_hit += metrics["fields_hit"]
        field_total += metrics["fields_total"]

    def _agg(s):
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f1, **s}

    summary = {
        "backend": backend,
        "n": len(golds),
        "iou_thr": iou_thr,
        "boxes": _agg(sum_box),
        "fields_exact_match": field_hit / field_total if field_total else None,
        "rules": _agg(sum_rule),
        "items": per_item,
    }
    out = resolve_path(cfg["work_dirs"]["reports"]) / f"eval_{backend}.json"
    dump_json(summary, out)
    summary["artifacts"] = {"report": str(out)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="S9 金标评测")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--backends", default=None, help="逗号分隔，对比多后端")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")] if args.backends else [args.backend]
    reports = []
    for b in backends:
        summary = run_eval(args.gold_dir, backend=b, config_path=args.config, iou_thr=args.iou)
        reports.append(summary)
        print(
            f"[{b}] n={summary['n']} "
            f"box_f1={summary['boxes']['f1']:.3f} "
            f"field_em={summary['fields_exact_match']} "
            f"rule_f1={summary['rules']['f1']:.3f}"
        )
        print(" ->", summary["artifacts"]["report"])

    if len(reports) > 1:
        cmp = {
            "compare": [
                {
                    "backend": r["backend"],
                    "boxes_f1": r["boxes"]["f1"],
                    "fields_exact_match": r["fields_exact_match"],
                    "rules_f1": r["rules"]["f1"],
                }
                for r in reports
            ]
        }
        out = resolve_path("work_dirs/reports/eval_compare.json")
        dump_json(cmp, out)
        print("compare ->", out)


if __name__ == "__main__":
    main()
