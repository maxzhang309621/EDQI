"""总入口：在线检测轨 S3→S8（含缓存分块、hybrid 回退、复核入队）。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from engines import collect_entities, load_rules
from engines.rule_engine import evaluate
from pipeline import dump_json, load_config, model_download_status, resolve_path
from pipeline.build_facts import build_facts
from pipeline.drawing_parse_plan import (
    OCR_OVERLAP_ENTITY_IDS,
    build_drawing_parse_plan,
    dimension_marks_backend,
    get_dimension_marks_config,
    is_dimension_marks_ocr_locate_vlm,
    is_dimension_marks_vlm,
    is_tables_only,
    merge_perception_plans,
    split_plan_for_backends,
)
from pipeline.ingest import ingest
from pipeline.dimension_parse import (
    prescreen_ocr_dimension_candidates,
    sanitize_number_mark_instances,
)
from pipeline.perceive_common import (
    build_table_exclude_regions,
    collect_table_value_tokens,
    filter_instances_matching_table_values,
    filter_instances_outside_bboxes,
    mock_perceive,
)
from pipeline.perceive_la_ocr import perceive_la_ocr
from pipeline.perceive_number_overlap import perceive_number_overlap
from pipeline.perceive_qwen_vl import (
    filter_ocr_dimension_candidates_with_vlm,
    perceive_qwen_vl,
)
from pipeline.perceive_utils import perceive_with_cache_and_tiles
from pipeline.render_report import render_report
from pipeline.review_queue import enqueue_from_result

import yaml


def load_run_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 run 运行参数 YAML。"""
    rel = path or "configs/run.yaml"
    p = resolve_path(rel)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def resolve_run_options(
    *,
    drawing: str | Path | None = None,
    run_config: dict[str, Any] | None = None,
    perception_backend: str | None = None,
    rule_set_id: str | None = None,
    rule_ids: list[str] | None = None,
    config_path: str | Path | None = None,
    page: int | None = None,
    enqueue_review: bool | None = None,
    drawing_parse: bool | None = None,
    tables_only: bool | None = None,
    allow_missing_drawing: bool = False,
) -> dict[str, Any]:
    """合并 run.yaml 与显式参数；显式非 None 优先。"""
    rc = dict(run_config or {})
    out_drawing = drawing if drawing not in (None, "") else rc.get("drawing")
    if out_drawing in (None, ""):
        if allow_missing_drawing:
            out_drawing = None
        else:
            raise ValueError(
                "未指定图纸路径：请在命令行传入 drawing，或在 configs/run.yaml 设置 drawing"
            )

    out_backend = perception_backend if perception_backend is not None else rc.get("backend")
    out_rule_set = rule_set_id if rule_set_id is not None else rc.get("rule_set_id", "default")
    out_rule_ids = rule_ids if rule_ids is not None else rc.get("rule_ids")
    if isinstance(out_rule_ids, str):
        out_rule_ids = [out_rule_ids]
    elif out_rule_ids is not None:
        out_rule_ids = [str(x) for x in out_rule_ids]

    out_config = config_path if config_path is not None else rc.get("config", "configs/default.yaml")
    out_page = int(page if page is not None else rc.get("page", 1))

    if enqueue_review is not None:
        out_enqueue = bool(enqueue_review)
    else:
        out_enqueue = bool(rc.get("enqueue_review", True))

    if drawing_parse is not None:
        out_parse = bool(drawing_parse)
    else:
        out_parse = bool(rc.get("drawing_parse", True))

    if tables_only is not None:
        out_tables_only: bool | None = bool(tables_only)
    elif "tables_only" in rc and rc.get("tables_only") is not None:
        out_tables_only = bool(rc.get("tables_only"))
    else:
        out_tables_only = None

    return {
        "drawing": out_drawing,
        "perception_backend": out_backend,
        "rule_set_id": str(out_rule_set or "default"),
        "rule_ids": out_rule_ids,
        "config_path": out_config,
        "page": out_page,
        "enqueue_review": out_enqueue,
        "drawing_parse": out_parse,
        "tables_only": out_tables_only,
    }


def _is_number_overlap_task(plan: list[dict[str, Any]], rules: list[dict[str, Any]] | None = None) -> bool:
    """仅数字重叠类规则时，不应走纯 VLM（重叠字易漏检/合并）。"""
    if rules and all(r.get("rule_id") == "NUM_TEXT_NO_OVERLAP" for r in rules):
        ocr_plan, vl_plan = split_plan_for_backends(plan)
        return bool(ocr_plan) and not vl_plan
    eids = {e.get("entity_id") for e in plan}
    return bool(eids) and eids.issubset(OCR_OVERLAP_ENTITY_IDS)


def _missing_required(instances: list[dict[str, Any]], plan: list[dict[str, Any]]) -> set[str]:
    by_eid = {e["entity_id"]: e for e in plan}
    present = {
        i["entity_id"]
        for i in instances
        if i.get("bbox") and list(i.get("bbox") or []) != [0, 0, 0, 0]
    }
    missing = set()
    for eid, ent in by_eid.items():
        if eid not in present:
            missing.add(eid)
            continue
        required_names = [f["name"] for f in ent.get("fields", []) if f.get("required")]
        if not required_names:
            continue
        ok_any = False
        for inst in instances:
            if inst.get("entity_id") != eid:
                continue
            fields = inst.get("fields") or {}
            if all(fields.get(n) not in (None, "") for n in required_names):
                ok_any = True
                break
        if not ok_any:
            missing.add(eid)
    return missing


def _needs_review_entities(instances: list[dict[str, Any]]) -> set[str]:
    return {i["entity_id"] for i in instances if i.get("needs_review")}


def _merge_instance_payloads(
    *payloads: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    notes: list[str] = []
    for p in payloads:
        if not p:
            continue
        instances.extend(list(p.get("instances") or []))
        notes.extend(list(p.get("notes") or []))
        note = p.get("note")
        if note:
            notes.append(str(note))
    return {"backend": backend, "instances": instances, "notes": notes}


def perceive(
    backend: str,
    image_path: Path,
    plan: list[dict[str, Any]],
    meta: dict[str, Any],
    config: dict[str, Any],
    *,
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    auto = bool((config.get("perception") or {}).get("auto_number_overlap_backend", True))
    dual = bool(config.get("perception_dual_backend", True))
    ocr_plan, vl_plan = split_plan_for_backends(plan)

    # 仅重叠实体：整段走 OCR
    if auto and _is_number_overlap_task(plan, rules) and backend in {"qwen_vl", "hybrid", "mock"}:
        if backend != "mock":
            payload = perceive_number_overlap(image_path, plan, meta, config, rules=rules)
            payload["notes"] = list(payload.get("notes") or []) + [
                f"auto_switched_from={backend} (VLM 不适合重叠数字检出)"
            ]
            return payload

    use_cache_tiles = bool((config.get("perception") or {}).get("cache_enabled", True)) or bool(
        (config.get("perception") or {}).get("tile_enabled", True)
    )

    def _wrap(name: str, fn, sub_plan: list[dict[str, Any]]):
        if use_cache_tiles and name != "mock":
            return perceive_with_cache_and_tiles(
                backend_name=name,
                image_path=image_path,
                plan=sub_plan,
                meta=meta,
                config=config,
                perceive_fn=fn,
            )
        return fn(image_path, sub_plan, meta, config)

    # VL 配置实体 + 重叠/OCR 属性实体：双后端合并
    # 纯 vlm 尺寸且无 VL 实体时不进此支；ocr / ocr_locate_vlm_filter 可在无表格时仍跑 OCR→(VLM精筛)
    if (
        dual
        and auto
        and ocr_plan
        and backend in {"qwen_vl", "hybrid"}
        and (vl_plan or not is_dimension_marks_vlm(config))
    ):
        if vl_plan:
            vl_payload = _wrap("qwen_vl", perceive_qwen_vl, vl_plan)
        else:
            vl_payload = {"backend": "qwen_vl", "instances": [], "notes": ["vl_plan_empty"]}
        dim_cfg = get_dimension_marks_config(config)
        dim_backend = dimension_marks_backend(config)
        dim_vlm = is_dimension_marks_vlm(config)
        dim_ocr_vlm = is_dimension_marks_ocr_locate_vlm(config)
        exclude_bbs: list[list[int]] = []
        table_tokens: set[str] = set()
        # OCR 属性路径（ocr / ocr_locate_vlm_filter）按表格框过滤候选
        if (not dim_vlm) and bool(dim_cfg.get("exclude_table_regions", True)):
            page_w = int(meta.get("width") or 0)
            page_h = int(meta.get("height") or 0)
            exclude_pad = float(dim_cfg.get("exclude_table_pad", 2.0))
            expand_up = float(dim_cfg.get("exclude_table_expand_up", 0.12))
            exclude_bbs = build_table_exclude_regions(
                vl_payload.get("instances") or [],
                page_w=page_w,
                page_h=page_h,
                pad=exclude_pad,
                expand_up_frac=expand_up,
            )
            table_tokens = collect_table_value_tokens(vl_payload.get("instances") or [])
        ocr_payload = perceive_number_overlap(
            image_path,
            ocr_plan,
            meta,
            config,
            rules=rules,
            exclude_bboxes=exclude_bbs or None,
            exclude_pad=0.0,
            exclude_table_texts=table_tokens or None,
        )
        ocr_instances = list(ocr_payload.get("instances") or [])
        ocr_notes = list(ocr_payload.get("notes") or [])
        if dim_vlm:
            # 属性已由 VLM 全图定位：重叠 OCR 只保留 keep_pair 证据
            before = len(ocr_instances)
            ocr_instances = [i for i in ocr_instances if i.get("keep_pair")]
            ocr_notes.append(f"ocr_keep_pair_only={len(ocr_instances)}/{before}")
        elif dim_ocr_vlm:
            # OCR 宽召回+规则初筛 → VLM crop 精筛；keep_pair 原样保留
            page_w = int(meta.get("width") or 0)
            page_h = int(meta.get("height") or 0)
            dim_ent = next(
                (
                    e
                    for e in ocr_plan
                    if str(e.get("parse_kind") or "").lower() == "dimension_marks"
                ),
                None,
            )
            if dim_ent is None:
                dim_ent = {
                    **dim_cfg,
                    "parse_kind": "dimension_marks",
                    "entity_id": str(dim_cfg.get("entity_id") or "number_mark"),
                    "fields": dim_cfg.get("fields") or [],
                }
            candidates, keep_pairs, dropped = prescreen_ocr_dimension_candidates(
                ocr_instances,
                page_w=page_w,
                page_h=page_h,
                require_dim_kind=bool(dim_ent.get("require_dim_kind", True)),
                require_basic_size=bool(dim_ent.get("require_basic_size", True)),
                max_bbox_width_ratio=float(dim_ent.get("max_bbox_width_ratio", 0.22)),
                max_aspect_ratio=float(dim_ent.get("max_aspect_ratio", 8.0)),
                max_height_ratio=float(dim_ent.get("max_bbox_height_ratio", 0.12)),
                max_area_ratio=float(dim_ent.get("max_bbox_area_ratio", 0.035)),
                max_candidates=int(dim_ent.get("vlm_filter_max_candidates", 48)),
                exclude_bboxes=exclude_bbs or None,
                exclude_pad=float(dim_ent.get("exclude_table_pad", 2.0)),
                exclude_table_texts=table_tokens or None,
            )
            ocr_notes.append(
                f"ocr_dim_prescreen={len(candidates)}/{max(0, len(ocr_instances) - len(keep_pairs))}"
            )
            ocr_notes.append(f"ocr_dim_prescreen_dropped={dropped}")
            ocr_notes.append(f"ocr_keep_pair={len(keep_pairs)}")
            accepted = filter_ocr_dimension_candidates_with_vlm(
                image_path,
                candidates,
                dim_ent,
                meta,
                config,
                notes=ocr_notes,
            )
            ocr_instances = accepted + keep_pairs
        ocr_payload = {**ocr_payload, "instances": ocr_instances, "notes": ocr_notes}
        merged = _merge_instance_payloads(
            vl_payload,
            ocr_payload,
            backend=f"{vl_payload.get('backend', 'qwen_vl')}+number_overlap",
        )
        # 最终清理：大范围假框 / 非声明属性 / 表格区内 number_mark
        page_w = int(meta.get("width") or 0)
        page_h = int(meta.get("height") or 0)
        if not exclude_bbs and bool(dim_cfg.get("exclude_table_regions", True)):
            exclude_bbs = build_table_exclude_regions(
                merged.get("instances") or [],
                page_w=page_w,
                page_h=page_h,
                pad=float(dim_cfg.get("exclude_table_pad", 2.0)),
                expand_up_frac=float(dim_cfg.get("exclude_table_expand_up", 0.12)),
            )
            if not table_tokens:
                table_tokens = collect_table_value_tokens(merged.get("instances") or [])
        allowed_names = [
            str(f.get("name"))
            for f in (dim_cfg.get("fields") or [])
            if isinstance(f, dict) and f.get("name")
        ]
        cleaned, n_drop = sanitize_number_mark_instances(
            list(merged.get("instances") or []),
            page_w=page_w,
            page_h=page_h,
            allowed_field_names=allowed_names or None,
            require_dim_kind=bool(dim_cfg.get("require_dim_kind", True)),
            require_basic_size=bool(dim_cfg.get("require_basic_size", True)),
            max_bbox_width_ratio=float(dim_cfg.get("max_bbox_width_ratio", 0.22)),
            max_aspect_ratio=float(dim_cfg.get("max_aspect_ratio", 8.0)),
            max_height_ratio=float(dim_cfg.get("max_bbox_height_ratio", 0.12)),
            max_area_ratio=float(dim_cfg.get("max_bbox_area_ratio", 0.035)),
            exclude_bboxes=exclude_bbs or None,
            exclude_pad=float(dim_cfg.get("exclude_table_pad", 2.0)),
            exclude_table_texts=table_tokens or None,
        )
        merged["instances"] = cleaned
        if isinstance(vl_payload.get("timing"), dict):
            merged["timing"] = vl_payload["timing"]
        merged["notes"] = list(merged.get("notes") or []) + [
            "dual_backend=vl+number_overlap",
            f"vl_entities={[e.get('entity_id') for e in vl_plan]}",
            f"ocr_entities={[e.get('entity_id') for e in ocr_plan]}",
            f"dimension_marks_backend={dim_backend}",
            f"table_exclude_boxes={len(exclude_bbs)}",
            f"table_value_tokens={len(table_tokens)}",
            f"number_mark_sanitized_dropped={n_drop}",
        ]
        return merged

    if backend == "mock":
        return mock_perceive(plan, meta)
    if backend in {"ocr_numbers", "number_overlap"}:
        # 显式 OCR 后端时只跑重叠实体，避免把表格实体塞进 OCR
        return perceive_number_overlap(image_path, ocr_plan or plan, meta, config, rules=rules)
    if backend == "qwen_vl":
        return _wrap("qwen_vl", perceive_qwen_vl, plan)
    if backend == "locateanything_ocr":
        return _wrap("locateanything_ocr", perceive_la_ocr, plan)
    if backend == "hybrid":
        primary = _wrap("qwen_vl", perceive_qwen_vl, plan)
        missing = _missing_required(primary.get("instances") or [], plan)
        missing |= _needs_review_entities(primary.get("instances") or [])
        if not missing:
            primary["backend"] = "hybrid"
            return primary
        sub_plan = [e for e in plan if e["entity_id"] in missing]
        fallback = perceive_with_cache_and_tiles(
            backend_name="locateanything_ocr",
            image_path=image_path,
            plan=sub_plan,
            meta=meta,
            config=config,
            perceive_fn=perceive_la_ocr,
        )
        kept = [i for i in primary.get("instances", []) if i["entity_id"] not in missing]
        merged = kept + list(fallback.get("instances") or [])
        return {
            "backend": "hybrid",
            "instances": merged,
            "fallback_entities": sorted(missing),
            "notes": list(primary.get("notes") or []) + list(fallback.get("notes") or []),
        }
    raise ValueError(f"未知 perception_backend: {backend}")


def run(
    drawing_path: str | Path,
    *,
    rule_set_id: str = "default",
    perception_backend: str | None = None,
    config_path: str | Path | None = None,
    page: int = 1,
    enqueue_review: bool | None = None,
    rule_ids: list[str] | None = None,
    drawing_parse: bool = True,
    tables_only: bool | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if tables_only is not None:
        dp = dict(cfg.get("drawing_parse") or {})
        dp["tables_only"] = bool(tables_only)
        cfg["drawing_parse"] = dp

    backend = perception_backend or cfg.get("perception_backend", "mock")

    if backend in {"qwen_vl", "hybrid"}:
        st = model_download_status(cfg.get("models", {}).get("qwen3_vl", {}))
        if not st["ready"] and backend != "mock":
            pass

    page_image, meta = ingest(
        drawing_path,
        output_dir=cfg.get("ingest", {}).get("output_dir"),
        max_side=int(cfg.get("ingest", {}).get("max_side", 2048)),
        dpi=int(cfg.get("ingest", {}).get("dpi", 200)),
        page=page,
    )

    rules = load_rules(
        cfg["rules"]["library_dir"],
        only_active=bool(cfg["rules"].get("only_active", True)),
        rule_ids=rule_ids,
    )
    if not rules:
        raise RuntimeError(
            f"未找到 active 规则，请检查 {cfg['rules']['library_dir']}"
            + (f" 或 rule_ids={rule_ids}" if rule_ids else "")
        )

    rule_plan = collect_entities(rules)
    parse_plan = build_drawing_parse_plan(cfg, enabled=drawing_parse)
    only_tables = is_tables_only(cfg)
    if only_tables:
        plan = list(parse_plan)
        if not plan:
            raise RuntimeError(
                "drawing_parse.tables_only=true，但未产出任何表格/对象解析实体，"
                "请检查 configs/drawing_parse.yaml"
            )
    else:
        plan = merge_perception_plans(rule_plan, parse_plan)

    run_t0 = time.perf_counter()
    perc_t0 = time.perf_counter()
    instances_payload = perceive(backend, page_image, plan, meta, cfg, rules=rules)
    perception_s = time.perf_counter() - perc_t0
    print(
        f"[timing] perception={perception_s:.2f}s "
        f"backend={instances_payload.get('backend', backend)}",
        flush=True,
    )
    if only_tables:
        notes = list(instances_payload.get("notes") or [])
        notes.append("tables_only=true (跳过数字重叠 OCR 合并)")
        instances_payload["notes"] = notes

    facts_path = resolve_path(cfg["work_dirs"]["facts"]) / f"{meta['drawing_id']}.json"
    facts = build_facts(instances_payload, meta, dump_path=str(facts_path))

    result = evaluate(
        facts,
        rules,
        rule_set_id=rule_set_id,
        perception_backend=instances_payload.get("backend", backend),
    )
    result["perception_notes"] = instances_payload.get("notes")
    if instances_payload.get("fallback_entities"):
        result["fallback_entities"] = instances_payload["fallback_entities"]

    paths = render_report(page_image, result, config=cfg, facts=facts)
    result["artifacts"] = {
        "page_image": str(page_image),
        "facts": str(facts_path),
        "vis": str(paths["vis"]),
        "report": str(paths["report"]),
        "markdown": str(paths.get("markdown")) if paths.get("markdown") else None,
    }
    dump_json(result, paths["report"])

    do_enqueue = cfg.get("review", {}).get("auto_enqueue", True) if enqueue_review is None else enqueue_review
    if do_enqueue:
        qpath = enqueue_from_result(result, facts, config=cfg)
        result["artifacts"]["review_queue"] = str(qpath) if qpath else None

    timing: dict[str, Any] = {"perception_s": round(perception_s, 3)}
    if isinstance(instances_payload.get("timing"), dict):
        timing["qwen_vl"] = instances_payload["timing"]
    timing["total_s"] = round(time.perf_counter() - run_t0, 3)
    result["timing"] = timing
    notes_all = list(instances_payload.get("notes") or [])
    cache_hits = [n for n in notes_all if str(n).startswith("cache_hit:")]
    if cache_hits:
        print(f"[cache] hits={len(cache_hits)} {cache_hits}", flush=True)
    else:
        print("[cache] hits=0 (本次未命中感知缓存)", flush=True)
    print(
        f"[timing] total={timing['total_s']:.2f}s "
        f"(perception={timing['perception_s']:.2f}s)",
        flush=True,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDQI 在线质检入口（默认参数见 configs/run.yaml，CLI 优先）"
    )
    parser.add_argument(
        "drawing",
        nargs="?",
        default=None,
        help="图纸路径（png/jpg/pdf）；可省略并在 run.yaml 中配置 drawing",
    )
    parser.add_argument(
        "--run-config",
        default="configs/run.yaml",
        help="run 运行参数 YAML（默认 configs/run.yaml）",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="qwen_vl|locateanything_ocr|hybrid|mock|number_overlap|ocr_numbers",
    )
    parser.add_argument("--rule-set-id", default=None, help="默认取自 run.yaml")
    parser.add_argument(
        "--rule-id",
        action="append",
        default=None,
        help="只跑指定规则（可多次）。例: --rule-id NUM_TEXT_NO_OVERLAP",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="主工程配置；默认取自 run.yaml 的 config",
    )
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--no-review-queue", action="store_true", help="关闭复核入队")
    parser.add_argument(
        "--no-drawing-parse",
        action="store_true",
        help="关闭 drawing_parse.yaml 信息解析（仅跑规则实体，如数字重叠）",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="仅表格 VLM，不合并数字重叠 OCR（覆盖配置中的 tables_only）",
    )
    args = parser.parse_args()

    run_cfg = load_run_config(args.run_config)
    opts = resolve_run_options(
        drawing=args.drawing,
        run_config=run_cfg,
        perception_backend=args.backend,
        rule_set_id=args.rule_set_id,
        rule_ids=args.rule_id,
        config_path=args.config,
        page=args.page,
        enqueue_review=False if args.no_review_queue else None,
        drawing_parse=False if args.no_drawing_parse else None,
        tables_only=True if args.tables_only else None,
    )

    result = run(
        opts["drawing"],
        rule_set_id=opts["rule_set_id"],
        perception_backend=opts["perception_backend"],
        config_path=opts["config_path"],
        page=opts["page"],
        enqueue_review=opts["enqueue_review"],
        rule_ids=opts["rule_ids"],
        drawing_parse=opts["drawing_parse"],
        tables_only=opts["tables_only"],
    )
    print(f"passed={result['passed']} findings={len(result['findings'])}")
    for f in result["findings"]:
        print(f" - [{f['severity']}] {f['rule_id']}: {f['message']} (actual={f.get('actual')})")
    if result.get("timing"):
        print("timing:", result["timing"])
    print("artifacts:", result.get("artifacts"))


if __name__ == "__main__":
    main()
