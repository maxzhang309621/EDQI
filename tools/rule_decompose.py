"""S1：自然语言规则 → draft JSON（Qwen3 本地）。

参考: https://github.com/QwenLM/Qwen3
权重: models/Qwen3-8B
无权重时可用 --offline-template 生成可编辑草稿。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from engines import validate_rule
from pipeline import dump_json, load_config, load_json, model_path_ready, resolve_model_source, resolve_path

_TEXT_MODEL = None
_TEXT_TOKENIZER = None


SYSTEM_PROMPT = """你是工程制图质检规则工程师。将自然语言检测规则拆成 JSON。
只输出一个 JSON 对象，不要 markdown，不要解释。
字段要求:
- rule_id: 大写蛇形命名
- name, severity(error|warn|info), status 固定为 draft
- entities: [{entity_id, locate_query, fields:[{name, parse_hint, required}]}]
- logic: 仅使用算子 not_empty/eq/neq/in/regex/count_gte/count_lte/ref_exists/and/or/not
- message: 面向质检员的中文说明
禁止发明白名单外的 logic.op。
"""


def _fewshot_block(fewshot_dir: Path) -> str:
    if not fewshot_dir.exists():
        return ""
    parts = []
    for path in sorted(fewshot_dir.glob("*.json"))[:3]:
        parts.append(json.dumps(load_json(path), ensure_ascii=False))
    if not parts:
        return ""
    return "示例规则:\n" + "\n\n".join(parts)


def _load_text_llm(model_cfg: dict[str, Any]):
    global _TEXT_MODEL, _TEXT_TOKENIZER
    if _TEXT_MODEL is not None:
        return _TEXT_MODEL, _TEXT_TOKENIZER
    if not model_path_ready(model_cfg):
        raise FileNotFoundError(model_cfg.get("path"))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolve_model_source(model_cfg)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        device_map=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    _TEXT_MODEL, _TEXT_TOKENIZER = model, tokenizer
    return model, tokenizer


def _generate_with_qwen(model_cfg: dict[str, Any], user_text: str) -> str:
    from pipeline.perceive_common import extract_json_payload

    model, tokenizer = _load_text_llm(model_cfg)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    # Qwen3: enable_thinking=False 保证 JSON 干净
    kwargs = {}
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(model_cfg.get("enable_thinking", False)),
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(model_cfg.get("max_new_tokens", 2048)),
        **kwargs,
    )
    gen = out[0][inputs.input_ids.shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    # 若含 think 标签，截掉
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    payload = extract_json_payload(text)
    return json.dumps(payload, ensure_ascii=False)


def offline_template(natural_language: str) -> dict[str, Any]:
    """无模型时的可编辑草稿骨架。"""
    return {
        "rule_id": "NEW_RULE_EDIT_ME",
        "name": natural_language.strip()[:40] or "新规则",
        "severity": "error",
        "status": "draft",
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entities": [
            {
                "entity_id": "title_block",
                "locate_query": "标题栏",
                "cardinality": "one",
                "fields": [
                    {"name": "part_no", "parse_hint": "图号", "required": True}
                ],
            }
        ],
        "logic": {"op": "not_empty", "path": "title_block.part_no"},
        "message": natural_language.strip() or "请手工完善规则文案",
        "_source_nl": natural_language,
        "_note": "由 offline-template 生成，请人工改 rule_id/entities/logic 后 validate+activate",
    }


def decompose(
    natural_language: str,
    *,
    config_path: str | Path | None = None,
    offline_template_mode: bool = False,
    retries: int = 2,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    model_cfg = cfg.get("models", {}).get("qwen3_text", {})
    fewshot = _fewshot_block(resolve_path(cfg["rules"].get("fewshot_dir", "rules/fewshot")))

    if offline_template_mode or not model_path_ready(model_cfg):
        draft = offline_template(natural_language)
        if not model_path_ready(model_cfg):
            draft["_note"] = (
                f"本地权重未就绪({model_cfg.get('path')})，已生成离线模板。"
                f"下载 {model_cfg.get('hf_id')} 后重跑可自动拆解。"
            )
        return draft

    user = f"{fewshot}\n\n请拆解以下规则:\n{natural_language}".strip()
    last_err = None
    for _ in range(max(1, retries)):
        try:
            raw = _generate_with_qwen(model_cfg, user)
            draft = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(draft, str):
                draft = json.loads(draft)
            draft["status"] = "draft"
            draft.setdefault("version", 1)
            draft["updated_at"] = datetime.now(timezone.utc).isoformat()
            draft["_source_nl"] = natural_language
            errors = validate_rule(draft)
            if errors:
                last_err = errors
                user = user + "\n\n上次输出校验失败: " + "; ".join(errors) + "\n请修正后重新输出完整 JSON。"
                continue
            return draft
        except Exception as e:
            last_err = [str(e)]
    draft = offline_template(natural_language)
    draft["status"] = "invalid"
    draft["_validation_errors"] = last_err
    return draft


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 规则语义拆解")
    parser.add_argument("text", nargs="?", help="自然语言规则")
    parser.add_argument("--file", help="从文件读取自然语言规则")
    parser.add_argument("--out", default=None, help="输出 draft json 路径")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--offline-template", action="store_true")
    args = parser.parse_args()

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise SystemExit("请提供规则文本或 --file")

    draft = decompose(text, config_path=args.config, offline_template_mode=args.offline_template)
    cfg = load_config(args.config)
    out = args.out
    if not out:
        rid = draft.get("rule_id", "draft")
        out = str(resolve_path(cfg["rules"]["drafts_dir"]) / f"{rid}.json")
    dump_json(draft, out)
    print(f"wrote {out} status={draft.get('status')}")
    errs = validate_rule(draft)
    if errs:
        print("validation warnings:")
        for e in errs:
            print(" -", e)


if __name__ == "__main__":
    main()
