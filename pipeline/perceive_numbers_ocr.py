"""OCR 检出图中数字/尺寸文字，用于「数字重叠」等规则。

不依赖 VLM/LocateAnything；需要 rapidocr-onnxruntime。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.dimension_parse import merge_dimension_fields
from pipeline.drawing_parse_plan import dimension_marks_backend, is_dimension_marks_enabled
from pipeline import load_config, resolve_path
from pipeline.text_angle import text_angle_from_quad

# 含数字的工程标注：12、φ20、R5、M8、3.5、±0.1、45° 等
_NUMBERISH = re.compile(
    r"(?:"
    r"[±＋+]?\d+(?:\.\d+)?"  # 纯数字/小数
    r"|[φΦ∅Ø]\s*\d+(?:\.\d+)?"
    r"|[Rr]\s*\d+(?:\.\d+)?"
    r"|[Mm]\s*\d+(?:\.\d+)?"
    r"|\d+\s*[°º]"
    r")"
)


def _looks_like_number_mark(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(_NUMBERISH.search(t))


def _quad_to_xyxy(box) -> list[int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def perceive_numbers_ocr(
    image_path: str | Path,
    plan: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    allow_mock_fallback: bool = True,
) -> dict[str, Any]:
    """全图 OCR，筛出数字类文字，输出 number_mark / annotation instances。"""
    cfg = config or load_config()
    ocr_cfg = cfg.get("models", {}).get("ocr", {})
    conf_th = float(ocr_cfg.get("confidence_threshold", 0.4))
    meta = meta or {}
    image = Image.open(resolve_path(image_path)).convert("RGB")
    width, height = image.size
    meta.setdefault("width", width)
    meta.setdefault("height", height)

    try:
        from pipeline.ocr_engine import get_rapid_ocr

        engine = get_rapid_ocr()
    except ImportError:
        if allow_mock_fallback:
            from pipeline.perceive_common import mock_perceive

            plan = plan or [
                {
                    "entity_id": "number_mark",
                    "locate_query": "数字标注",
                    "fields": [{"name": "text", "parse_hint": "数字"}],
                }
            ]
            payload = mock_perceive(plan, meta)
            payload["note"] = "OCR 未安装，数字重叠检测已回退 mock"
            return payload
        raise

    import numpy as np

    result, _ = engine(np.array(image))
    target_eids = {
        e.get("entity_id")
        for e in (plan or [])
        if e.get("entity_id") in {"number_mark", "annotation", "annotations"}
    }
    if not target_eids:
        target_eids = {"number_mark"}
    entity_id = "number_mark" if "number_mark" in target_eids else next(iter(target_eids))

    instances: list[dict[str, Any]] = []
    idx = 0
    for row in result or []:
        # row: [box, text, score]
        if not row or len(row) < 3:
            continue
        box, text, score = row[0], str(row[1]), float(row[2])
        if score < conf_th:
            continue
        if not _looks_like_number_mark(text):
            continue
        bbox = _quad_to_xyxy(box)
        text_ang = text_angle_from_quad(box)
        fields: dict[str, Any] = {"text": text, "angle": text_ang}
        if is_dimension_marks_enabled(cfg) and dimension_marks_backend(cfg) == "ocr":
            fields = merge_dimension_fields(fields, text)
        instances.append(
            {
                "entity_id": entity_id,
                "instance_id": f"{entity_id}#{idx}",
                "label": "数字标注",
                "bbox": bbox,
                "fields": fields,
                "raw_text": text,
                "confidence": score,
                "needs_review": score < 0.6,
                "angle": text_ang,
            }
        )
        idx += 1

    return {"backend": "ocr_numbers", "instances": instances}
