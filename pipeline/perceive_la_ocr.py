"""S4+S5 备选：LocateAnything 定位 + OCR 解析。

定位参考: https://github.com/NVlabs/Eagle (Embodied/LocateAnythingWorker)
OCR: RapidOCR / PaddleOCR
权重本地目录默认: models/LocateAnything-3B
未就绪时回退 mock。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from pipeline import load_config, model_path_ready, resolve_path
from pipeline.perceive_common import expand_bbox, mock_perceive

_LA_WORKER = None
_OCR = None


def _load_locateanything(model_cfg: dict[str, Any]):
    global _LA_WORKER
    if _LA_WORKER is not None:
        return _LA_WORKER
    if not model_path_ready(model_cfg):
        raise FileNotFoundError(model_cfg.get("path"))

    # 官方 worker 位于 Eagle/Embodied；本仓库预留 vendor 或 PYTHONPATH
    try:
        from locateanything_worker import LocateAnythingWorker  # type: ignore
    except ImportError as e:
        raise ImportError(
            "未找到 locateanything_worker。请按 NVlabs/Eagle Embodied 安装，"
            "或将 locateanything_worker.py 放到 PYTHONPATH。"
        ) from e

    source = str(resolve_path(model_cfg["path"]))
    _LA_WORKER = LocateAnythingWorker(source)
    return _LA_WORKER


def _load_ocr(ocr_cfg: dict[str, Any]):
    global _OCR
    if _OCR is not None:
        return _OCR
    backend = (ocr_cfg or {}).get("backend", "rapidocr_onnxruntime")

    if backend == "paddleocr":
        from paddleocr import PaddleOCR  # type: ignore

        _OCR = ("paddle", PaddleOCR(use_angle_cls=True, lang="ch", show_log=False))
        return _OCR

    if backend in {"rapidocr", "rapidocr_v3"}:
        try:
            from rapidocr import RapidOCR  # type: ignore

            _OCR = ("rapid_v3", RapidOCR())
            return _OCR
        except ImportError:
            # 回退旧包名
            backend = "rapidocr_onnxruntime"

    # 默认 / 兼容：rapidocr-onnxruntime（Py3.13 可用 1.2.x）
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except ImportError as e:
        raise ImportError(
            "未安装 OCR 库。Python 3.13 请安装: pip install 'rapidocr-onnxruntime>=1.2.0,<1.3.0'\n"
            "（不要装 >=1.3.0，该系列不支持 3.13）"
        ) from e
    _OCR = ("rapid", RapidOCR())
    return _OCR


def _ocr_read(ocr, crop: Image.Image) -> tuple[str, float]:
    kind, engine = ocr
    import numpy as np

    arr = np.array(crop.convert("RGB"))
    if kind == "rapid":
        result, _ = engine(arr)
        if not result:
            return "", 0.0
        texts = [r[1] for r in result]
        confs = [float(r[2]) for r in result]
        return "\n".join(texts), (sum(confs) / len(confs) if confs else 0.0)
    if kind == "rapid_v3":
        # rapidocr>=3 返回对象，字段因版本略有差异
        out = engine(arr)
        result = out[0] if isinstance(out, (tuple, list)) else getattr(out, "txts", None)
        if not result:
            return "", 0.0
        if hasattr(out, "txts"):
            texts = list(out.txts or [])
            confs = [float(x) for x in (out.scores or [])]
        else:
            texts, confs = [], []
            for r in result:
                if isinstance(r, (list, tuple)) and len(r) >= 3:
                    texts.append(str(r[1]))
                    confs.append(float(r[2]))
        return "\n".join(texts), (sum(confs) / len(confs) if confs else 0.0)
    # paddle
    result = engine.ocr(arr, cls=True)
    lines = []
    confs = []
    for block in result or []:
        for line in block or []:
            if len(line) >= 2:
                lines.append(line[1][0])
                confs.append(float(line[1][1]))
    return "\n".join(lines), (sum(confs) / len(confs) if confs else 0.0)


def _parse_boxes_from_la(answer: Any, width: int, height: int) -> list[list[int]]:
    """尽量兼容 LocateAnythingWorker 输出。"""
    boxes: list[list[int]] = []
    try:
        from locateanything_worker import LocateAnythingWorker  # type: ignore

        parsed = LocateAnythingWorker.parse_boxes(answer, width, height)
        for b in parsed or []:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            elif isinstance(b, dict) and "bbox" in b:
                bb = b["bbox"]
                boxes.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])
        return boxes
    except Exception:
        pass

    # 兜底：answer 已是 list
    if isinstance(answer, list):
        for b in answer:
            if isinstance(b, dict):
                bb = b.get("bbox") or b.get("box")
                if bb and len(bb) >= 4:
                    boxes.append([int(x) for x in bb[:4]])
            elif isinstance(b, (list, tuple)) and len(b) >= 4:
                boxes.append([int(x) for x in b[:4]])
    return boxes


def _heuristic_fields(raw_text: str, ent: dict[str, Any]) -> dict[str, Any]:
    """无文本 LLM 时的简易字段抽取：按 parse_hint 关键词后取值。"""
    import re

    fields: dict[str, Any] = {}
    for f in ent.get("fields", []):
        hint = f.get("parse_hint") or f["name"]
        # 图号 A-1001 / 材料 45钢
        m = re.search(rf"{re.escape(hint)}\s*[:：]?\s*([^\s,，;；]+)", raw_text)
        if m:
            fields[f["name"]] = m.group(1)
        else:
            fields[f["name"]] = None
    return fields


def perceive_la_ocr(
    image_path: str | Path,
    plan: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    allow_mock_fallback: bool = True,
) -> dict[str, Any]:
    cfg = config or load_config()
    la_cfg = cfg.get("models", {}).get("locateanything", {})
    ocr_cfg = cfg.get("models", {}).get("ocr", {})
    meta = meta or {}
    image = Image.open(resolve_path(image_path)).convert("RGB")
    width, height = image.size
    meta.setdefault("width", width)
    meta.setdefault("height", height)

    if not model_path_ready(la_cfg):
        if allow_mock_fallback:
            payload = mock_perceive(plan, meta)
            payload["backend"] = "mock"
            payload["note"] = "LocateAnything 权重未就绪，已回退 mock"
            return payload
        raise FileNotFoundError(la_cfg.get("path"))

    worker = _load_locateanything(la_cfg)
    ocr = _load_ocr(ocr_cfg)
    conf_th = float(ocr_cfg.get("confidence_threshold", 0.5))
    mode = la_cfg.get("generation_mode", "hybrid")

    instances: list[dict[str, Any]] = []
    for ent in plan:
        eid = ent["entity_id"]
        query = ent.get("locate_query", eid)
        result = worker.ground_multi(image, query, generation_mode=mode)
        answer = result.get("answer", result)
        boxes = _parse_boxes_from_la(answer, width, height)
        if not boxes:
            # 尝试 detect
            result = worker.detect(image, [query], generation_mode=mode)
            boxes = _parse_boxes_from_la(result.get("answer", result), width, height)

        if not boxes:
            instances.append(
                {
                    "entity_id": eid,
                    "instance_id": f"{eid}#0",
                    "label": query,
                    "bbox": [0, 0, 0, 0],
                    "fields": {f["name"]: None for f in ent.get("fields", [])},
                    "raw_text": "",
                    "confidence": 0.0,
                    "needs_review": True,
                }
            )
            continue

        for i, bbox in enumerate(boxes):
            crop_box = expand_bbox(bbox, width, height, 0.15)
            crop = image.crop(tuple(crop_box))
            raw_text, conf = _ocr_read(ocr, crop)
            fields = _heuristic_fields(raw_text, ent)
            needs = conf < conf_th or any(
                f.get("required") and not fields.get(f["name"]) for f in ent.get("fields", [])
            )
            instances.append(
                {
                    "entity_id": eid,
                    "instance_id": f"{eid}#{i}",
                    "label": query,
                    "bbox": bbox,
                    "fields": fields,
                    "raw_text": raw_text,
                    "confidence": conf,
                    "needs_review": needs,
                }
            )

    return {"backend": "locateanything_ocr", "instances": instances}
