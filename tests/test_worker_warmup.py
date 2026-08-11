"""P0：OCR 单例与 worker warmup 接口。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import ocr_engine
from pipeline.worker import warmup_perception


def test_rapid_ocr_singleton_reused():
    sentinel = object()
    prev = ocr_engine._RAPID_OCR
    try:
        ocr_engine._RAPID_OCR = sentinel
        assert ocr_engine.get_rapid_ocr() is sentinel
        assert ocr_engine.get_rapid_ocr() is sentinel
    finally:
        ocr_engine._RAPID_OCR = prev


def test_warmup_perception_skips_vlm_for_mock():
    with patch("pipeline.worker.warmup_rapid_ocr", return_value=True):
        with patch("pipeline.worker.warmup_qwen_vl") as vw:
            out = warmup_perception(backend="mock")
    assert out["ocr"] is True
    assert out["backend"] == "mock"
    vw.assert_not_called()
