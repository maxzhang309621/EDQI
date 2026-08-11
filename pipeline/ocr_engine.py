"""RapidOCR 进程内单例：避免每次感知重建引擎。"""
from __future__ import annotations

from typing import Any

_RAPID_OCR: Any = None


def get_rapid_ocr():
    """返回进程内复用的 RapidOCR 实例。"""
    global _RAPID_OCR
    if _RAPID_OCR is not None:
        return _RAPID_OCR
    from rapidocr_onnxruntime import RapidOCR

    _RAPID_OCR = RapidOCR()
    return _RAPID_OCR


def warmup_rapid_ocr() -> bool:
    """预热 OCR；成功返回 True，未安装则 False。"""
    try:
        get_rapid_ocr()
        return True
    except ImportError:
        return False
