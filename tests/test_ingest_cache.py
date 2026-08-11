"""ingest 复用与感知缓存签名。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from pipeline.ingest import ingest
from pipeline.perceive_utils import PerceptionCache


def test_ingest_reuses_page_without_touching_mtime(tmp_path):
    src = tmp_path / "d.pdf"
    # 用 png 更简单，避免依赖复杂 pdf
    src = tmp_path / "drawing.png"
    img = Image.new("RGB", (100, 80), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([10, 10, 40, 40], outline=(0, 0, 0))
    img.save(src)

    out = tmp_path / "ingest"
    p1, m1 = ingest(src, output_dir=out, max_side=2048, dpi=200)
    t1 = p1.stat().st_mtime
    time.sleep(0.05)
    p2, m2 = ingest(src, output_dir=out, max_side=2048, dpi=200)
    t2 = p2.stat().st_mtime
    assert p1 == p2
    assert m1.get("ingest_sig") == m2.get("ingest_sig")
    assert t1 == t2
    assert m2.get("ingest_sig")


def test_image_signature_stable_with_ingest_sig(tmp_path):
    src = tmp_path / "drawing.png"
    Image.new("RGB", (60, 40), (200, 200, 200)).save(src)
    out = tmp_path / "ingest"
    page, meta = ingest(src, output_dir=out)
    s1 = PerceptionCache.image_signature(page)
    # 即使人为改 mtime，只要 meta.ingest_sig 不变，签名应不变
    time.sleep(0.05)
    page.touch()
    s2 = PerceptionCache.image_signature(page)
    assert s1 == s2
    assert meta.get("ingest_sig")
