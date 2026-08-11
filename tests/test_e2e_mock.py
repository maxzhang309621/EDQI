"""端到端：mock 感知跑通 ingest→引擎→报告。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run import run


def _make_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 850), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # 简易标题栏
    draw.rectangle([650, 650, 1150, 820], outline=(20, 20, 20), width=2)
    draw.text((670, 670), "图号 A-1001", fill=(0, 0, 0))
    draw.text((670, 710), "材料 45钢", fill=(0, 0, 0))
    draw.text((670, 750), "名称 示例零件", fill=(0, 0, 0))
    draw.rectangle([80, 80, 500, 400], outline=(80, 80, 80), width=2)
    draw.text((100, 100), "示例工程图（演示用）", fill=(40, 40, 40))
    img.save(path)


def test_run_mock(tmp_path=None):
    sample = ROOT / "data" / "samples" / "demo_drawing.png"
    _make_sample(sample)
    result = run(sample, perception_backend="mock", rule_set_id="demo")
    assert "passed" in result
    assert "artifacts" in result
    assert Path(result["artifacts"]["vis"]).exists()
    assert Path(result["artifacts"]["report"]).exists()
    assert Path(result["artifacts"]["facts"]).exists()
