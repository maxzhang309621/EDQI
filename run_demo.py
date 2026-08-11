"""生成示例图并用 mock 后端跑一遍质检。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from pipeline.run import run


def make_sample() -> Path:
    path = ROOT / "data" / "samples" / "demo_drawing.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 850), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([650, 650, 1150, 820], outline=(20, 20, 20), width=2)
    draw.text((670, 670), "图号 A-1001", fill=(0, 0, 0))
    draw.text((670, 710), "材料 45钢", fill=(0, 0, 0))
    draw.text((670, 750), "名称 示例零件", fill=(0, 0, 0))
    draw.rectangle([80, 80, 500, 400], outline=(80, 80, 80), width=2)
    draw.text((100, 100), "EDQI demo drawing", fill=(40, 40, 40))
    img.save(path)
    return path


def main() -> None:
    sample = make_sample()
    print("sample:", sample)
    result = run(sample, perception_backend="mock", rule_set_id="demo")
    print(f"passed={result['passed']} findings={len(result['findings'])}")
    for f in result["findings"]:
        print(f" - [{f['severity']}] {f['rule_id']}: {f['message']}")
    print("artifacts:")
    for k, v in result.get("artifacts", {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
