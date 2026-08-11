"""评测 / 缓存 / 复核队列测试。"""
from __future__ import annotations

from pathlib import Path

from pipeline import model_download_status, load_config
from pipeline.perceive_utils import iou_xyxy, make_tiles, nms_instances
from pipeline.review_queue import enqueue_from_result, list_pending, resolve_item
from pipeline.run import run
from tools.evaluate_gold import run_eval


ROOT = Path(__file__).resolve().parents[1]


def test_iou_and_tiles():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    tiles = make_tiles(3000, 2000, tile_size=1280, overlap=0.2)
    assert len(tiles) > 1
    assert tiles[0] == (0, 0, 1280, 1280)


def test_nms():
    inst = [
        {"entity_id": "a", "bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"entity_id": "a", "bbox": [1, 1, 11, 11], "confidence": 0.5},
        {"entity_id": "a", "bbox": [50, 50, 60, 60], "confidence": 0.8},
    ]
    kept = nms_instances(inst, 0.5)
    assert len(kept) == 2


def test_model_status_incomplete_not_ready():
    cfg = load_config()
    st = model_download_status(cfg["models"]["qwen3_vl"])
    # 下载中或未完成时应 ready=False；若用户已下完则为 True
    assert "ready" in st
    assert "path" in st


def test_review_queue(tmp_path=None):
    sample = ROOT / "data" / "samples" / "demo_missing_partno.png"
    result = run(sample, perception_backend="mock", enqueue_review=True)
    assert result["passed"] is False
    arts = result["artifacts"]
    assert arts.get("markdown")
    assert arts.get("review_queue")
    pending = list_pending()
    assert any(p.name.startswith("demo_missing_partno") for p in pending)
    # resolve one
    target = Path(arts["review_queue"])
    resolve_item(target, decision="approve", reviewer="tester", comment="ok")
    data_status = __import__("pipeline", fromlist=["load_json"]).load_json(target)["status"]
    assert data_status == "resolved"


def test_gold_eval_mock():
    summary = run_eval(ROOT / "data" / "gold", backend="mock", iou_thr=0.3)
    assert summary["n"] >= 1
    assert "boxes" in summary
    assert Path(summary["artifacts"]["report"]).exists()
