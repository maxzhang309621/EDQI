# 实现日志：尺寸准确率 + 倾斜框

- 日期：2026-08-14
- 任务：T1–T3（见 `02-algorithm-plan.md`）
- 实现者：algorithm-implementer

## 实现内容

| 任务 | 变更 |
|------|------|
| T1 | 新增 `pipeline/oriented_box.py`：`refine_aabb_to_obb` / `iou_quad` / `warp_quad_crop` / `apply_obb_to_instances` |
| T2 | `nms_instances(..., use_quad)`；`filter_dimension_accuracy`；Pass1 后 OBB 精修；Pass2 `warp_quad_crop`；finalize 路径 Skew-NMS + 准确率过滤 |
| T3 | `configs/default.yaml` 增加 `dimension_obb` / `dimension_accuracy`；缓存指纹；`build_facts` 透传 `quad`；`render_report` 画多边形；`opencv-python-headless` 入 requirements；`tests/test_oriented_box.py` |

## 冒烟结果

```text
pytest tests/test_oriented_box.py → 5 passed
python smoke (OBB→facts→render polygon) → smoke_ok
```

## 遗留

- 全量 VLM 图纸回归需在 system-test 用真实图验证邻框误杀是否下降（单测已覆盖合成邻接斜字场景）
- `tests/test_m4_hardening.py` 因 `_bootstrap` 导入问题与本任务无关，未纳入冒烟
