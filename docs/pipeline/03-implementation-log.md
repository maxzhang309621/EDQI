# 实现日志：粗线四边界生成部件 raw 框（抑幽灵）

## 任务完成情况

| 任务 ID | 实现内容 | 状态 |
|---------|----------|------|
| T1 | `extract_thick_strokes` + `_binarize_ink`（灰底自适应阈值；OPEN 过猛降核） | 完成 |
| T2 | `propose_boxes_from_thick_boundaries`（CC 外接框） | 完成 |
| T3 | `filter_ghost_boxes`（四边证据/密度/表冲突/IoU-NMS） | 完成 |
| T4 | `locate_views_from_thick_boundaries`；`locate_drawing_views` 默认 `thick_boundary`；config；指纹 v6 | 完成 |

## 主要改动文件

- `pipeline/view_regions.py`：C1–C3 纯函数 + `locate_views_from_thick_boundaries`
- `pipeline/perceive_qwen_vl.py`：`propose_mode` 接入；VLM 抽为 `_locate_drawing_views_vlm` 兜底
- `pipeline/perceive_utils.py`：`_view_regions_cache_fp` → v6
- `configs/default.yaml`：新增 propose/反幽灵配置
- `tests/test_view_regions.py`：新增 4 例单测

## 冒烟结果

```
pytest tests/test_view_regions.py → 16 passed
pytest tests/ --ignore=tests/test_m4_hardening.py → 95 passed

demo_drawing locate_drawing_views:
  view_propose_mode=thick_boundary
  thick_boundary_raw=2 → kept=2
  view_locate_source=thick
  boxes: [80,80,505,405], [650,650,1155,825]（均 side_evidence=4）
```

## 踩坑

- `demo_drawing` 背景灰度 244，固定 `ink_threshold=245` 会整页变墨迹 → `_binarize_ink` 饱和时改用 `P90-20`
- 图中「粗线」仅约 2px，OPEN 核强制 ≥3 会消掉轮廓 → 降核回退保留 ≥8% 墨迹

## 遗留

- `tests/test_m4_hardening.py` 收集期缺 `_bootstrap`（与本任务无关的既有问题）
