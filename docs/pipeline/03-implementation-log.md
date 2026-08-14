# 实现日志：部件粗边框识别 + 扩展收紧

- 日期：2026-08-14
- 任务：T1–T3（见 `02-algorithm-plan.md`）
- 实现者：algorithm-implementer

## 实现内容

| 任务 | 变更 |
|------|------|
| T1 | `view_regions.tighten_bbox_to_thick_outline` / `tighten_view_bbox`；单测粗框+细线噪声 |
| T2 | Pass0 提示强调粗实线；`locate_drawing_views` 走 thick_outline；`expand_ratio` 0.28→0.18 |
| T3 | `_view_regions_cache_fp` v5 含 `tighten_mode` / `thick_min_width` |

## 冒烟结果

```text
pytest tests/test_view_regions.py → 13 passed
run_demo.py → passed=True
pytest tests/ --ignore=test_m4_hardening → 91 passed
```

## 遗留

- 真实大图线宽差异大时，可调 `thick_min_width`（默认 3）；过强则回退全墨迹收紧
