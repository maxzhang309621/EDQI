# 测试报告：部件粗边框识别 + 扩展收紧

- 日期：2026-08-14
- 结论：**系统测试通过**
- 循环：L1 0/3 · L2 0

## 验收对照（01 架构 v3.0）

| 步骤 | 用例 | 结果 |
|------|------|------|
| B1 粗轮廓收紧 | `test_tighten_thick_outline_ignores_thin_peripheral_lines` | PASS |
| B1 回退全墨迹 | `test_tighten_bbox_to_ink` | PASS |
| B3 扩展收紧 | `test_expand_ratio_tighter_default_behavior`（0.18 < 0.28） | PASS |
| B4 回归 | 全量 pytest（忽略无关 m4） | **91 passed** |
| 入口 | `run_demo.py` | passed=True |

## 配置检查

- `expand_ratio: 0.18`
- `tighten_mode: thick_outline`
- `thick_min_width: 3`

## 备注

清缓存后再跑业务图：`Remove-Item -Recurse -Force work_dirs\cache\perception, work_dirs\cache\tiles`
