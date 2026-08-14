# 测试报告：粗线四边界生成部件 raw 框（抑幽灵）

- 日期：2026-08-14
- 依据：`docs/pipeline/01-architecture.md` v4.0
- 结果：**系统测试通过**

## 验收对照

| 步骤 | 用例 | 结果 |
|------|------|------|
| C1 粗线掩膜 | 粗矩形+细线：细线像素被抑制、粗边保留 | PASS |
| C2 四边界成框 | 两分离粗矩形 → 恰好 2 框且贴边 | PASS |
| C3 反幽灵 | 空白假框丢弃；高 IoU 重复抑制为 1 | PASS |
| C4 接入 | `propose_mode=thick_boundary`；demo → `view_locate_source=thick`、2 框 | PASS |
| C5 回归 | `pytest tests/ --ignore=test_m4_hardening` → 95 passed | PASS |
| 入口冒烟 | `run_demo.py` → passed=True | PASS |
| 边界 | 全白图 → 0 框 | PASS |

## demo_drawing 观测

```
view_propose_mode=thick_boundary
thick_boundary_raw=2 → kept=2
view_locate_source=thick
bbox: [80,80,505,405], [650,650,1155,825]（side_evidence=4）
```

## 已知非阻断问题

- `tests/test_m4_hardening.py` 收集失败（缺 `_bootstrap`），与本变更无关，沿用 ignore。

## 分流

无失败项；不触发 L1/L2/L3。
