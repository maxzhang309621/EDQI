# 系统测试报告：标题栏表格定位 + 附属表分离

- 日期：2026-08-14
- 结果：**系统测试通过**
- 依据：`01-architecture.md` v4.0 验收标准 T1–T6

## 测试项清单

| ID | 类型 | 验收点 | 结果 |
|----|------|--------|------|
| T1 | 正常 | plan 加载 `bbox_only` / aux；material/main 契约不变 | 通过 |
| T2 | 正常 | 线网/间隙可拆出上方附属框；无附属时不硬造 aux | 通过（单测） |
| T3 | 正常/边界 | Product 分界保留；`aux.y2 ≤ material.y1` | 通过 |
| T4 | 正常 | Pass2 跳过 bbox_only；material above_cells / main 字段 mock 仍有值 | 通过 |
| T5 | 正常 | aux 进入 `facts.tables` + 排除区/渲染路径 | 通过（L1 修复后） |
| T6 | 回归 | 全量单元测试 | 通过 99 passed |
| E1 | 入口 | `run_demo.py` mock 端到端 | 通过 |
| B1 | 边界 | 无 Product OCR → 几何回退路径仍可调用 | 通过（既有+扩展 notes） |
| A1 | 异常 | 非法 read_mode 回退 cell_content | 通过 |

## L1 修复记录

| 轮次 | 问题 | 修复 |
|------|------|------|
| 1 | `aux_table` 未进 `ENTITY_FACTS_MAP`，落入顶层 `facts.aux_table` 而非 `tables` | 映射 + `build_facts` 对 `aux_table*` / parse_kind=table 兜底 |

## 命令与结果

```
python -m pytest tests/ --ignore=tests/test_m4_hardening.py -q
→ 99 passed

python run_demo.py
→ passed=True；facts.tables 含 aux / material / main
```

（`test_m4_hardening.py` 既有 `_bootstrap` 导入失败，与本轮无关。）

## 稳定节点

- 提交：`d76c6cd`（L1 修复）/ `c9a8118`（功能）
- tag：`pipeline-pass-2026-08-14-title-block-tables`
