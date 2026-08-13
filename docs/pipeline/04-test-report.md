# 测试报告：EDQI 部件分区 VLM 属性识别

- 日期：2026-08-13
- 负责人：system-test-engineer
- 依据：`01-architecture.md` 验收标准 S1–S6
- 结论：**系统测试通过**（本功能范围内；见已知限制）

## 验收对照

| 步骤 | 验收标准 | 结果 | 证据 |
|------|----------|------|------|
| S1 | 关闭重叠配置与 draft 规则 | 通过 | `ACCEPT_S1_S5_CONFIG_OK`；only_active 无 NUM_TEXT |
| S2 | Pass0 API / notes 约定 | 通过（单测级） | `locate_drawing_views` 存在；几何与空结果路径单测 |
| S3 | 外扩/扣表/回退网格 | 通过 | `tests/test_view_regions.py` |
| S4 | parent_id ↔ component | 通过 | `tests/test_component_parent_facts.py`；冒烟 SMOKE_OK |
| S5 | 部件框可视化 | 通过 | `tests/test_render_view_regions.py`；config `show_view_regions=true` |
| S6 | 相关 pytest 回归 | 通过 | 见下 |

## 测试执行

### 单元 / 回归

```text
pytest tests --ignore=tests/test_m4_hardening.py -q
→ 80 passed
```

### 冒烟（implementer）

```text
31 passed（核心相关子集）+ SMOKE_OK view_regions 1 front
```

### L1 修复记录

| 问题 | 根因 | 修复 | 重测 |
|------|------|------|------|
| `test_run_config` 期望 `tables_only is None` | `configs/run.yaml` 写死 `false`，与注释「null=不覆盖」冲突 | 改为 `tables_only: null` | 80 passed |

## 已知限制 / 非阻断

| 项 | 说明 | 分流 |
|----|------|------|
| `tests/test_m4_hardening.py` 收集失败 | `tools/evaluate_gold.py` 依赖缺失 `_bootstrap`（既有问题，与本功能无关） | 不升 L1 至本特性；另案处理 |
| 端到端 `pipeline.run` + 真 VLM | 本机系统测未跑完整图纸推理 | 建议用户清缓存后实机验证 notes：`view_regions:n=` / `view_regions_source=` |

## 循环计数

- L1：1/3（tables_only 配置，已通过）
- L2：0
- L3：无

## 通过后动作

1. 本报告标记「系统测试通过」
2. 打 tag：`pipeline-pass-2026-08-13`
3. 更新 `00-state.md` → 已完成
4. 交接 `dev-retrospective`
