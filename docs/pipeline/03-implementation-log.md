# 实现日志：EDQI 部件分区 VLM 属性识别

- 日期：2026-08-13
- 负责人：algorithm-implementer（回填）
- 关联方案：`02-algorithm-plan.md` 任务 T1–T5
- 代码提交基线：`a641a1b`（feat: add view region visualization and processing）

## 实现内容

| 任务 | 状态 | 路径 |
|------|------|------|
| T1 关闭重叠 | 完成 | `configs/default.yaml`（auto/dual false）；`rules/library/NUM_TEXT_NO_OVERLAP.json` status=draft；`pipeline/run.py` 允许无 active 规则仅 drawing_parse |
| T2 Pass0+几何 | 完成 | `pipeline/view_regions.py`；`pipeline/perceive_qwen_vl.py::locate_drawing_views` |
| T3 分区+关联 | 完成 | `pipeline/perceive_utils.py::perceive_with_cache_and_tiles`（view_regions 分支、parent_id、component 实例、缓存指纹） |
| T4 Facts+可视化 | 完成 | `pipeline/build_facts.py`（bbox_expanded）；`pipeline/render_report.py`（show_view_regions） |
| T5 单测 | 完成 | `tests/test_view_regions.py`、`test_render_view_regions.py`、`test_component_parent_facts.py`；适配 rule/drawing_parse/dimension 测试 |

## 冒烟测试记录

命令：

```text
pytest tests/test_view_regions.py tests/test_render_view_regions.py tests/test_component_parent_facts.py tests/test_rule_engine.py tests/test_dimension_vlm_backend.py tests/test_drawing_parse_plan.py -q
python -c "<resolve_part_region_specs + reindex + build_facts 冒烟>"
```

结果（2026-08-13）：

- pytest：**31 passed**
- 脚本：`SMOKE_OK view_regions 1 front`

## 遗留问题

- 未在本机对真实图纸跑完整 `pipeline.run`（依赖本地 VLM 权重与耗时）；系统测试阶段以单元/回归为主，端到端需用户环境验证。
- 清缓存提醒：上线后需清 `work_dirs/cache/perception` 与 `tiles`，否则可能命中旧网格缓存。

## 交接

→ system-test-engineer：按 `01-architecture.md` 验收标准做系统测试与回归。
