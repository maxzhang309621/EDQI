# 经验总结：EDQI 部件分区 VLM 属性识别

## 算法选型结论

| 步骤 | 最终采用 | 被否决的候选 | 否决原因 |
|------|----------|--------------|----------|
| Pass0 定位 | Qwen3-VL JSON 视图框+label | 墨迹连通域 | 尺寸线/剖面线碎裂，不稳定 |
| 识别区 | 比例外扩 + 单边最大剩余扣表 | 同时裁 X/Y | 上方有效区易被裁没 |
| 保底 | make_tiles 网格 | — | Pass0 空结果时必要 |
| 关联 | parent_id + component 实例 | 仅嵌套 attributes[] | 与现有 facts/schema 兼容成本最低 |

## 踩坑记录

1. **先写代码后补流水线**：本轮 Cursor available_skills 未注入本地 `dev-*` skill，导致跳过文档卡点；后续需保证 personal skills 对 Agent 可见，或在开干时显式「按 pipeline-coordinator」。
2. **缓存指纹**：部件分区启用后必须把 view_regions 配置写入 PerceptionCache plan_key，否则会命中旧网格结果。
3. **reindex 破坏关联**：全局重编号会改写 `component#i`，导致 parent_id 悬空；对 component 保留预设 instance_id。
4. **run.yaml tables_only**：注释写 null、文件写 false，与单测冲突；语义应以「null=不覆盖」为准。
5. **空 active 规则**：关闭 NUM_TEXT 后 `run()` 不能再强制要求至少一条 active 规则，应允许仅 drawing_parse。

## 可复用组件

| 组件 | 路径 |
|------|------|
| 视图外扩/扣表/回退 | `pipeline/view_regions.py` |
| Pass0 定位 | `pipeline/perceive_qwen_vl.py::locate_drawing_views` |
| 分区感知入口 | `pipeline/perceive_utils.py` view_regions 分支 |
| 部件框渲染 | `pipeline/render_report.py::_draw_view_region_boxes` |
| 流水线文档模板 | `docs/pipeline/00-state.md` 等 |

## 流程改进建议

1. 将 `dev-architect` / `pipeline-coordinator` 等加入 Cursor Agent 默认可视 skills，避免「磁盘有、会话无」。
2. 架构卡点可通过「用户批准 Cursor plan」视为等价确认，但须在 `01-architecture.md` 显式记录确认来源。
3. 系统测默认 `--ignore` 列出已知坏测（如 m4_hardening），或先修 `_bootstrap` 依赖，避免误报阻断。
4. 功能合入后强制提醒：清 `work_dirs/cache/perception` 与 `tiles`。
