# 流水线状态

- 项目：EDQI — 标题栏表格定位优化 + 附属表分离
- 当前阶段：已完成
- 当前负责人：—
- 阻塞原因：无
- 循环计数：L1 修复 1/3 ｜ L2 重新选型 0 次
- 最近更新：2026-08-14
- 稳定节点：`pipeline-pass-2026-08-14-title-block-tables`

## 历史时间线
| 时间 | 事件 | 详情 |
|------|------|------|
| 2026-08-14 | 需求变更 | 优化表格定位；物料上方其它表分离（bbox_only）；主表/物料字段不变 |
| 2026-08-14 | 架构确认 | 01 v4.0 |
| 2026-08-14 | 算法方案 | 02：VLM+形态学线网 |
| 2026-08-14 | 实现 | A1–A6；提交 c9a8118 |
| 2026-08-14 | L1 修复 | aux 映射进 facts.tables；d76c6cd |
| 2026-08-14 | 系统测试通过 | 99 pytest；run_demo；打 tag；05 总结 |

## 产物清单
- [x] 01-architecture.md（用户已确认？是）
- [x] 02-algorithm-plan.md
- [x] 03-implementation-log.md
- [x] 04-test-report.md
- [x] 05-retrospective.md
