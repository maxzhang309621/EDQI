# 流水线状态

- 项目：EDQI — 尺寸检测准确率提升 + 倾斜框（OBB）
- 当前阶段：系统测试
- 当前负责人：system-test-engineer
- 阻塞原因：无
- 循环计数：L1 修复 0/3 ｜ L2 重新选型 0 次
- 最近更新：2026-08-14

## 历史时间线
| 时间 | 事件 | 详情 |
|------|------|------|
| 2026-08-14 | 需求变更 | 大量目标已可检出 → 提准确率；倾斜 AABB 过大误伤邻框 → 支持倾斜矩形 |
| 2026-08-14 | 启动新流水线 | pipeline-coordinator 重置状态；默认：quad+外接 xyxy；准确率=去重+Pass2 |
| 2026-08-14 | 架构/算法 | 01 v2.0、02 方案确认（minAreaRect + Skew-NMS + warp Pass2） |
| 2026-08-14 | 实现完成 | T1–T3 冒烟通过；交接系统测试 |

## 产物清单
- [x] 01-architecture.md（确认来源=完成本任务开发指令）
- [x] 02-algorithm-plan.md
- [x] 03-implementation-log.md
- [ ] 04-test-report.md
- [ ] 05-retrospective.md
