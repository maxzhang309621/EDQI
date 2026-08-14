# 测试报告：尺寸准确率 + 倾斜框

- 日期：2026-08-14
- 结论：**系统测试通过**
- 提交：`39e2ff7`
- 循环：L1 0/3 · L2 0

## 验收对照（01 架构）

| 步骤 | 用例类型 | 结果 | 说明 |
|------|----------|------|------|
| A1 OBB 精修 | 正常 | PASS | `test_refine_aabb_to_obb_tilted_ink`：倾斜墨迹 loose AABB 面积 > 精修后 |
| A1 | 边界 | PASS | 墨迹不足返回 None（由 apply 退回原 AABB） |
| A2 Skew-NMS | 正常 | PASS | `test_nms_skew_keeps_adjacent_tilted_neighbors`：AABB-NMS 杀 1，Skew-NMS 保留 2 |
| A2 | 边界 | PASS | `test_iou_quad_identical_and_disjoint` |
| A3 warp Pass2 | 正常 | PASS | `test_warp_quad_crop_and_apply_obb` |
| A4 准确率 | 正常 | PASS | 弱结果剔除 + 同文近距去重 |
| A5 集成 | 正常 | PASS | facts 透传 quad；render 画 polygon；mock 全链路 |

## 整体运行

| 项 | 命令/入口 | 结果 |
|----|-----------|------|
| Demo 入口 | `python run_demo.py` | passed=True，产物写出正常 |
| 单元回归 | `pytest tests/ --ignore=test_m4_hardening` | **89 passed** |
| 已知无关失败 | `test_m4_hardening` | 收集期 `ModuleNotFoundError: _bootstrap`（既有问题，非本任务） |

## 异常/边界补充

- 非法/空框：OBB 失败不抛错，保留原实例
- 无 `quad`：NMS 回退 AABB；渲染回退矩形

## 备注

真实大图 VLM 效果需清缓存后跑业务图验证：`rm -rf work_dirs/cache/perception work_dirs/cache/tiles`
