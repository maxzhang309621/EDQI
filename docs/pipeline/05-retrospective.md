# 经验总结：尺寸准确率 + 倾斜框（OBB）

## 算法选型结论

| 步骤 | 最终采用 | 被否决的候选 | 否决原因 |
|------|----------|--------------|----------|
| A1 OBB | OpenCV `minAreaRect` + 最大连通域 | VLM 直接出四点 | 不稳定、难单测 |
| A2 NMS | Skew-IoU（`intersectConvexConvex`）+ prefer_smaller | 仅 AABB-NMS | 斜字外接框虚高 IoU 误杀邻框 |
| A3 Pass2 | `getPerspectiveTransform` 拉正 | 仅 AABB crop | 邻字易进入 crop |
| A4 准确率 | 弱结果剔除 + 同文近距去重 | 先再训检测器 | 当前主因是重复/假阳而非漏检 |

## 踩坑记录

- PowerShell 不支持 bash heredoc：`git commit` 需用 `@"..."@`
- 邻接斜字单测需保证 AABB-IoU 确实高于阈值，否则断言失败不能说明 Skew-NMS 价值
- 缓存指纹必须纳入 `dimension_obb` / `dimension_accuracy`，否则配置变更仍命中旧结果

## 可复用组件

- `pipeline/oriented_box.py`：AABB→OBB、旋转 IoU、仿射裁剪
- `pipeline/perceive_utils.py`：`nms_instances(use_quad=...)`、`filter_dimension_accuracy`
- `tests/test_oriented_box.py`：邻接斜字 NMS 对照用例

## 流程改进建议

- 系统测可默认 `--ignore` 已知坏掉的无关收集错误（如 `_bootstrap`），避免淹没真实回归信号
- 倾斜框可视化默认开时，建议在报告中注明「多边形=OBB / 矩形=无 quad」以免误读
