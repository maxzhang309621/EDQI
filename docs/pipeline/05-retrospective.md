# 经验总结：粗线四边界生成部件 raw 框（抑幽灵）

## 算法选型结论

| 步骤 | 最终采用 | 被否决的候选 | 否决原因 |
|------|----------|--------------|----------|
| C1 | OPEN 粗线 + 自适应二值 + 降核回退 | SWT | 过重；CAD 实线用形态学足够 |
| C2 | connectedComponentsWithStats → AABB | approxPolyDP | AABB 已满足四边界需求 |
| C3 | 四边证据 + 密度 + 表冲突 + IoU-NMS | 仅面积过滤 | 无法抑空白幽灵/重复框 |
| C4 | 默认 thick_boundary，VLM 兜底 | 纯 VLM | 易出幽灵框；权重未就绪时不可用 |

## 踩坑记录

- **灰底图纸**：背景 244、阈值 245 → 整页变墨迹。解法：`_binarize_ink` 在墨迹占比 >35% 时改用 `P90−20`。
- **2px「粗线」**：OPEN 核强制 ≥3 会消轮廓。解法：多档核尝试，保留 ≥8% 墨迹的第一档成功结果。
- **空洞矩形密度低**：用相对框面积的粗线占比阈值（默认 0.002），勿用过高阈值。

## 可复用组件

- `pipeline/view_regions.py`：`extract_thick_strokes`、`propose_boxes_from_thick_boundaries`、`filter_ghost_boxes`、`locate_views_from_thick_boundaries`
- 配置开关：`perception.view_regions.propose_mode`

## 流程改进建议

- 线宽相关默认值应相对 DPI/图像短边缩放；合成单测用白底、真实样例用灰底，验收须覆盖两类。
- 入口冒烟（`run_demo` mock）不一定走部件定位；系统测应显式调用 `locate_drawing_views`。
