# 算法方案：部件粗边框识别 + 扩展收紧

依据：`docs/pipeline/01-architecture.md` v3.0（已确认）

## 步骤 B1：粗轮廓收紧

- 选定算法：二值墨迹 → **形态学 OPEN（核≈粗线宽）保留粗笔画** → CLOSE 补缝 → 显著连通域并集的轴对齐外接框；失败则回退现有 `tighten_bbox_to_ink`
- 选型理由：CAD/线稿中粗轮廓与细标注线宽可分；OpenCV 已依赖；无需再训检测器
- 候选：1. OPEN 线宽筛选（首选）2. 形态学梯度仅边带 3. 仅全墨迹 AABB（退化）
- 参考资料：[OpenCV Morphological Operations](https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/)；[Contour Features / boundingRect](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- 接口：`tighten_bbox_to_thick_outline(image, bbox, *, thick_min_width, ink_threshold, pad) → bbox`
- 依赖：opencv-python-headless、numpy

## 步骤 B2：Pass0 提示 + 接入

- 选定算法：强化 `_VIEW_LOCATE_PROMPT`（强调粗实线/object line）；`locate_drawing_views` 按 `tighten_mode=thick_outline` 调用 B1
- 接口：配置 `perception.view_regions.tighten_mode`

## 步骤 B3：扩展收紧

- 选定算法：默认 `expand_ratio: 0.28 → 0.18`（约收紧 35% 外扩量）
- 理由：用户明确要求扩展框收紧；补扫网格仍在

## 步骤 B4：测试与指纹

- 单测：粗框+细线噪声场景；expand 比例；cache fp 字段
- 指纹：`_view_regions_cache_fp` 增加 `tighten_mode` / `thick_min_width`

## 候选尝试记录

| 步骤 | 候选算法 | 状态 | 失败原因 |
|------|----------|------|----------|
| B1 | OPEN 粗线筛选 | 采用 | — |
| B1 | 仅全墨迹 | 回退路径 | 会被细线撑大 |

## 任务分配

| 任务 ID | 实现内容 | 关联步骤 |
|---------|----------|----------|
| T1 | `view_regions.tighten_bbox_to_thick_outline` + 单测 | B1 |
| T2 | `locate_drawing_views` + prompt + config expand | B2–B3 |
| T3 | 缓存指纹 + 冒烟 | B4 |
