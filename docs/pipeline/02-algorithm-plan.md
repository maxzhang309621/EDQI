# 算法方案：粗线四边界生成部件 raw 框（抑幽灵）

依据：`docs/pipeline/01-architecture.md` v4.0（已确认）

## 步骤 C1：粗线掩膜

- 选定算法：**二值墨迹 + 形态学 OPEN（核≈粗线宽）保留粗笔画 + CLOSE 补缝**
- 选型理由：ISO 工程图粗/细线宽比 ≥2:1；与项目既有 `tighten_bbox_to_thick_outline` 同源、零新依赖；SWT（CVPR 文本检测）对 CAD 实线过重且实现复杂，作为失败时再试候选
- 候选：1. OPEN 线宽筛选（首选）2. Stroke Width Transform 阈值掩膜 3. 水平/垂直形态学线提取后并集
- 参考资料：
  - [OpenCV Morphological Operations](https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/)
  - [OpenCV Extract horizontal/vertical lines](https://docs.opencv.org/4.6.0/dd/dd7/tutorial_morph_lines_detection.html)
  - [Line Width Recovery in Engineering Drawings](https://doi.org/10.1515/rput-2016-0015)（粗细线宽语义）
  - 候选：[SWT CVPR](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/201020CVPR20TextDetection.pdf) / [mypetyak/StrokeWidthTransform](https://github.com/mypetyak/StrokeWidthTransform)
- 接口：`extract_thick_strokes(image, *, ink_threshold, thick_min_width) → np.ndarray[uint8] thick_mask`（全页，0/255）
- 依赖：opencv-python-headless、numpy（已有）

## 步骤 C2：四边界成框

- 选定算法：**`connectedComponentsWithStats` 对粗线掩膜取显著连通域 → 各域轴对齐外接框（四边=粗线外缘 min/max）**；可选对近邻域做轻量合并（中心距/外扩 IoU）
- 选型理由：直接对应「以粗线为四边界」；OpenCV 成熟；无需学习模型
- 候选：1. CC + AABB（首选）2. `findContours` + `boundingRect` 3. 外轮廓 approxPolyDP 后再 AABB
- 参考资料：[OpenCV Connected Component Labeling](https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)；[Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- 接口：`propose_boxes_from_thick_boundaries(thick_mask, *, min_area, min_side, pad) → list[{bbox, score}]`
- 依赖：同上

## 步骤 C3：反幽灵闸门

- 选定算法：**规则引擎流水线**（按序）：
  1. 面积：`min_side` / `max_area_frac`（抑图框全页幽灵）
  2. **四边粗线证据**：在 bbox 四条边带（宽≈`thick_min_width`）上统计粗线像素，至少 `min_side_evidence`（默认 3）边有支撑
  3. **墨迹密度**：框内粗线像素占比 ≥ `min_ink_density`（抑空白漂移框）
  4. **表格冲突**：`exclude_tables` 时中心落表内或 IoU 过高则丢弃
  5. **IoU-NMS**：按 score（边证据数×密度）降序，`box_nms_iou` 抑制重复
- 选型理由：幽灵框多为无几何支撑的假候选；规则可单测、可解释；Supervision 式 NMS 逻辑可自实现（避免新依赖）
- 候选：1. 上述规则（首选）2. 仅 NMS+面积 3. 形态学梯度边带匹配加强
- 参考资料：[IoU and NMS](https://supervision.roboflow.com/0.28.0/detection/utils/iou_and_nms/)；既有 `view_regions.shrink_region_away_from_tables`
- 接口：`filter_ghost_boxes(candidates, thick_mask, *, page_wh, table_bboxes, cfg) → list[{bbox, label?}]`
- 依赖：numpy

## 步骤 C4：接入与回退

- 选定算法：配置 `propose_mode`：
  - `thick_boundary`（默认）：`extract → propose → filter` 生成 views；空结果且 `fallback_vlm=true` 时回退 v3 `locate_drawing_views` VLM 路径；再空则现有 grid
  - `vlm_then_tighten`：保留 v3 行为
- 接口：在 `locate_drawing_views` 或 `perceive_utils` 部件路径前置几何提案；notes 含 `view_propose_mode=thick_boundary` / `view_locate_source=thick|vlm|empty`
- 配置默认值建议：
  - `propose_mode: thick_boundary`
  - `min_side_evidence: 3`
  - `min_ink_density: 0.002`
  - `box_nms_iou: 0.45`
  - `exclude_tables: true`
  - `fallback_vlm: true`

## 步骤 C5：指纹与测试

- 指纹 `_view_regions_cache_fp` 升 `v: 6`，增加：`propose_mode`、`min_side_evidence`、`min_ink_density`、`box_nms_iou`、`exclude_tables`、`fallback_vlm`
- 单测覆盖 C1–C3 合成图 + 接入 notes 可观测

## 候选尝试记录

| 步骤 | 候选算法 | 状态 | 失败原因 |
|------|----------|------|----------|
| C1 | OPEN 粗线筛选 | 采用 | — |
| C1 | SWT | 未试 | 过重；仅当 OPEN 漏检升级 |
| C2 | CC + AABB | 采用 | — |
| C3 | 边证据+密度+NMS | 采用 | — |

## 任务分配

| 任务 ID | 实现内容 | 关联步骤 | 参考资料 |
|---------|----------|----------|----------|
| T1 | `extract_thick_strokes` + 单测（细线抑制） | C1 | OpenCV morph ops |
| T2 | `propose_boxes_from_thick_boundaries` + 单测（两分离粗矩形→2 框贴边） | C2 | CC with stats |
| T3 | `filter_ghost_boxes` + 单测（空白假框丢弃、NMS、表冲突） | C3 | IoU/NMS |
| T4 | `propose_mode` 接入 locate/部件路径 + config + 指纹 v6 + 冒烟 | C4–C5 | 既有 `view_regions` / `perceive_utils` |
