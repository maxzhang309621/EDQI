# 算法方案：尺寸检测准确率 + 倾斜框

依据：`docs/pipeline/01-architecture.md` v2.0（已确认）

## 步骤 A1：OBB 精修

- 选定算法：框内二值墨迹 →（可选）最大连通域 → **OpenCV `minAreaRect`** → 四点 `boxPoints`
- 选型理由：工程图线划清晰；不必另训 RRPN；cv2 5.x 已在环境中可用
- 候选：1. minAreaRect（首选）2. 仅 ink AABB 收紧（退化）3. VLM 直接出四点（不稳，备选）
- 参考资料：[R2CNN](https://arxiv.org/pdf/1706.09579)（倾斜框动机）；OpenCV `minAreaRect` 文档
- 接口：`(image, bbox_xyxy) → {bbox, quad, angle}`；失败则退回原 AABB
- 依赖：`opencv-python` / 现有 `cv2`

## 步骤 A2：旋转 IoU + NMS

- 选定算法：**多边形裁剪求交面积的 Skew-IoU** + prefer_smaller NMS（同 R2CNN Inclined-NMS 思路）
- 选型理由：邻接斜字 AABB-IoU 虚高；旋转 IoU 才能正确保留；纯几何可单测
- 参考资料：[RRPN Skew IoU](https://arxiv.org/pdf/1703.01086)；[R2CNN Inclined NMS](https://arxiv.org/pdf/1706.09579)
- 接口：`iou_quad(q1,q2)`；`nms_instances(..., use_quad=True, prefer_smaller=True)`
- 依赖：numpy；可用 `shapely` 若已装，否则用 cv2.intersectConvexConvex / 自实现

## 步骤 A3：旋转裁剪 Pass2

- 选定算法：按 `angle`/`quad` **仿射拉正**到水平再送 VLM（扩展既有 deskew）
- 选型理由：减少邻字进入 crop；与现有 orientation retry 兼容
- 接口：`warp_quad_crop(image, quad, pad) → PIL.Image`
- 依赖：cv2.warpAffine / getPerspectiveTransform

## 步骤 A4：准确率过滤

- 选定算法：
  1. OBB-NMS（A2）
  2. 规范化 `text` 后，同 parent 内完全相同 text + 中心距过近 → 合并
  3. 可配置丢弃「弱结果」（无数字且无 dim_kind）
- 选型理由：多尺度/多 tile 主因是重复而非漏检（用户称已检出大量目标）
- 接口：`filter_dimension_accuracy(instances, cfg) → instances`

## 步骤 A5：渲染集成

- 选定算法：PIL `polygon` 画 `quad`；无 quad 时回退矩形
- 依赖：Pillow

## 候选尝试记录

| 步骤 | 候选算法 | 状态 | 失败原因 |
|------|----------|------|----------|
| A1 | minAreaRect | 采用 | — |
| A1 | VLM 直接四点 | 未试 | 优先后处理稳妥 |
| A2 | AABB-NMS only | 否决（主路径） | 斜字误杀邻框 |
| A2 | Skew-IoU NMS | 采用 | — |

## 任务分配

| 任务 ID | 实现内容 | 关联步骤 |
|---------|----------|----------|
| T1 | `pipeline/oriented_box.py`：ink→OBB、quad IoU、warp crop | A1–A3 |
| T2 | NMS/准确率过滤接入 `perceive_utils` + Pass2 用 warp | A2–A4 |
| T3 | `render_report` 画多边形；config；单测；缓存指纹 | A5 |
