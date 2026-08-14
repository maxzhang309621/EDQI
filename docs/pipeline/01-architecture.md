# 架构方案：尺寸检测准确率 + 倾斜框

- 版本：v2.0
- 日期：2026-08-14
- 用户确认：是（指令：依照本地开发 skill **完成此任务开发**）
- 前序：v1.0 部件分区 VLM 属性识别

## 需求概述

- **核心目标**
  1. 在已能检出大量尺寸目标的前提下，**提高检测/识读准确率**（降假阳、降重复、提字段正确率）。
  2. 倾斜尺寸若用轴对齐大框，易吞并邻字；引入**倾斜矩形（OBB）**表示，减小邻域干扰，Pass2 按朝向裁剪识读。
- **输入**：现有 VLM Pass1 轴对齐框 + 页图；配置 `perception.dimension_vlm` / `dimension_obb`。
- **输出**：实例含 `bbox`（外接 AABB，兼容旧路径）、`quad`（四点倾斜框）、`angle`；可视化画倾斜多边形；NMS 对倾斜框用旋转 IoU。
- **约束**：不改数字重叠 OCR 超参；不大改表格路径；技术栈保持 Python + Qwen3-VL + Pillow（可选 OpenCV 几何）。

## 可行性结论

**可行**

- 倾斜框：场景文字检测（RRPN / R2CNN）已验证「倾斜框 + 倾斜 NMS」优于 AABB；本仓库可在 VLM AABB 后用墨迹最小外接旋转矩形精修，无需另训检测器。
- 准确率：多尺度/分块导致重复框，可用「更小框优先 NMS + 文本去重 + 弱结果剔除」；Pass2 用旋转裁剪减少邻字污染。

## 模块划分

| 模块 | 职责 | 所需算法/逻辑/架构类型 |
|------|------|------------------------|
| OBB 精修 | AABB→墨迹→最小面积旋转矩形→quad | 连通域/二值化 + 旋转最小外接矩形 |
| 旋转 IoU / NMS | 倾斜框去重，避免 AABB 高 IoU 误杀邻字 | 多边形裁剪 IoU / Skew-NMS |
| 旋转裁剪 Pass2 | 按 OBB 仿射拉正后识读 | 仿射变换 + 既有 VLM Pass2 |
| 准确率后处理 | 重复合并、假阳抑制、字段一致性 | 规则启发式去重 + 弱结果过滤 |
| 可视化 | 画 quad 多边形 | 2D 多边形描边 |

## 数据流与接口约定

```
Pass1 AABB boxes
  → refine_dim_obb(image, box) → {bbox_aabb, quad[4][2], angle}
  → nms_oriented (prefer_smaller)
  → Pass2: warp_crop(quad) → VLM 识读
  → accuracy_filter (弱结果/重复 text)
  → facts.annotations (+quad) → render polygon
```

接口：

| 字段 | 含义 |
|------|------|
| `bbox` | 轴对齐外接框 [x1,y1,x2,y2]（兼容） |
| `quad` | 四点 [[x,y]×4]，顺时针 |
| `angle` | 文本朝向角（度），与 Pass2 deskew 一致 |

## 开发步骤与验收标准

| 步骤 | 描述 | 所需算法类型 | 验收标准 |
|------|------|--------------|----------|
| A1 | OBB 精修纯函数 | 旋转最小外接矩形 | 倾斜墨迹条的 AABB 面积 > OBB 外接 AABB；quad 四点合法 |
| A2 | 旋转 IoU + NMS | Skew-NMS | 两邻接斜条 AABB-IoU 高但 OBB-IoU 低时，二者均可保留 |
| A3 | 旋转裁剪 Pass2 | 仿射拉正 | 裁剪图近水平；邻字少进入 crop |
| A4 | 准确率过滤 | 启发式去重 | 同 text+高 IoU 合并；空/弱 fields 标记或丢弃可配 |
| A5 | 渲染 + 配置 + 单测 | 工程集成 | show_attribute 画多边形；相关 pytest 通过 |

## 风险与外部依赖

- VLM 仍可能给出偏大 AABB；OBB 依赖框内墨迹质量，框内多目标时可能融框——用 prefer_smaller + 连通域最大块缓解。
- OpenCV 若未声明依赖，优先用已有传递依赖或纯 numpy 实现；否则写入 requirements。
- 缓存指纹需纳入 OBB/准确率配置。
