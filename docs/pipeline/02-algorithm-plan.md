# 算法方案：EDQI 部件分区 VLM 属性识别

依据：`docs/pipeline/01-architecture.md` v1.0（用户已确认）

## 步骤 S1：关闭数字重叠识别

- 选定算法：配置开关 + 规则状态机（draft / active）
- 选型理由：无需新算法；与现有 `load_rules(only_active=True)` 一致，改动面最小且可逆
- 参考资料：项目内 `engines/__init__.py`（only_active 过滤）、`pipeline/run.py` perceive 双后端条件
- 接口约定：配置生效后默认 plan 无 OCR 重叠实体；evaluate 不跑 NUM_TEXT
- 依赖：PyYAML / 现有配置加载

## 步骤 S2：Pass0 部件定位

- 选定算法：Qwen3-VL Instruct 单次 JSON 定位（提示词约束几何视图 + 短 label）
- 选型理由：项目已集成同一 VLM；与表格/尺寸同一权重，避免引入第二检测器；工程图开放域检测开源模型（如通用 YOLO）对「视图」语义弱
- 候选（调试用）：
  1. Qwen3-VL JSON bbox（首选，已采用）
  2. 固定网格直接当区（保底，S3 fallback）
  3. 墨迹连通域（否决：尺寸线/剖面线碎裂连通域）
- 参考资料：项目 `pipeline/perceive_qwen_vl.py` 既有 `_vlm_generate` / `norm_bbox_to_pixels`；Qwen2/3-VL 官方多模态指令格式
- 接口约定：Image → `[{bbox:[x1,y1,x2,y2], label:str}, ...]` + notes
- 依赖：transformers / 本地 `models/Qwen3-VL-8B-Instruct`；`locate_max_new_tokens≈512`

## 步骤 S3：识别区几何

- 选定算法：比例外扩 + 与表框单边最大剩余裁切 + 面积阈值拆分/回退 `make_tiles`
- 选型理由：实现简单、可单测、与可视化表框一致；无需学习式分割
- 参考资料：项目 `pipeline/perceive_utils.make_tiles`；轴对齐 IoU / 裁切启发式
- 接口约定：views + table_bboxes → specs(`part_id,label,bbox_raw,bbox_expanded,regions[]`) 或 grid fallback
- 依赖：无新增第三方库

## 步骤 S4：分区属性识别 + 关联

- 选定算法：既有 two-pass dimension_marks VLM + `parent_id` 外键关联 + NMS + 半页面积过滤
- 选型理由：复用已验证的尺寸 Pass2/斜向 deskew；关联用显式 parent_id，兼容 `build_facts`
- 参考资料：`pipeline/perceive_qwen_vl.perceive_qwen_vl`；`engines.ENTITY_FACTS_MAP`（component→components）
- 接口约定：每 region crop 识别 → shift 回全图 → `parent_id=component#i`；同时产出 component 实例
- 依赖：同 S2 VLM

## 步骤 S5：可视化

- 选定算法：PIL ImageDraw 双框叠画（先 expanded 后 raw）
- 选型理由：与现有 table/attribute 渲染一致
- 参考资料：`pipeline/render_report.py`
- 接口约定：`facts.components` → 黄 raw / 红 expanded；配置 `render.show_view_regions`
- 依赖：Pillow

## 步骤 S6：测试

- 选定算法：pytest 单元 + 回归
- 选型理由：项目既有测试基建
- 接口约定：几何/关联/渲染/规则状态断言
- 依赖：pytest

## 候选尝试记录

| 步骤 | 候选算法 | 状态 | 失败原因 |
|------|----------|------|----------|
| S2 | Qwen3-VL JSON 定位 | 采用 | — |
| S2 | 纯墨迹连通域 | 否决 | 工程图线划碎裂，不稳定 |
| S2 | 固定网格 | 采用（fallback） | 主路径失败时保底 |
| S3 | 双边同时收缩扣表 | 否决 | 上方有效区易被裁没；改为单边最大剩余 |

## 任务分配

| 任务 ID | 实现内容 | 关联步骤 | 参考资料 |
|---------|----------|----------|----------|
| T1 | 配置关闭重叠 + NUM_TEXT draft + run 允许空规则 | S1 | 01-architecture S1 |
| T2 | `view_regions.py` 几何 + `locate_drawing_views` | S2–S3 | 本方案 S2/S3 |
| T3 | `perceive_with_cache_and_tiles` 部件分区与 parent_id/component | S4 | 本方案 S4 |
| T4 | `build_facts` 透传 bbox_expanded；`render_report` 部件框 | S5 | 本方案 S5 |
| T5 | 单测与重叠相关断言适配 | S6 | tests/* |
