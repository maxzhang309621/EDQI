# 算法方案：标题栏表格定位优化 + 附属表分离

依据：`docs/pipeline/01-architecture.md` v4.0（已确认）

## 步骤 T1：plan 契约扩展（bbox_only / aux）

- 选定算法：**配置归一化扩展** — `drawing_parse_plan._normalize_table_block` 接受 `read_mode ∈ {cell_content, above_cells, bbox_only}`；`section=aux`；`cardinality=zero_or_more`；动态实例 `aux_table` / `aux_table_{i}`
- 选型理由：纯契约变更，零新依赖；与现有 material/main 并存
- 参考资料：现有 `pipeline/drawing_parse_plan.py`；架构 v4.0 接口表
- 接口：yaml / 运行时 dict → plan entity（含 `read_mode=bbox_only`）
- 依赖：无新增

## 步骤 T2：多表定位 + 角色分类

- 选定算法（首选）：**既有 Qwen-VL Pass1 多实例定位提示增强** + **标题栏 ROI 内形态学横竖线网连通域拆分**（经典 bordered-table 线网法）
  1. Pass1 提示：明确「物料上方若有其它表须单独输出 bbox；禁止并入 material_table」；允许输出 `aux_table` 或多框
  2. 后处理：在 material∪main∪粗框的外接 ROI（略上扩）上做 HV-line morphology → 交叉成表候选 → 按 y 排序
  3. 角色：OCR 找 Product → 其下=main，紧邻其上含 Article/Material/Volume 等=material，再上方剩余=aux*
- 选型理由：无训练、兼容现栈（已有 RapidOCR + OpenCV + VLM）；工程图标题栏多为有线表，线网法成熟
- 候选（按优先级）：
  1. **VLM 多实例 + 形态学线网拆分**（首选）
  2. **仅几何拆分**：若 material 框过高/含多段水平间隙，按水平间隙切条再分类（无线网时兜底）
  3. **YOLO/RT-DETR 标题栏检测**（需标注与训练，本轮不默认启用）— 参考 [engineering-doc-parser](https://github.com/K-Hooshanfar/engineering-doc-parser)、[HF RT-DETR title-block](https://huggingface.co/hsarfraz/eng-drawing-title-block-bill-of-material-extractor)
- 参考资料：
  - Multi-Type-TD-TSR（有线表结构，形态学线）[arXiv:2105.11021](https://arxiv.org/pdf/2105.11021) / [GitHub](https://github.com/Psarpei/Multi-Type-TD-TSR)
  - OpenCV morphological line extraction（横核/竖核 OPEN）
  - 既有 `split_title_block_by_product` / `refine_material_main_boxes`
- 接口：`(page_image, seed_boxes, plan) → boxes[+aux], notes`
- 依赖：opencv-python-headless、numpy、Pillow、既有 RapidOCR / Qwen-VL

## 步骤 T3：分界 refine（aux 不侵入 material）

- 选定算法：**Product OCR 硬分界（保留）** + **垂直栈几何约束**
  - material/main：沿用 `apply_product_split_to_boxes` / `enforce_material_main_boundary`
  - aux：强制 `aux.y2 ≤ material.y1 - gap`；与 material x 重叠过大且纵向重叠则裁切或丢弃劣质 aux
  - 无 Product：几何回退，按 y 排序取最下=main、次下=material、其余=aux
- 选型理由：Product 锚点已在生产路径验证；几何约束实现简单可单测
- 接口：扩展 `refine_material_main_boxes` → `refine_title_block_tables`（或同函数扩展）
- 依赖：无新增

## 步骤 T4：Pass2 门控

- 选定算法：**策略门控** — `read_mode == bbox_only` 或 `section == aux` 时跳过字段识读，只写元数据 fields（entity_id/section/read_mode）
- 选型理由：满足「暂不识别内部」；material/main 路径零改业务逻辑
- 接口：perceive Pass2 / single-pass 分支前置 return
- 依赖：无新增

## 步骤 T5：渲染 + 尺寸排除

- 选定算法：复用 `collect_table_bboxes` / `build_table_exclude_regions` / `_draw_table_boxes`；aux 走 `table_other` 配色；`is_table_entity_id` 识别 `aux_table*`
- 参考：`configs/default.yaml` 已有 `table_other`
- 依赖：无新增

## 步骤 T6：单测 + 冒烟 + 缓存指纹

- 选定算法：合成夹具（双表栈 / 三表栈 AABB+假 Product 行）+ pytest；`perceive` 缓存指纹加入 table 定位相关键（read_mode 集合、aux 启用标志、线网拆分开关）
- 验收：无 aux 退化一致；有 aux 独立 bbox；material/main fields 路径不回归

## 候选尝试记录

| 步骤 | 候选算法 | 状态 | 失败原因 |
|------|----------|------|----------|
| T2 | VLM 多实例 + 形态学线网 | 采用（首选） | — |
| T2 | 仅水平间隙几何切条 | 未试（兜底） | — |
| T2 | YOLO/RT-DETR 标题栏检测 | 未试（后备） | 需训练数据，本轮不做 |

## 任务分配

| 任务 ID | 实现内容 | 关联步骤 | 参考资料 |
|---------|----------|----------|----------|
| A1 | `drawing_parse_plan` 支持 `bbox_only` / aux；yaml 可选 aux 条目或运行时注入 | T1 | 01 §接口；现有 normalize |
| A2 | 标题栏 ROI 线网候选表提取 + VLM Pass1 提示增强；角色分类写入 aux/material/main | T2 | arXiv:2105.11021；OpenCV morph lines |
| A3 | `refine_*` 扩展：Product 分界 + aux 上边界约束；无 Product 几何回退 | T3 | `table_layout_ocr.py` / `perceive_qwen_vl.py` |
| A4 | Pass2 / single-pass 跳过 bbox_only；facts 写出空业务 fields | T4 | `perceive_qwen_vl.py` |
| A5 | 渲染 table_other + 排除区纳入 aux；缓存指纹 | T5–T6 | `render_report.py` / `perceive_common.py` / `perceive_utils.py` |
| A6 | 单测（双表/三表夹具）+ 冒烟 + 实现日志 | T6 | pytest；03-implementation-log |
