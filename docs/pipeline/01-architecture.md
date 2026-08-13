# 架构方案：EDQI 部件分区 VLM 属性识别

- 版本：v1.0（回填自已批准方案「部件分区属性架构」）
- 日期：2026-08-13
- 用户确认：是（Cursor plan 批准 + 「按本地 skill 全流程做补充」）

## 需求概述

- **核心目标**：关闭数字重叠 OCR 路径；将 VLM 尺寸属性识别改为「先定位带标签部件/视图 → 按部件外扩区识别属性 → 属性经 parent_id 关联部件」；结果图可视化部件原始框与扩展框以便调试。
- **输入**：工程图页图（ingest 后 RGB）、drawing_parse 计划（含 `number_mark` / 表格）、配置 `perception.view_regions`。
- **输出**：
  - `component` 实例 → `facts.components`（label、bbox、bbox_expanded）
  - `number_mark` 实例 → `facts.annotations`（含 `parent_id`）
  - 可视化：部件黄/红框 + 属性框 + 表格框
- **约束**：
  - 不改 `perceive_number_overlap` 墨迹判定超参
  - 不重写表格 VLM batch
  - 技术栈保持现有 Python + Qwen3-VL + PIL 管线
  - 0 个部件时回退网格分块保底

## 可行性结论

**可行**

- 技术成熟度：现有 Pass1/Pass2 尺寸 VLM、表格 batch、facts/render 均可复用；仅增加 Pass0 定位与几何分区。
- 数据可得性：无需新数据集；用现有图纸调试即可。
- 资源约束：多一次轻量 VLM 定位调用 + 按部件多次 crop 识别，显存与耗时可接受；可用 `locate_max_new_tokens` 控制。

## 模块划分

| 模块 | 职责 | 所需算法/逻辑/架构类型 |
|------|------|------------------------|
| 重叠路径开关 | 关闭双后端/自动重叠 OCR，规则置 draft | 配置与规则生命周期管理 |
| Pass0 部件定位 | 整页框出几何视图并赋短 label | 视觉语言模型定位（检测式输出） |
| 识别区几何 | 外扩、扣表、过小丢弃、过大拆分、空结果回退网格 | 轴对齐框几何 / 启发式区域规划 |
| 分区属性识别 | 各区 crop 后跑既有 dimension Pass1/Pass2 | 既有 VLM 两阶段抽取 |
| 关联与 Facts | component 实例 + annotation.parent_id | 对象图关联（父子引用） |
| 可视化 | 画原始/扩展部件框与属性框 | 2D 标注渲染 |

## 数据流与接口约定

```
page → tables_batch（既有）
     → Pass0 locate_drawing_views → [{bbox, label}, ...]
     → resolve_part_region_specs → [{part_id, label, bbox_raw, bbox_expanded, regions[]}]
     → 每 region: perceive_qwen_vl([number_mark]) → shift → parent_id=part_id
     → component 实例 + number_mark 实例 → build_facts → render_report
```

关键接口：

| 接口 | 输入 | 输出 |
|------|------|------|
| `locate_drawing_views` | 整页 Image / 路径, config | `(views[{bbox,label}], notes)` |
| `resolve_part_region_specs` | views, page_wh, table_bboxes, expand… | `(specs, source∈{view_regions,grid,full_page})` |
| `perceive_with_cache_and_tiles`（尺寸分支） | plan 含 dimension_marks | instances含 component + number_mark(parent_id) |
| `build_facts` | instances | `components[]`, `annotations[].parent_id`, `bbox_expanded` |
| `render_report` | facts + show_view_regions | 标注图 |

关联约定：`parent_id == component.instance_id`（形如 `component#i`）。

## 开发步骤与验收标准

| 步骤 | 描述 | 所需算法类型 | 验收标准（可测试） |
|------|------|--------------|---------------------|
| S1 | 关闭数字重叠识别 | 配置/规则状态 | `auto_number_overlap_backend=false`；`perception_dual_backend=false`；`NUM_TEXT_NO_OVERLAP.status=draft`；only_active 加载不含该规则 |
| S2 | Pass0 部件定位 API | VLM 定位 | 权重就绪时返回带 bbox+label 的列表；近整页框被丢弃；notes 含 `view_locate_raw=` |
| S3 | 识别区几何 | 框几何启发式 | 外扩后更大；中心在表内→丢弃；与表重叠可裁剪；空视图→grid/full_page |
| S4 | 分区识别 + 关联 | VLM 抽取 + 父子引用 | 属性带 `parent_id`；存在对应 `component`；半页大框丢弃；notes 含 `view_regions:n=` / `view_regions_source=` |
| S5 | 可视化 | 2D 渲染 | `show_view_regions=true` 时画出 raw+expanded；图例含 view raw/expanded |
| S6 | 单测与回归 | 单元测试 | 相关 pytest 全部通过 |

## 风险与外部依赖

- Pass0 漏检/融框 → 依赖 `fallback_grid` 保底
- 部件框切碎尺寸标注 → `expand_ratio`（默认 0.2）可调
- 缓存未含 view_regions 指纹时会沿用旧结果 → plan_key 需纳入指纹
- 无 active 规则时 `run()` 须允许仅 drawing_parse 计划
- 依赖本地 Qwen3-VL 权重；未就绪时 Pass0 空列表并回退网格
