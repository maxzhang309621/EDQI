# 架构方案：标题栏表格定位优化 + 物料表上方附属表分离

- 版本：v4.0
- 日期：2026-08-14
- 用户确认：是（指令：确认）
- 前序：v3.0 部件粗边框识别 + 扩展收紧（已完成，本需求为表格子系统局部变更）

## 需求概述

- **核心目标**
  1. **优化表格定位识别**：更准确地区分标题栏区域内的多张表（整表 bbox），避免把多张表合成一个大框、或把上方附属表并入物料表。
  2. **物料表上方附属表**：若存在（修订表、备注表、其它参数小表等），**分离为独立表格实例**，暂时**只输出整表 bbox**，不做内部字段识别。
  3. **主表 / 物料表字段识别保持不变**：`material_table` 的 `above_cells` OCR、`main_table` 的 `cell_content` VL 字段配置与读值逻辑不改。
- **输入**：页图 + `drawing_parse.yaml` 中已配置的 material/main 表实体；可选 VLM Pass1 粗框
- **输出**：`facts.tables` 中至少包含 material / main（有则）；另可含 0..N 个附属表（`section=aux` 或等价），仅 bbox + 元数据，fields 为空或仅元字段
- **约束**
  - 不改主表/物料表已配置字段清单与取值规则
  - 技术栈：Python + 既有 VLM / OCR / OpenCV 几何，与现流水线兼容
  - 附属表暂不进入字段 Pass2

## 可行性结论

**可行**

- 现有链路已具备「多表实体 plan + Product 行 OCR 硬分界 + 几何硬分界」；问题主要集中在**定位阶段把上方附属表吞进物料框**。
- 在**定位与后处理**层增加「多表栈拆分 / 附属表检出」，并将附属表标记为 `bbox_only`，即可满足「分离但不识读」；material/main 的 Pass2 读值路径可原样保留。

## 模块划分

| 模块 | 职责 | 所需算法/逻辑/架构类型 |
|------|------|------------------------|
| 标题栏多表定位 | 在右下图框/标题栏邻域检出 1..N 个整表 bbox（而非仅 material+main） | 检测/定位类（VLM 多实例或几何线网候选）+ 提示约束 |
| 表栈角色分类 | 将检出框归类为：附属(aux) / 物料(material) / 主表(main) | 规则引擎（锚点行 + 垂直顺序 + 关键词） |
| Product 分界（保留） | 物料下沿 / 主表上沿仍以 Product 行为硬分界 | 已有 OCR 锚点分界 |
| 附属表 bbox 收紧 | 可选：按表格线网/墨迹收紧附属表外框，避免粘连 | 形态学/连通域或线网外接框 |
| 识读门控 | material/main 走原 Pass2；aux 跳过字段识读 | 策略模式 / read_mode 扩展 |
| 渲染与排除区 | 附属表可绘制 `table_other`；计入尺寸排除区（与 material/main 同等） | 既有渲染/排除管线扩展 |

**明确不改**

- `material_table.fields` / `main_table.fields` 配置与 `above_cells` / `cell_content` 取值实现

## 数据流与接口约定

```
Pass1 多表定位（标题栏邻域）
  → 候选整表 bbox 列表
  → 角色分类（aux* / material / main；Product 锚点约束 material|main）
  → 几何/Product 硬分界 refine（仅 material↔main；aux 不得侵入 material 有效区）
  → Pass2：
       material → 原 above_cells OCR
       main     → 原 cell_content VL
       aux*     → 跳过（fields 仅元数据：section/read_mode/entity_id）
  → facts.tables + 渲染/尺寸排除
```

| 字段/约定 | 含义 |
|-----------|------|
| `entity_id` | material/main 保持；附属表建议 `aux_table` 或 `aux_table_{i}` |
| `section` | `material` \| `main` \| `aux` |
| `read_mode` | material=`above_cells`；main=`cell_content`；aux=**`bbox_only`**（新增合法值） |
| `cardinality` | material/main=`one`；aux=`zero_or_more` |
| 输出实例 | 每张表一条 instance：`bbox` 必有；aux 无业务 fields |

配置侧（高层，不绑具体算法）：在 `tables` 中增加可选附属表条目或允许运行时动态产出 aux 实例；`drawing_parse_plan` 需接受 `read_mode=bbox_only`。

## 开发步骤与验收标准

| 步骤 | 描述 | 所需算法类型 | 验收标准（可测试） |
|------|------|--------------|---------------------|
| T1 | 扩展 plan：`read_mode=bbox_only`；aux 实体规范化 | 配置/契约 | plan 可加载 aux；非法 read_mode 仍回退；material/main 契约不变 |
| T2 | 多表定位与角色分类（含「无附属表」退化） | 多实例定位 + 规则分类 | 仅 material+main 的样例：行为与现网一致；有上方附属表时：独立 bbox，不并入 material |
| T3 | 分界 refine：aux 与 material 不粘连；Product 分界仍只服务 material/main | 锚点 + 几何约束 | Product 找到时 material/main 分界正确；aux.y2 ≤ material.y1（允许小 gap）；无 Product 时几何回退仍可用 |
| T4 | Pass2 门控：aux 不调字段识读 | 策略门控 | aux 实例 fields 无业务键；material/main 字段路径单测/冒烟与改前一致 |
| T5 | 渲染 + 尺寸排除纳入 aux bbox | 管线集成 | `show_table_boxes` 可画 aux（`table_other`）；尺寸排除区覆盖 aux |
| T6 | 单测 + 冒烟 + 缓存指纹 | 工程集成 | pytest 通过；含「双表 / 三表栈」几何夹具；感知缓存指纹含新定位相关配置 |

## 风险与外部依赖

- VLM 仍可能把附属表与物料表合成一大框 → 需后处理拆分或二次定位；失败时应至少保证 material/main 可读，aux 可缺省。
- 附属表形态多样（无标准表头）→ 本阶段只要求「分离成框」，不承诺语义命名（统一 `aux`）。
- 搜索区若上扩过大可能误框视图区表格线 → 定位应限制在标题栏邻域（右下图框带）。
- 缓存：定位逻辑变更必须进入感知指纹，避免旧缓存掩盖改进。

## 待用户确认的一点（非阻塞假设）

- 附属表**暂不**要求区分具体类型名（修订表/BOM 等），统一 `section=aux` 即可。若你需要按类型命名，确认方案时说明即可。
