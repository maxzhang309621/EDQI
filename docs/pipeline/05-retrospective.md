# 经验总结：标题栏表格定位 + 附属表分离

## 算法选型结论

| 步骤 | 最终采用 | 被否决的候选 | 否决原因 |
|------|----------|--------------|----------|
| T2 | VLM 多实例提示 + 形态学 HV 线网 + 水平间隙切条 | YOLO/RT-DETR 标题栏检测 | 需标注训练，本轮成本过高 |
| T3 | Product OCR 硬分界（保留）+ aux 上边界几何约束 | 仅依赖 VLM 框 | 生产路径已验证 Product 更稳 |
| T4 | `read_mode=bbox_only` 门控 | 对 aux 跑空字段 VL | 浪费且易污染 |

## 踩坑记录

- **facts 映射遗漏**：新增 `entity_id` 必须同步 `ENTITY_FACTS_MAP`，否则 `build_facts` 会写成顶层单键而非 `tables[]`；动态 `aux_table_{i}` 还需前缀兜底。
- **表格批缓存按精确 entity_id 切片**：多 aux 实例需 `startswith("aux_table")`，且不能 `_finalize_entity` 强行改写为同一 id 导致塌缩（aux 分支单独处理）。
- **循环导入**：`title_block_tables` 勿直接 import `perceive_qwen_vl`；由后者回调前者。

## 可复用组件

- `pipeline/title_block_tables.py`：`detect_bordered_table_candidates`、`split_bbox_by_horizontal_gaps`、`separate_aux_tables`、`enforce_aux_above_material`
- 配置：`perception.title_block_tables` + 缓存指纹 `_title_block_tables_cache_fp`

## 流程改进建议

- 新增表格 `entity_id` 时在 checklist 增加「ENTITY_FACTS_MAP + render section + exclude」三项。
- Windows PowerShell 下 git commit 勿用 bash HEREDOC；实现 skill 可注明平台差异。
