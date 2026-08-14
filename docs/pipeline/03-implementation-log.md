# 实现日志：标题栏表格定位优化 + 附属表分离

- 日期：2026-08-14
- 依据：`02-algorithm-plan.md` 任务 A1–A6

## 实现内容

| 任务 | 变更 |
|------|------|
| A1 | `drawing_parse_plan` 支持 `read_mode=bbox_only` / `section=aux`；`drawing_parse.yaml` 增加 `aux_table` |
| A2 | 新增 `pipeline/title_block_tables.py`：形态学线网候选、水平间隙切条、附属表分离；Pass1 提示强调 aux |
| A3 | `refine_material_main_boxes` 在 Product/几何分界后调用 `separate_aux_tables` + `enforce_aux_above_material` |
| A4 | Pass2 / single-pass 对 bbox_only 跳过字段识读，仅写元数据 |
| A5 | 渲染 `table_other`；排除区沿用全表框；缓存指纹 `_perception_title_block_tables`；`default.yaml` 增加配置 |
| A6 | `tests/test_title_block_tables.py`；更新 plan 期望含 `aux_table` |

**未改**：material `above_cells` OCR、main `cell_content` 字段配置与取值逻辑。

## 冒烟 / 单测

```
python -m pytest tests/ --ignore=tests/test_m4_hardening.py -q
→ 98 passed
```

plan+mock：`aux_table` 为 `bbox_only`；material/main 字段 mock 正常。

（`test_m4_hardening.py` 为既有 `_bootstrap` 导入问题，与本轮无关。）

## 遗留

- 真实图纸端到端效果依赖 VLM Pass1 + 线网后处理；无线框附属表主要靠间隙切条兜底。
- 附属表暂不区分修订表/BOM 等语义名。
