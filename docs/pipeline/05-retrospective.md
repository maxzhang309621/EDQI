# 经验总结：部件粗边框识别 + 扩展收紧

## 算法选型结论

| 步骤 | 最终采用 | 被否决的候选 | 否决原因 |
|------|----------|--------------|----------|
| B1 | 形态学 OPEN 保留粗笔画 + CLOSE 补缝 | 仅全墨迹 AABB | 细尺寸线会撑大框 |
| B3 | `expand_ratio` 0.28→0.18 | 更大外扩 | 用户要求收紧；网格补扫兜底漏检 |

## 踩坑记录

- OPEN 核过大会打断略细「粗线」→ 失败回退 `tighten_bbox_to_ink`
- 缓存指纹必须含 `tighten_mode` / `thick_min_width` / `expand_ratio`

## 可复用组件

- `pipeline/view_regions.py`：`tighten_bbox_to_thick_outline`、`tighten_view_bbox`

## 流程改进建议

- 线宽相关默认值可按图纸 DPI 再缩放（当前固定 px）
