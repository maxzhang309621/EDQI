# 把待检工程图放到此目录（支持 png/jpg/pdf）
# 示例：
#   data/input/my_drawing.png
#
# 主路径检测数字重叠：
#   python -m pipeline.run data/input/my_drawing.png --backend qwen_vl --rule-id NUM_TEXT_NO_OVERLAP
#
# 结果：
#   work_dirs/vis/*_annotated.jpg
#   work_dirs/reports/*_report.json
#   work_dirs/facts/*.json
