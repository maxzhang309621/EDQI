# 将权重下载到此目录（目录名与 configs/default.yaml 一致）
#
# Qwen3-8B（规则拆解）:
#   python tools/download_models.py qwen3_text
#   -> models/Qwen3-8B/
#
# Qwen3-VL-8B-Instruct（主路径感知）:
#   python tools/download_models.py qwen3_vl
#   -> models/Qwen3-VL-8B-Instruct/
#
# LocateAnything-3B（备选定位）:
#   python tools/download_models.py locateanything
#   -> models/LocateAnything-3B/
#
# 国内可用: python tools/download_models.py qwen3_vl --source modelscope
#
# LocateAnything 代码依赖:
#   git clone https://github.com/NVlabs/Eagle.git third_party/Eagle
#   cd third_party/Eagle/Embodied && pip install -e .
#   并将 locateanything_worker 加入 PYTHONPATH
