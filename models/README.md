# 将权重下载到此目录（目录名与 configs/default.yaml 一致）
#
# Qwen3-8B（规则拆解）:
#   python tools/download_models.py qwen3_text
#   -> models/Qwen3-8B/
#
# Qwen3-VL-30B-A3B-Instruct（主路径感知，MoE）:
#   HF: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
#   ModelScope: https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct
#   python tools/download_models.py qwen3_vl
#   -> models/Qwen3-VL-30B-A3B-Instruct/
#   显存紧可改配置为 Qwen3-VL-30B-A3B-Instruct-FP8 再下载
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
