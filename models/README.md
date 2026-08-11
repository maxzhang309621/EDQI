# 将权重下载到此目录（目录名与 configs/default.yaml 一致）
#
# Qwen3-8B（规则拆解）:
#   python tools/download_models.py qwen3_text
#   -> models/Qwen3-8B/
#
# Qwen3-VL-2B-Instruct（主路径感知，默认，约 8GB 显存）:
#   python tools/download_models.py qwen3_vl
#   huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct
#   -> models/Qwen3-VL-2B-Instruct/
#
# 更大 VLM（需改 configs/default.yaml 的 path/hf_id）:
#   Qwen/Qwen3-VL-4B-Instruct / Qwen/Qwen3-VL-8B-Instruct
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
