# EDQI — 工程制图 AI 质检（初步实现）

用户给定检测规则 → 拆解入库 → 从图中定位/解析 → 规则引擎判定 → 标注报告。

实现依据：[docs/工程制图AI质检_实现流程.md](docs/工程制图AI质检_实现流程.md)

## 当前里程碑

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| M1 契约与引擎 | 已实现 | Schema / 规则引擎 / 可视化 / 单测 |
| M2 感知闭环 | 已实现骨架 | Qwen3-VL 主路径 + LA+OCR 备选；无权重回退 mock；缓存/分块 |
| M3 规则生产 | 已实现骨架 | `rule_decompose` / `validate` / `activate`；多条示例规则 |
| M4 硬化 | 部分完成 | 金标评测、批跑、复核队列、权重就绪检测；CAD/LoRA 未做 |

## 快速开始（无需 GPU）

```bash
cd "D:\maxzhang\python files\EDQI"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 端到端 demo（mock 感知）
python run_demo.py

# 单测
python -m pytest tests -q
```

> **Python 3.13 注意**：不要安装 `rapidocr-onnxruntime>=1.3.0`（不支持 3.13）。  
> 本仓库已约束为 `>=1.2.0,<1.3.0`。若仍报错，可单独执行：  
> `pip install "rapidocr-onnxruntime==1.2.3"`

## M4 硬化能力

```bash
# 查看本地权重是否下完（仅有 config.json 不算就绪）
python tools/check_models.py

# 批跑目录
python tools/batch_check.py data/samples --backend mock

# 金标评测（可对比多后端）
python tools/evaluate_gold.py --gold-dir data/gold --backend mock
python tools/evaluate_gold.py --gold-dir data/gold --backends mock,qwen_vl

# Findings 复核队列
python -m pipeline.review_queue list
python -m pipeline.review_queue resolve work_dirs/review_queue/xxx.json --decision approve
```

感知缓存与大图分块默认开启，配置见 `configs/default.yaml` 的 `perception:` 段。

**常驻推理（推荐，避免每次重载 VLM）**：

```bash
# 预热 Qwen3-VL + RapidOCR 后交互输入图纸路径
python -m pipeline.worker

# 或预热后批跑多张（同进程复用权重；重复跑同图应看到 [cache] hits）
python -m pipeline.worker data/input/test.pdf data/input/test.pdf
```

输出在 `work_dirs/{facts,vis,reports}/`。

## 目录结构

```text
EDQI/
├── configs/default.yaml          # 后端与本地权重路径
├── rules/schema/                 # rule/facts/findings/instances schema
├── rules/library/                # status=active 正式规则
├── rules/drafts/                 # S1 草稿
├── engines/rule_engine.py        # S7 确定性引擎
├── pipeline/
│   ├── ingest.py                 # S3
│   ├── perceive_qwen_vl.py       # S4+S5 主路径
│   ├── perceive_la_ocr.py        # S4+S5 备选
│   ├── build_facts.py            # S6
│   ├── render_report.py          # S8
│   └── run.py              # 在线总入口
├── tools/
│   ├── rule_decompose.py         # S1
│   ├── rule_validate.py          # S2
│   ├── rule_activate.py          # S2
│   └── download_models.py        # 下载本地权重
└── models/                       # 权重落盘目录（预留）
```

## 下载本地权重（稍后）

在 [configs/default.yaml](configs/default.yaml) 中已预留路径：

| 用途 | 配置键 | 本地目录 | HF |
|------|--------|----------|-----|
| 规则拆解 | `models.qwen3_text` | `models/Qwen3-8B` | `Qwen/Qwen3-8B` |
| 主路径 VLM（默认，约 8GB 显存） | `models.qwen3_vl` | `models/Qwen3-VL-2B-Instruct` | `Qwen/Qwen3-VL-2B-Instruct` |
| 备选定位 | `models.locateanything` | `models/LocateAnything-3B` | `nvidia/LocateAnything-3B` |

```bash
pip install "transformers>=4.57.0" torch accelerate qwen-vl-utils huggingface_hub

# 推荐：轻量 VLM（Qwen3-VL-2B-Instruct，适配约 8GB 显存）
python tools/download_models.py qwen3_vl
# 若 SSL/连不上 HuggingFace，优先用镜像或 ModelScope：
python tools/download_models.py qwen3_vl --mirror
# 或:
# $env:HF_ENDPOINT="https://hf-mirror.com"
# python tools/download_models.py qwen3_vl
# 等价 huggingface-cli：
# huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct

python tools/download_models.py qwen3_text
python tools/download_models.py locateanything
# 或一次性: python tools/download_models.py all
```

国内可用 ModelScope：

```bash
python tools/download_models.py qwen3_vl --source modelscope
```

显存更充足时可改用更大 VLM（需同步改 `configs/default.yaml` 的 `path` / `hf_id`）：

| 型号 | HF | 说明 |
|------|-----|------|
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | 质量更好，8GB 建议量化 |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | 原默认大模型，约需 16GB+ |

权重就绪后，改配置：

```yaml
perception_backend: qwen_vl   # 或 locateanything_ocr / hybrid
```

然后：

```bash
python -m pipeline.run data/samples/demo_drawing.png --backend qwen_vl
```

LocateAnything 还需官方代码（参考 [NVlabs/Eagle Embodied](https://github.com/NVlabs/Eagle)）：

```bash
git clone https://github.com/NVlabs/Eagle.git third_party/Eagle
cd third_party/Eagle/Embodied
pip install -e .
# 保证 locateanything_worker 可 import
```

## 规则生产（S1→S2）

```bash
# 无权重：生成可编辑草稿模板
python tools/rule_decompose.py "标题栏必须填写图号" --offline-template

# 有 Qwen3-8B 权重：自动拆解
python tools/rule_decompose.py "标题栏必须填写图号"

python tools/rule_validate.py rules/drafts/TB_PART_NO_REQUIRED.json
# 人工确认后入库
python tools/rule_activate.py rules/drafts/TB_PART_NO_REQUIRED.json
```

在线检测**只加载** `rules/library/*.json` 且 `status=active` 的规则。

## 感知后端切换

```bash
python -m pipeline.run drawing.png --backend mock
python -m pipeline.run drawing.png --backend qwen_vl
python -m pipeline.run drawing.png --backend locateanything_ocr
python -m pipeline.run drawing.png --backend hybrid
```

主/备路径输出统一为 `instances[]`，再经 `build_facts` → 规则引擎，互不耦合。

## 参考仓库

- [Qwen3](https://github.com/QwenLM/Qwen3)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [NVlabs/Eagle · LocateAnything](https://github.com/NVlabs/Eagle)
- [RapidOCR](https://github.com/RapidAI/RapidOCR) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
