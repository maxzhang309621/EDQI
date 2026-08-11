# 通过国内镜像安装 GPU 版 PyTorch + torchvision
# 机器：RTX 40 系 / Driver 支持 CUDA 12.x → 默认 cu126
#
# 用法（项目根、已激活 .venv）:
#   .\tools\install_torch_gpu.ps1
#   .\tools\install_torch_gpu.ps1 -Cuda cu126 -Mirror aliyun
#   .\tools\install_torch_gpu.ps1 -Force   # 强制重装

param(
    [ValidateSet("cu126", "cu124", "cu121", "cu118")]
    [string]$Cuda = "cu126",

    # aliyun 推荐；nju / official 备选
    [ValidateSet("aliyun", "nju", "official")]
    [string]$Mirror = "aliyun",

    [switch]$Force
)

$ErrorActionPreference = "Stop"

$urls = @{
    aliyun   = "https://mirrors.aliyun.com/pytorch-wheels/$Cuda"
    nju      = "https://mirror.nju.edu.cn/pytorch/whl/$Cuda"
    official = "https://download.pytorch.org/whl/$Cuda"
}

$wheelUrl = $urls[$Mirror]
$pypiMirror = "https://mirrors.aliyun.com/pypi/simple/"

Write-Host "Mirror=$Mirror  CUDA=$Cuda"
Write-Host "Wheel URL=$wheelUrl"

if ($Force) {
    Write-Host "Uninstalling existing torch/torchvision..."
    python -m pip uninstall -y torch torchvision torchaudio 2>$null
}

# 阿里云 pytorch-wheels 用 --find-links 更稳；官方/南大可用 --index-url
if ($Mirror -eq "aliyun") {
    python -m pip install torch torchvision `
        --find-links $wheelUrl `
        -i $pypiMirror `
        --trusted-host mirrors.aliyun.com
} else {
    python -m pip install torch torchvision --index-url $wheelUrl
}

python -c @"
import torch, torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda_available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
"@
