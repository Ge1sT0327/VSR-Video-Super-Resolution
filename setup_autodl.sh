#!/bin/bash
# ============================================================
# AutoDL 一键部署脚本
# 使用方法: bash setup_autodl.sh
# ============================================================
set -e

echo "============================================"
echo "  VSR 视频超分辨率 — AutoDL 环境配置"
echo "============================================"

# -------- 1. pip 镜像加速 --------
echo "[1/4] 配置 pip 镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# -------- 2. 安装依赖 --------
echo "[2/4] 安装 Python 依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy Pillow opencv-python tqdm

# -------- 3. 准备 Vimeo-90K 数据集 --------
echo "[3/4] 准备 Vimeo-90K 数据集..."
DATASET_DIR="/root/autodl-tmp/vimeo_septuplet"

if [ -d "$DATASET_DIR/sequences" ]; then
    echo "  数据集已就绪: $DATASET_DIR"
else
    ZIP_FILE="/root/autodl-pub/Vimeo-90k/vimeo_septuplet.zip"

    if [ -f "$ZIP_FILE" ]; then
        echo "  发现压缩包, 开始解压到 /root/autodl-tmp/ (约需 10-20 分钟)..."
        unzip -q "$ZIP_FILE" -d /root/autodl-tmp/
        echo "  解压完成: $DATASET_DIR"
    elif [ -d "/root/autodl-pub/vimeo_septuplet" ]; then
        echo "  使用公开数据集软链接"
        ln -s /root/autodl-pub/vimeo_septuplet "$DATASET_DIR"
    elif [ -d "/root/autodl-fs/vimeo_septuplet" ]; then
        echo "  使用网盘数据集软链接"
        ln -s /root/autodl-fs/vimeo_septuplet "$DATASET_DIR"
    else
        echo "  *** 未找到 Vimeo-90K 数据集 ***"
        echo "  请在 AutoDL 创建实例时挂载公开数据集 'Vimeo-90k'"
        echo "  或用合成数据先跑: bash run_autodl.sh standard"
    fi
fi

# 创建软链接到项目目录
if [ -d "$DATASET_DIR" ] && [ ! -L "./data/vimeo_septuplet" ]; then
    mkdir -p ./data
    ln -s "$DATASET_DIR" ./data/vimeo_septuplet
    echo "  已链接: ./data/vimeo_septuplet -> $DATASET_DIR"
fi

# -------- 4. 验证环境 --------
echo "[4/4] 验证环境..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"

echo ""
echo "============================================"
echo "  环境配置完成!"
echo "============================================"
echo ""
echo "  运行完整实验:"
echo "    bash run_autodl.sh full"
echo ""
echo "  快速测试 (合成数据):"
echo "    bash run_autodl.sh standard"
echo ""
