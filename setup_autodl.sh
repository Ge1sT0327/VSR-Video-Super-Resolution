#!/bin/bash
# ============================================================
# AutoDL 一键部署脚本
# 使用方法: bash setup_autodl.sh
# ============================================================
set -e

echo "============================================"
echo "  VSR 视频超分辨率 — AutoDL 环境配置"
echo "============================================"

# -------- 1. 配置 pip 镜像 (国内加速) --------
echo "[1/5] 配置 pip 镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# -------- 2. 安装依赖 --------
echo "[2/5] 安装 Python 依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy Pillow opencv-python tqdm

# -------- 3. 下载 Vimeo-90K 数据集 --------
echo "[3/5] 下载 Vimeo-90K 数据集 (~80GB, 需要较长时间)..."
DATASET_DIR="./data/vimeo_septuplet"

if [ -d "$DATASET_DIR/sequences" ]; then
    echo "  数据集已存在，跳过下载"
else
    mkdir -p ./data

    # Vimeo-90K 官方下载地址
    # 如果 autodl 有预置数据集，可以直接软链接
    if [ -d "/root/autodl-pub/Vimeo-90k" ]; then
        echo "  使用 AutoDL 公开数据集"
        ln -s /root/autodl-pub/Vimeo-90k ./data/vimeo_septuplet
    elif [ -d "/root/autodl-tmp/vimeo_septuplet" ]; then
        echo "  使用 AutoDL 预置数据集"
        ln -s /root/autodl-tmp/vimeo_septuplet ./data/vimeo_septuplet
    elif [ -d "/root/autodl-fs/vimeo_septuplet" ]; then
        echo "  使用 AutoDL 网盘数据集"
        ln -s /root/autodl-fs/vimeo_septuplet ./data/vimeo_septuplet
    else
        echo "  下载 Vimeo-90K (可能需要数小时)..."
        echo "  提示: 也可以在 AutoDL 的'公开数据集'中搜索 vimeo_septuplet 直接挂载"

        # 尝试 wget 下载
        wget -c http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip -P ./data/ 2>/dev/null || {
            echo ""
            echo "  *** 自动下载失败 ***"
            echo "  请手动操作:"
            echo "  1. 在 AutoDL 控制台 → 我的网盘 → 上传 vimeo_septuplet"
            echo "  2. 或在创建实例时挂载公开数据集 'vimeo_septuplet'"
            echo "  3. 将数据集软链接到 ./data/vimeo_septuplet"
            echo ""
            echo "  如果暂无 Vimeo-90K，可用合成数据先跑:"
            echo "    python run_experiment.py --mode standard"
        }
    fi
fi

# -------- 4. 下载测试集 VID4 --------
echo "[4/5] 下载 VID4 测试集..."
VID4_DIR="./data/VID4"

if [ -d "$VID4_DIR" ]; then
    echo "  VID4 已存在，跳过"
else
    mkdir -p "$VID4_DIR"
    # VID4 包含 4 个序列: calendar, city, foliage, walk
    # 从 GitHub 或其他源下载
    git clone https://github.com/LIUTINGT/VID4.git "$VID4_DIR" 2>/dev/null || {
        echo "  git clone 失败, 尝试直接下载..."
        pip install gdown
        gdown https://drive.google.com/uc?id=1T8TuyyOxEUfXzpb2bRL6jd1T2EXnPYuC -O "$VID4_DIR/VID4.zip" 2>/dev/null || {
            echo "  VID4 下载失败，将使用合成数据评估"
            rm -rf "$VID4_DIR"
        }
    }
fi

# -------- 5. 验证环境 --------
echo "[5/5] 验证环境..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"

echo ""
echo "============================================"
echo "  环境配置完成!"
echo "============================================"
echo ""
echo "运行实验:"
echo "  # 完整训练 (Vimeo-90K)"
echo "  python run_experiment.py --mode full --dataset vimeo --vimeo_root ./data/vimeo_septuplet --epochs 50 --batch_size 4"
echo ""
echo "  # 快速测试 (合成数据)"
echo "  python run_experiment.py --mode standard"
echo ""
