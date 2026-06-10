# 基于深度学习的视频超分辨率重建

## 项目概述

本项目实现了三种基于深度学习的视频超分辨率 (VSR) 模型：

| 模型 | 方法 | 特点 |
|------|------|------|
| **BasicVSR** | 光流对齐 + 双向传播 | 显式运动补偿，帧对齐精准 |
| **3D CNN** | Early Fusion + 3D卷积 | 结构简单，推理快速 |
| **ConvLSTM** | 双向 ConvLSTM 时序建模 | 擅长长期时序依赖 |

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 一键运行实验

```bash
# 快速验证 (合成数据, 2 epochs, ~5分钟)
python run_experiment.py --mode quick

# 标准实验 (合成数据, 5 epochs, ~15分钟)
python run_experiment.py --mode standard

# 完整实验 (需要 Vimeo-90K)
python run_experiment.py --mode full --vimeo_root /path/to/vimeo_septuplet
```

### 单独训练

```bash
# BasicVSR
python train.py --model basicvsr_light --epochs 5

# 3D CNN
python train.py --model vsr_3dcnn --epochs 5

# ConvLSTM
python train.py --model vsr_convlstm --epochs 5
```

### 推理

```bash
python inference.py --model_path results/basicvsr_light/best_model.pth --model_type basicvsr_light
```

### 评估

```bash
python evaluate.py \
  --model_paths results/basicvsr_light/best_model.pth results/vsr_3dcnn/best_model.pth \
  --model_types basicvsr_light vsr_3dcnn
```

## 项目结构

```
VSR_Project/
├── models/
│   ├── basicvsr.py        # BasicVSR (光流对齐+双向传播)
│   ├── vsr_3dcnn.py       # 3D CNN (Early Fusion)
│   └── vsr_convlstm.py    # ConvLSTM (RNN方法)
├── data/
│   └── dataset.py         # 数据加载 (Tiny, Vimeo-90K, VID4/UDM10)
├── utils/
│   └── common.py          # PSNR, SSIM, 可视化, 光流warp
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本 (支持多模型对比)
├── inference.py           # 推理脚本
├── run_experiment.py      # 一键实验脚本
└── results/               # 输出目录
```

## 评估指标

- **PSNR** (峰值信噪比): >30dB 为良好
- **SSIM** (结构相似性): >0.9 为优秀

## 参考文献

1. Chan et al., "BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond", CVPR 2021
2. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016
3. Wang et al., "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks", CVPRW 2019
