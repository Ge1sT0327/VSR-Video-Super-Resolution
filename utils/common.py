"""
utils/common.py — 通用工具函数
========================================
包含:
- PSNR / SSIM 指标计算
- 光流 warp 操作
- 图像 I/O 与可视化
- 设备检测
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from PIL import Image


# ============================================================
# 图像质量评估
# ============================================================

def calculate_psnr(img1, img2, max_val=1.0):
    """计算 PSNR (dB)，越高越好"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(max_val ** 2 / mse)


def calculate_ssim(img1, img2, window_size=11):
    """计算 SSIM [0, 1]，越接近 1 越好"""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = img1.size(1)
    kernel = torch.ones(channel, 1, window_size, window_size,
                        device=img1.device) / (window_size ** 2)
    padding = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=padding, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=padding, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ============================================================
# 光流 Warp
# ============================================================

def flow_warp(x, flow):
    """使用光流对特征图进行对齐 (backward warp)"""
    B, C, H, W = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow_grid = torch.zeros_like(grid)
    flow_grid[..., 0] = grid[..., 0] + flow[:, 0] / ((W - 1) / 2)
    flow_grid[..., 1] = grid[..., 1] + flow[:, 1] / ((H - 1) / 2)

    return F.grid_sample(x, flow_grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)


# ============================================================
# 图像 I/O
# ============================================================

def tensor_to_numpy(tensor):
    """Tensor [C,H,W] / [1,C,H,W] → numpy [H,W,C] uint8"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


def save_image(tensor, path):
    """保存 tensor 为 PNG"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    img = tensor_to_numpy(tensor)
    Image.fromarray(img).save(path)


def save_comparison(lr, bicubic, sr, hr, path):
    """水平拼接保存 LR / Bicubic / SR / HR 四宫格对比图"""
    if lr.shape[-1] != hr.shape[-1]:
        lr_up = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:],
                              mode='nearest').squeeze(0)
    else:
        lr_up = lr

    images = [lr_up, bicubic, sr, hr]
    labels = ['LR (Nearest)', 'Bicubic', 'SR (Ours)', 'HR (GT)']
    imgs_np = [tensor_to_numpy(img) for img in images]

    # 添加标签到每张图上
    from PIL import ImageDraw, ImageFont
    h, w = imgs_np[0].shape[:2]
    font_h = 24
    labeled = []
    for i, (img_np, lbl) in enumerate(zip(imgs_np, labels)):
        pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (w, font_h)], fill=(0, 0, 0))
        draw.text((5, 3), lbl, fill=(255, 255, 255))
        labeled.append(np.array(pil))

    concat = np.concatenate(labeled, axis=1)
    Image.fromarray(concat).save(path)
    print(f"  对比图已保存: {path}")


# ============================================================
# 设备
# ============================================================

def setup_device():
    """自动检测 GPU/CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"使用 GPU: {name} ({mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


def count_parameters(model):
    """统计可训练参数量"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total:,} ({total / 1e6:.2f}M)")
    return total
