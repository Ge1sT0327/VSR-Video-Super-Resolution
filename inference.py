"""
inference.py — 推理脚本
========================================
对任意 LR 视频帧序列进行超分重建。
支持:
- 从文件夹加载帧
- 合成演示数据
- 视频文件 (需要 opencv-python)
"""

import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import create_model
from utils.common import (calculate_psnr, calculate_ssim, save_image,
                           save_comparison, setup_device, tensor_to_numpy)


def load_frames_from_dir(frame_dir, max_frames=None):
    """从目录加载图像帧序列"""
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(frame_dir, pat))))
    if not files:
        raise FileNotFoundError(f"在 {frame_dir} 中未找到图像文件")
    if max_frames:
        files = files[:max_frames]
    print(f"加载 {len(files)} 帧从 {frame_dir}")
    frames = []
    for f in files:
        img = Image.open(f).convert('RGB')
        frames.append(TF.to_tensor(img))
    return torch.stack(frames)


def create_demo_input(scale=4, num_frames=7, lr_size=64):
    """生成合成演示数据"""
    hr_size = lr_size * scale
    canvas_size = hr_size + 40
    np.random.seed(42)

    base = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    for _ in range(15):
        color = np.random.rand(3)
        y1, x1 = np.random.randint(0, canvas_size, 2)
        h, w = np.random.randint(20, canvas_size // 3, 2)
        base[y1:y1+h, x1:x1+w] = color
    for i in range(0, canvas_size, 8):
        for j in range(0, canvas_size, 8):
            if (i // 8 + j // 8) % 2 == 0:
                base[i:i+8, j:j+8] += 0.1
    base = np.clip(base, 0, 1)

    hr_frames = []
    for t in range(num_frames):
        y = max(0, min(20 + t * 2, canvas_size - hr_size))
        x = max(0, min(20 + t * 3, canvas_size - hr_size))
        frame = base[y:y+hr_size, x:x+hr_size]
        hr_frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    hr_frames = torch.stack(hr_frames)

    lr_frames = F.interpolate(hr_frames, size=(lr_size, lr_size),
                               mode='bicubic', align_corners=False).clamp(0, 1)
    return lr_frames, hr_frames


@torch.no_grad()
def inference(model, lr_frames, device, batch_frames=7):
    """对 LR 帧序列进行超分推理 (滑动窗口支持长序列)"""
    model.eval()
    T = lr_frames.shape[0]

    if T <= batch_frames:
        lr_input = lr_frames.unsqueeze(0).to(device)
        sr_output = model(lr_input)
        return sr_output.squeeze(0).cpu()

    print(f"长序列 ({T} 帧)，滑动窗口 size={batch_frames}")
    sr_frames = []
    stride = max(1, batch_frames - 2)

    for start in range(0, T, stride):
        end = min(start + batch_frames, T)
        if end - start < 3:
            break
        chunk = lr_frames[start:end].unsqueeze(0).to(device)
        sr_chunk = model(chunk).squeeze(0).cpu()
        if start == 0:
            sr_frames.append(sr_chunk)
        else:
            sr_frames.append(sr_chunk[batch_frames - stride:])

    return torch.cat(sr_frames, dim=0)[:T]


def main():
    parser = argparse.ArgumentParser(description='VSR 推理')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--model_type', type=str, default='basicvsr_light',
                        help='模型类型')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入 LR 帧目录 (留空=合成演示)')
    parser.add_argument('--output_dir', type=str, default='results/inference',
                        help='输出目录')
    parser.add_argument('--scale', type=int, default=4, help='超分倍率')
    parser.add_argument('--num_frames', type=int, default=7, help='处理帧数')
    args = parser.parse_args()

    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"\n[1/4] 加载模型: {args.model_type}")
    model = create_model(args.model_type, scale=args.scale)

    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f"  已加载: {args.model_path} (epoch={ckpt.get('epoch', '?')})")
    else:
        print(f"  * 模型文件不存在，使用随机初始化 (仅验证流程)")

    model = model.to(device)
    model.eval()

    # 准备数据
    print("\n[2/4] 准备输入...")
    hr_frames = None
    if args.input_dir and os.path.isdir(args.input_dir):
        lr_frames = load_frames_from_dir(args.input_dir, args.num_frames)
    else:
        print("使用合成演示数据")
        lr_frames, hr_frames = create_demo_input(args.scale, args.num_frames)

    print(f"  LR: {lr_frames.shape}")

    # 推理
    print("\n[3/4] 超分重建...")
    import time
    t0 = time.time()
    sr_frames = inference(model, lr_frames, device)
    elapsed = time.time() - t0
    print(f"  完成! {elapsed:.2f}s ({elapsed/lr_frames.shape[0]:.2f}s/帧)")
    print(f"  SR: {sr_frames.shape}")

    # 保存结果
    print("\n[4/4] 保存结果...")
    T = sr_frames.shape[0]
    bicubic_frames = F.interpolate(
        lr_frames, scale_factor=args.scale,
        mode='bicubic', align_corners=False).clamp(0, 1)

    for t in range(T):
        save_image(sr_frames[t].clamp(0, 1),
                   os.path.join(args.output_dir, f'sr_{t+1:03d}.png'))
        save_image(bicubic_frames[t],
                   os.path.join(args.output_dir, f'bicubic_{t+1:03d}.png'))

        if hr_frames is not None:
            save_image(hr_frames[t],
                       os.path.join(args.output_dir, f'hr_{t+1:03d}.png'))
            save_comparison(
                lr_frames[t], bicubic_frames[t], sr_frames[t].clamp(0, 1), hr_frames[t],
                os.path.join(args.output_dir, f'compare_{t+1:03d}.png'))

    # 报告
    print(f"\n{'='*60}")
    print(f"  推理完成! 结果保存在: {args.output_dir}/")
    if hr_frames is not None:
        sr_psnr = np.mean([calculate_psnr(sr_frames[t].clamp(0, 1), hr_frames[t])
                           for t in range(T)])
        bic_psnr = np.mean([calculate_psnr(bicubic_frames[t], hr_frames[t])
                            for t in range(T)])
        print(f"  SR PSNR: {sr_psnr:.2f} dB")
        print(f"  Bicubic PSNR: {bic_psnr:.2f} dB")
        print(f"  提升: {sr_psnr - bic_psnr:+.2f} dB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
