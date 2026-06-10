"""
evaluate.py — 综合评估脚本
========================================
功能:
1. 加载训练好的模型，在测试集上评估
2. 计算 PSNR / SSIM 指标
3. 生成 LR / Bicubic / SR / HR 对比图
4. 输出详细的评估报告

支持多个模型对比评估。
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import create_model
from data.dataset import get_dataloader
from utils.common import (calculate_psnr, calculate_ssim, save_image,
                           save_comparison, setup_device, count_parameters, tensor_to_numpy)


@torch.no_grad()
def evaluate_model(model, dataloader, device, output_dir, scale=4):
    """
    完整评估一个模型。

    对每个测试序列:
    - 生成 SR 帧
    - 计算 PSNR/SSIM (与 HR 对比)
    - 计算 Bicubic 基线的 PSNR/SSIM
    - 保存对比图

    返回: 平均 PSNR, SSIM 字典
    """
    model.eval()
    results = {
        'sr_psnr': [], 'sr_ssim': [],
        'bic_psnr': [], 'bic_ssim': [],
        'seq_names': [],
    }

    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, data in enumerate(dataloader):
        # 根据数据集类型解析数据
        if len(data) == 2:
            name = f"seq_{batch_idx:03d}"
            lr_frames, hr_frames = data
        else:
            name, lr_frames, hr_frames = data
            name = name[0] if isinstance(name, (list, tuple)) else name

        lr_frames = lr_frames.to(device)
        hr_frames = hr_frames.to(device)

        # lr_frames 已经是 [B, T, C, H, W] (B=1)，直接送入模型
        sr_frames = model(lr_frames).squeeze(0).clamp(0, 1)
        hr_frames_2d = hr_frames.squeeze(0)
        lr_frames_2d = lr_frames.squeeze(0)

        # Bicubic 基线 (lr_frames: [1, T, C, H, W] → 逐帧上采样)
        B, T_lr, C, H_lr, W_lr = lr_frames.shape
        lr_2d = lr_frames.reshape(B * T_lr, C, H_lr, W_lr)
        bicubic_frames = F.interpolate(
            lr_2d, scale_factor=scale, mode='bicubic', align_corners=False
        ).clamp(0, 1)  # [B*T, C, H*s, W*s]
        bicubic_frames = bicubic_frames.reshape(B, T_lr, C, H_lr * scale, W_lr * scale).squeeze(0)

        T = sr_frames.shape[0]
        seq_sr_psnr, seq_sr_ssim = [], []
        seq_bic_psnr, seq_bic_ssim = [], []

        seq_out = os.path.join(output_dir, name)
        os.makedirs(seq_out, exist_ok=True)

        for t in range(T):
            sr_psnr = calculate_psnr(sr_frames[t], hr_frames_2d[t])
            sr_ssim = calculate_ssim(sr_frames[t], hr_frames_2d[t])
            bic_psnr = calculate_psnr(bicubic_frames[t], hr_frames_2d[t])
            bic_ssim = calculate_ssim(bicubic_frames[t], hr_frames_2d[t])

            seq_sr_psnr.append(sr_psnr)
            seq_sr_ssim.append(sr_ssim)
            seq_bic_psnr.append(bic_psnr)
            seq_bic_ssim.append(bic_ssim)

            # 保存 SR 和对比图
            save_image(sr_frames[t], os.path.join(seq_out, f'sr_{t:03d}.png'))
            save_comparison(lr_frames_2d[t], bicubic_frames[t], sr_frames[t], hr_frames_2d[t],
                            os.path.join(seq_out, f'compare_{t:03d}.png'))

        results['sr_psnr'].append(np.mean(seq_sr_psnr))
        results['sr_ssim'].append(np.mean(seq_sr_ssim))
        results['bic_psnr'].append(np.mean(seq_bic_psnr))
        results['bic_ssim'].append(np.mean(seq_bic_ssim))
        results['seq_names'].append(name)

        print(f"  {name}: SR PSNR={np.mean(seq_sr_psnr):.2f}dB, "
              f"Bicubic PSNR={np.mean(seq_bic_psnr):.2f}dB, "
              f"提升={np.mean(seq_sr_psnr) - np.mean(seq_bic_psnr):+.2f}dB")

    return results


def print_report(all_results):
    """打印综合评估报告"""
    print("\n" + "=" * 70)
    print("  综合评估报告")
    print("=" * 70)

    for model_name, results in all_results.items():
        avg_sr_psnr = np.mean(results['sr_psnr'])
        avg_sr_ssim = np.mean(results['sr_ssim'])
        avg_bic_psnr = np.mean(results['bic_psnr'])
        avg_bic_ssim = np.mean(results['bic_ssim'])
        delta_psnr = avg_sr_psnr - avg_bic_psnr
        delta_ssim = avg_sr_ssim - avg_bic_ssim

        print(f"\n--- {model_name} ---")
        print(f"  平均 SR PSNR:      {avg_sr_psnr:.2f} dB")
        print(f"  平均 SR SSIM:      {avg_sr_ssim:.4f}")
        print(f"  平均 Bicubic PSNR: {avg_bic_psnr:.2f} dB")
        print(f"  平均 Bicubic SSIM: {avg_bic_ssim:.4f}")
        print(f"  PSNR 提升:         {delta_psnr:+.2f} dB")
        print(f"  SSIM 提升:         {delta_ssim:+.4f}")

        if results['seq_names']:
            print(f"  各序列 PSNR:")
            for name, psnr, bic in zip(results['seq_names'],
                                        results['sr_psnr'], results['bic_psnr']):
                print(f"    {name}: {psnr:.2f} dB (Bicubic: {bic:.2f} dB, "
                      f"{'↑' if psnr > bic else '↓'}{abs(psnr-bic):.2f})")

    # 模型对比
    if len(all_results) > 1:
        print(f"\n--- 模型对比 (平均 PSNR) ---")
        for name, results in all_results.items():
            print(f"  {name}: {np.mean(results['sr_psnr']):.2f} dB")


def main():
    parser = argparse.ArgumentParser(description='VSR 评估')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help='模型路径列表 (支持多个模型对比)')
    parser.add_argument('--model_types', type=str, nargs='+', required=True,
                        help='对应的模型类型')
    parser.add_argument('--dataset', type=str, default='tiny', help='数据集类型')
    parser.add_argument('--test_root', type=str, default=None, help='测试集路径')
    parser.add_argument('--scale', type=int, default=4, help='超分倍率')
    parser.add_argument('--num_frames', type=int, default=7, help='帧数')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='输出目录')

    args = parser.parse_args()

    if len(args.model_paths) != len(args.model_types):
        print("错误: --model_paths 和 --model_types 数量必须相同")
        return

    device = setup_device()

    # 加载测试数据
    _, test_loader = get_dataloader(
        dataset_type=args.dataset if args.test_root is None else 'test',
        test_root=args.test_root,
        num_frames=args.num_frames, scale=args.scale)

    # 逐个评估模型
    all_results = {}
    for path, mtype in zip(args.model_paths, args.model_types):
        print(f"\n{'='*60}")
        print(f"  评估: {mtype}")
        print(f"{'='*60}")

        model = create_model(mtype, scale=args.scale).to(device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=True)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)
            print(f"  已加载: {path} (epoch={ckpt.get('epoch', '?')}, "
                  f"PSNR={ckpt.get('psnr', '?'):.2f}dB)")
        else:
            print(f"  * 模型文件不存在: {path}，使用随机初始化")

        count_parameters(model)
        results = evaluate_model(model, test_loader, device,
                                 os.path.join(args.output_dir, mtype), args.scale)
        all_results[mtype] = results

    print_report(all_results)


if __name__ == '__main__':
    main()
