"""
train.py — 统一训练脚本
========================================
支持多模型训练:
- basicvsr / basicvsr_light: 光流对齐 + 双向传播
- vsr_3dcnn: 3D CNN Early Fusion
- vsr_convlstm: ConvLSTM 时序建模

损失函数: Charbonnier Loss (L1 平滑版)
优化器: Adam + CosineAnnealing
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import create_model
from data.dataset import get_dataloader
from utils.common import calculate_psnr, calculate_ssim, save_comparison, setup_device, count_parameters


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss = sqrt((x - y)² + eps²), L1 的平滑版本"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (lr_frames, hr_frames) in enumerate(dataloader):
        lr_frames = lr_frames.to(device)
        hr_frames = hr_frames.to(device)

        sr_frames = model(lr_frames)
        loss = criterion(sr_frames, hr_frames)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.6f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, epoch=0):
    """评估模型 (PSNR + SSIM + 可视化)"""
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0

    for batch_idx, (lr_frames, hr_frames) in enumerate(dataloader):
        lr_frames = lr_frames.to(device)
        hr_frames = hr_frames.to(device)
        sr_frames = model(lr_frames)

        B, T = sr_frames.shape[:2]
        for b in range(B):
            for t in range(T):
                total_psnr += calculate_psnr(sr_frames[b, t].clamp(0, 1), hr_frames[b, t])
                total_ssim += calculate_ssim(sr_frames[b, t].clamp(0, 1), hr_frames[b, t])
                count += 1

        if save_dir and batch_idx < 3:
            mid_t = T // 2
            lr_img = lr_frames[0, mid_t].cpu()
            hr_img = hr_frames[0, mid_t].cpu()
            sr_img = sr_frames[0, mid_t].clamp(0, 1).cpu()
            bicubic_img = torch.nn.functional.interpolate(
                lr_img.unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False
            ).squeeze(0).clamp(0, 1)

            save_comparison(lr_img, bicubic_img, sr_img, hr_img,
                            os.path.join(save_dir, f'epoch{epoch}_seq{batch_idx}.png'))

    return total_psnr / max(count, 1), total_ssim / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description='VSR 训练')

    # 模型
    parser.add_argument('--model', type=str, default='basicvsr_light',
                        choices=['basicvsr', 'basicvsr_light', 'vsr_3dcnn', 'vsr_convlstm'],
                        help='模型类型')
    # 数据
    parser.add_argument('--dataset', type=str, default='tiny',
                        choices=['tiny', 'vimeo', 'test'],
                        help='数据集类型')
    parser.add_argument('--vimeo_root', type=str, default=None, help='Vimeo-90K 根目录')
    parser.add_argument('--test_root', type=str, default=None, help='测试集根目录')
    parser.add_argument('--num_frames', type=int, default=7, help='帧数')
    parser.add_argument('--scale', type=int, default=4, help='超分倍率')
    # 训练
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    # 输出
    parser.add_argument('--save_dir', type=str, default='results', help='保存目录')
    parser.add_argument('--save_interval', type=int, default=2, help='保存间隔')
    parser.add_argument('--tag', type=str, default='', help='实验标签')

    args = parser.parse_args()

    print("=" * 60)
    print(f"  VSR 训练 — 模型: {args.model}")
    print("=" * 60)

    device = setup_device()
    save_dir = os.path.join(args.save_dir, args.model + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(save_dir, exist_ok=True)

    # 数据
    print("\n[1/4] 准备数据集...")
    train_loader, test_loader = get_dataloader(
        dataset_type=args.dataset, batch_size=args.batch_size,
        num_workers=0, vimeo_root=args.vimeo_root, test_root=args.test_root,
        num_frames=args.num_frames, scale=args.scale)

    if train_loader is None:
        print("错误: 无训练数据。请使用 --dataset tiny 或指定数据路径。")
        return

    # 模型
    print("\n[2/4] 创建模型...")
    model = create_model(args.model, scale=args.scale).to(device)
    count_parameters(model)

    # 训练配置
    print("\n[3/4] 配置训练...")
    criterion = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    print(f"损失: Charbonnier | 优化器: Adam (lr={args.lr}) | 调度: CosineAnnealing")

    # 训练循环
    print(f"\n[4/4] 开始训练 ({args.epochs} epochs)...")
    print("-" * 60)
    best_psnr = 0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        avg_psnr, avg_ssim = evaluate(
            model, test_loader, device,
            save_dir=save_dir if epoch % args.save_interval == 0 else None,
            epoch=epoch)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LR: {current_lr:.2e}")
        print(f"{'='*60}\n")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': avg_psnr, 'ssim': avg_ssim, 'model': args.model,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  * 新最佳模型 (PSNR: {avg_psnr:.2f} dB)")

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'psnr': avg_psnr, 'ssim': avg_ssim,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))

    total_time = time.time() - train_start
    print(f"\n训练完成! 耗时 {total_time:.1f}s ({total_time/60:.1f}min) | 最佳 PSNR: {best_psnr:.2f} dB")
    print(f"模型保存在: {save_dir}/")


if __name__ == '__main__':
    main()
