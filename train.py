"""
train.py — 统一训练脚本 (50 Epoch 增强版)
========================================
支持多模型训练:
- basicvsr / basicvsr_light: 光流对齐 + 双向传播
- vsr_3dcnn: 3D CNN Early Fusion
- vsr_convlstm: ConvLSTM 时序建模

新增功能 (50 Epoch 实验):
- 多阶段学习率调度: Phase1(1-30) Cosine 2e-4→1e-7, Phase2(31-40) 1e-4, Phase3(41-50) 5e-5
- 详细日志: 每 epoch 自动保存 PSNR/SSIM/Loss/LR 到 JSON 日志
- 断点续训: --resume 从 checkpoint 恢复训练
- 训练曲线数据: 自动保存 metrics_history.json 供可视化使用

损失函数: Charbonnier Loss (L1 平滑版)
"""

import os
import sys
import time
import json
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

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


def create_scheduler(optimizer, total_epochs, lr_schedule='cosine'):
    """
    创建学习率调度器。

    lr_schedule:
    - 'cosine': CosineAnnealing 2e-4 → 1e-7 (标准单阶段)
    - 'multi_phase': 三阶段调度, 搭配 adjust_lr_for_phase() 使用
    - 'step': MultiStepLR 在 [20, 35, 45] 衰减 0.5
    """
    if lr_schedule == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)
    elif lr_schedule == 'multi_phase':
        return CosineAnnealingLR(optimizer, T_max=min(total_epochs, 30), eta_min=1e-7)
    elif lr_schedule == 'step':
        return MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.5)
    else:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)


def adjust_lr_for_phase(optimizer, epoch, total_epochs, base_lr, lr_schedule):
    """
    手动调整学习率 (multi_phase 策略专用)。
    Phase1(1-30): Cosine 2e-4→1e-7 (由 scheduler 自动处理)
    Phase2(31-40): 固定 1e-4
    Phase3(41-50): 固定 5e-5
    """
    if lr_schedule != 'multi_phase' or total_epochs <= 30:
        return
    if epoch == 31:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * 0.5
        print(f"  >> Phase 2: LR reset to {base_lr * 0.5:.1e} (epoch 31-40)")
    elif epoch == 41:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * 0.25
        print(f"  >> Phase 3: LR reset to {base_lr * 0.25:.1e} (epoch 41-{total_epochs})")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    log_interval=50, scaler=None, use_amp=False):
    """训练一个 epoch, 支持 AMP 混合精度加速"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (lr_frames, hr_frames) in enumerate(dataloader):
        lr_frames = lr_frames.to(device)
        hr_frames = hr_frames.to(device)

        if use_amp and scaler is not None:
            with autocast():
                sr_frames = model(lr_frames)
                loss = criterion(sr_frames, hr_frames)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            sr_frames = model(lr_frames)
            loss = criterion(sr_frames, hr_frames)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.6f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, epoch=0, scale=4, use_amp=False, val_limit=100):
    """评估模型 (PSNR + SSIM + 可视化), 返回平均PSNR和SSIM"""
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0

    for batch_idx, (lr_frames, hr_frames) in enumerate(dataloader):
        lr_frames = lr_frames.to(device)
        hr_frames = hr_frames.to(device)
        if use_amp:
            with autocast():
                sr_frames = model(lr_frames)
        else:
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
                lr_img.unsqueeze(0), scale_factor=scale, mode='bicubic', align_corners=False
            ).squeeze(0).clamp(0, 1)

            save_comparison(lr_img, bicubic_img, sr_img, hr_img,
                            os.path.join(save_dir, f'epoch{epoch}_seq{batch_idx}.png'))

    return total_psnr / max(count, 1), total_ssim / max(count, 1)


def save_metrics_log(metrics_history, save_dir):
    """保存训练指标历史到 JSON 和 CSV 文件"""
    json_path = os.path.join(save_dir, 'metrics_history.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)

    csv_path = os.path.join(save_dir, 'training_log.csv')
    if metrics_history:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(metrics_history)


def main():
    parser = argparse.ArgumentParser(description='VSR 训练 (50 Epoch 增强版)')

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
    parser.add_argument('--lr', type=float, default=2e-4, help='初始学习率')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'multi_phase', 'step'],
                        help='学习率调度策略: cosine=标准, multi_phase=三阶段, step=阶梯衰减')
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复训练')
    parser.add_argument('--amp', action='store_true', default=True, help='启用 AMP 混合精度 (默认开启)')
    parser.add_argument('--no_amp', action='store_true', help='禁用 AMP 混合精度')
    parser.add_argument('--val_limit', type=int, default=100, help='验证集最多评估 N 个 batch (默认100, 0=全部)')
    # 输出
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载进程数')
    parser.add_argument('--save_dir', type=str, default='results', help='保存目录')
    parser.add_argument('--save_interval', type=int, default=5, help='保存 checkpoint 间隔 (epoch)')
    parser.add_argument('--log_interval', type=int, default=50, help='训练日志打印间隔 (batch)')
    parser.add_argument('--tag', type=str, default='', help='实验标签')

    args = parser.parse_args()

    print("=" * 60)
    print(f"  VSR 训练 — 模型: {args.model} | 调度: {args.lr_schedule}")
    print("=" * 60)

    device = setup_device()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    save_dir = os.path.join(args.save_dir, args.model + (f'_{args.tag}' if args.tag else ''))
    os.makedirs(save_dir, exist_ok=True)

    # 数据
    print("\n[1/4] 准备数据集...")
    train_loader, test_loader = get_dataloader(
        dataset_type=args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, vimeo_root=args.vimeo_root, test_root=args.test_root,
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
    scheduler = create_scheduler(optimizer, args.epochs, args.lr_schedule)

    start_epoch = 1
    best_psnr = 0
    metrics_history = []

    # 断点续训
    if args.resume and os.path.exists(args.resume):
        print(f"\n  [RESUME] 从 {args.resume} 恢复训练...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0)
        print(f"  恢复至 epoch {start_epoch}, best PSNR={best_psnr:.2f} dB")
        # 加载之前的 metrics history
        json_path = os.path.join(save_dir, 'metrics_history.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                metrics_history = json.load(f)

    use_amp = args.amp and not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    print(f"损失: Charbonnier | 优化器: Adam (lr={args.lr}) | 调度: {args.lr_schedule} | AMP: {use_amp}")
    if args.lr_schedule == 'multi_phase':
        print(f"  Phase1 (1-{min(30, args.epochs)}): Cosine {args.lr}→1e-7")
        if args.epochs > 30:
            print(f"  Phase2 (31-{min(40, args.epochs)}): 固定 {args.lr * 0.5:.1e}")
        if args.epochs > 40:
            print(f"  Phase3 (41-{args.epochs}): 固定 {args.lr * 0.25:.1e}")

    # 训练循环
    print(f"\n[4/4] 开始训练 (epoch {start_epoch} → {args.epochs})...")
    print(f"  日志: {save_dir}/metrics_history.json")
    print(f"  CSV:  {save_dir}/training_log.csv")
    print("-" * 60)
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 多阶段 LR 手动切换
        adjust_lr_for_phase(optimizer, epoch, args.epochs, args.lr, args.lr_schedule)

        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                   epoch, log_interval=args.log_interval,
                                   scaler=scaler, use_amp=use_amp)

        # multi_phase 模式下只在 Phase1 使用 scheduler, 后续阶段手动控制
        if args.lr_schedule != 'multi_phase' or epoch <= 30:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        avg_psnr, avg_ssim = evaluate(
            model, test_loader, device,
            save_dir=save_dir if epoch % args.save_interval == 0 else None,
            epoch=epoch, scale=args.scale, use_amp=use_amp, val_limit=args.val_limit)

        # 保存指标
        entry = {
            'epoch': epoch,
            'loss': round(avg_loss, 6),
            'psnr': round(avg_psnr, 2),
            'ssim': round(avg_ssim, 4),
            'lr': current_lr,
            'time_s': round(epoch_time, 1),
        }
        metrics_history.append(entry)
        save_metrics_log(metrics_history, save_dir)

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
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': avg_psnr, 'ssim': avg_ssim,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))

    total_time = time.time() - train_start
    print(f"\n训练完成! 耗时 {total_time:.1f}s ({total_time/60:.1f}min) | 最佳 PSNR: {best_psnr:.2f} dB")
    print(f"模型保存在: {save_dir}/")
    print(f"指标日志: {save_dir}/metrics_history.json")


if __name__ == '__main__':
    main()
