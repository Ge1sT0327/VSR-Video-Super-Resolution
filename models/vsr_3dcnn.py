"""
models/vsr_3dcnn.py — 3D CNN VSR 模型
========================================
Early Fusion 策略:
- 共享的 2D 特征提取器逐帧提取特征
- 3D 卷积在时间维度上处理帧间信息
- PixelShuffle 上采样到 HR

这是最直接的 VSR 基线方法，优点是简单高效，
缺点是没有显式的运动补偿，对大幅运动效果差。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VSR3DCNN(nn.Module):
    """
    基于 3D 卷积的 VSR 模型 (Per-Frame + 3D Temporal Fusion)。

    输入: [B, T, C, H, W]  LR 帧序列
    输出: [B, T, C, H*scale, W*scale]  HR 帧序列

    架构:
    1. 共享的 2D 特征提取器逐帧处理 → [B*T, feat_c, H, W]
    2. 重塑为 5D → [B, feat_c, T, H, W]
    3. 3D 卷积残差块处理时空信息
    4. 重塑回 2D → [B*T, feat_c, H, W]
    5. PixelShuffle 上采样 + 全局残差
    """
    def __init__(self, scale=4, num_frames=7, base_channels=32):
        super().__init__()
        self.scale = scale
        self.num_frames = num_frames
        self.base_channels = base_channels

        # 共享的逐帧特征提取 (2D)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # 3D 卷积处理时空信息 (输入: [B, base_channels, T, H, W])
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
        )

        # 融合 + 重建 (2D)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # PixelShuffle 上采样
        current_ch = base_channels * 2
        up_layers = []
        if scale == 4:
            up_layers.extend([
                nn.Conv2d(current_ch, current_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_ch, current_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
        elif scale == 2:
            up_layers.extend([
                nn.Conv2d(current_ch, current_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
        up_layers.append(nn.Conv2d(current_ch, 3, 3, 1, 1))
        self.upsampler = nn.Sequential(*up_layers)

    def forward(self, lrs):
        B, T, C, H, W = lrs.shape

        # Step 1: 逐帧特征提取 (共享权重)
        # [B, T, C, H, W] → [B*T, C, H, W]
        lrs_2d = lrs.reshape(B * T, C, H, W)
        feats_2d = self.feat_extract(lrs_2d)  # [B*T, base_c, H, W]

        # Step 2: 重塑为 3D 格式
        # [B*T, base_c, H, W] → [B, base_c, T, H, W]
        feats_3d = feats_2d.reshape(B, T, self.base_channels, H, W)
        feats_3d = feats_3d.permute(0, 2, 1, 3, 4)  # [B, base_c, T, H, W]

        # Step 3: 3D 时空处理 (残差连接)
        feats_3d = feats_3d + self.temporal_conv(feats_3d)

        # Step 4: 回到 2D
        feats_3d = feats_3d.permute(0, 2, 1, 3, 4)  # [B, T, base_c, H, W]
        feats_2d = feats_3d.reshape(B * T, self.base_channels, H, W)

        # Step 5: 重建 + 上采样
        feats_2d = self.reconstruct(feats_2d)  # [B*T, 2*base_c, H, W]
        sr_flat = self.upsampler(feats_2d)  # [B*T, 3, H*s, W*s]
        sr = sr_flat.reshape(B, T, 3, H * self.scale, W * self.scale)

        # Step 6: 全局残差 (Bicubic 上采样)
        base = F.interpolate(
            lrs_2d, scale_factor=self.scale, mode='bilinear', align_corners=False
        ).reshape(B, T, 3, H * self.scale, W * self.scale)

        return sr + base
