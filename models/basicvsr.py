"""
models/basicvsr.py — BasicVSR 模型
========================================
参考: "BasicVSR: The Search for Essential Components in
       Video Super-Resolution and Beyond" (CVPR 2021)

核心组件:
1. SimpleFlowNet — 简化光流估计
2. 双向循环传播 — 前向+后向时序信息传递
3. PixelShuffle — 亚像素卷积上采样
4. 全局残差学习 — 模型学习 Bicubic 基础上的细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import flow_warp


class ResidualBlock(nn.Module):
    """残差块: x → Conv→ReLU→Conv → (+x)"""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class ResidualBlockGroup(nn.Module):
    """多个残差块堆叠"""
    def __init__(self, channels=64, num_blocks=5):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class SimpleFlowNet(nn.Module):
    """
    简化光流估计网络。
    输入: 两帧拼接 [B, 6, H, W]
    输出: 光流场 [B, 2, H, W]
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, 1, 1),
        )

    def forward(self, ref, target):
        x = torch.cat([ref, target], dim=1)
        feat = self.encoder(x)
        flow = self.decoder(feat)
        if flow.shape[2:] != ref.shape[2:]:
            flow = F.interpolate(flow, size=ref.shape[2:],
                                 mode='bilinear', align_corners=False)
        return flow


class PixelShuffleUpsampler(nn.Module):
    """Pixel Shuffle 上采样: [B, C, H, W] → [B, 3, H*scale, W*scale]"""
    def __init__(self, channels=64, scale=4):
        super().__init__()
        self.scale = scale
        layers = []
        if scale == 4:
            layers.extend([
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
        elif scale == 2:
            layers.extend([
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(channels, 3, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BasicVSR(nn.Module):
    """
    BasicVSR — 双向循环视频超分辨率网络。

    输入: [B, T, C, H, W]  LR 帧序列
    输出: [B, T, C, H*scale, W*scale]  HR 帧序列

    流程:
    1. 逐帧特征提取
    2. 后向传播 (t = T-1 → 0): 传递未来信息
    3. 前向传播 (t = 0 → T-1): 传递过去信息
    4. 双向特征融合 + PixelShuffle 上采样
    """
    def __init__(self, mid_channels=64, num_blocks=5, scale=4):
        super().__init__()
        self.mid_channels = mid_channels
        self.scale = scale

        self.flow_net = SimpleFlowNet()

        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        )

        self.forward_trunk = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResidualBlockGroup(mid_channels, num_blocks),
        )

        self.backward_trunk = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResidualBlockGroup(mid_channels, num_blocks),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

        self.upsampler = PixelShuffleUpsampler(mid_channels, scale)

    def forward(self, lrs):
        B, T, C, H, W = lrs.shape

        # Step 1: 逐帧特征提取
        feats = [self.feat_extract(lrs[:, t]) for t in range(T)]

        # Step 2: 后向传播
        backward_feats = [None] * T
        h_backward = torch.zeros(B, self.mid_channels, H, W, device=lrs.device)
        for t in range(T - 1, -1, -1):
            if t < T - 1:
                flow = self.flow_net(lrs[:, t + 1], lrs[:, t])
                h_backward = flow_warp(h_backward, flow)
            h_input = torch.cat([h_backward, feats[t]], dim=1)
            h_backward = self.backward_trunk(h_input)
            backward_feats[t] = h_backward

        # Step 3: 前向传播
        forward_feats = [None] * T
        h_forward = torch.zeros(B, self.mid_channels, H, W, device=lrs.device)
        for t in range(T):
            if t > 0:
                flow = self.flow_net(lrs[:, t - 1], lrs[:, t])
                h_forward = flow_warp(h_forward, flow)
            h_input = torch.cat([h_forward, feats[t]], dim=1)
            h_forward = self.forward_trunk(h_input)
            forward_feats[t] = h_forward

        # Step 4: 融合 + 上采样
        outputs = []
        for t in range(T):
            merged = torch.cat([forward_feats[t], backward_feats[t]], dim=1)
            fused = self.fusion(merged)
            base = F.interpolate(lrs[:, t], scale_factor=self.scale,
                                 mode='bilinear', align_corners=False)
            hr = self.upsampler(fused) + base
            outputs.append(hr)

        return torch.stack(outputs, dim=1)


class BasicVSRLight(BasicVSR):
    """轻量版 BasicVSR (mid_channels=32, num_blocks=3)"""
    def __init__(self, scale=4):
        super().__init__(mid_channels=32, num_blocks=3, scale=scale)
