"""
models/vsr_convlstm.py — ConvLSTM VSR 模型
========================================
RNN-based 方法:
- 使用 ConvLSTM 在时间维度上传递隐藏状态
- 逐帧特征提取 → ConvLSTM 时序传播 → 上采样重建
- 支持双向 ConvLSTM (前向+后向)

ConvLSTM 将标准 LSTM 中的全连接替换为卷积，
在保留空间结构的同时建模时序依赖。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元。
    将 LSTM 中的矩阵乘法替换为卷积操作。

    输入: x_t (当前帧特征), h_{t-1} (上一隐藏状态), c_{t-1} (上一细胞状态)
    输出: h_t, c_t
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, 1, padding)

    def forward(self, x, state):
        h, c = state
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # 分割为四个门
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BidirectionalConvLSTM(nn.Module):
    """
    双向 ConvLSTM。
    前向: t = 0 → T-1
    后向: t = T-1 → 0
    融合: 拼接前向和后向的隐藏状态
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.backward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x_seq):
        """
        x_seq: [T, B, C, H, W] (注意: T 在第一个维度)
        返回: [T, B, 2*hidden_dim, H, W]
        """
        T, B, C, H, W = x_seq.shape

        # 前向传播
        h_f = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        c_f = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        forward_outputs = []
        for t in range(T):
            h_f, c_f = self.forward_cell(x_seq[t], (h_f, c_f))
            forward_outputs.append(h_f)

        # 后向传播
        h_b = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        c_b = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        backward_outputs = [None] * T
        for t in range(T - 1, -1, -1):
            h_b, c_b = self.backward_cell(x_seq[t], (h_b, c_b))
            backward_outputs[t] = h_b

        # 融合
        outputs = []
        for t in range(T):
            out = torch.cat([forward_outputs[t], backward_outputs[t]], dim=1)
            outputs.append(out)

        return torch.stack(outputs)  # [T, B, 2*hidden_dim, H, W]


class VSRConvLSTM(nn.Module):
    """
    基于 ConvLSTM 的 VSR 模型。

    输入: [B, T, C, H, W]  LR 帧序列
    输出: [B, T, C, H*scale, W*scale]  HR 帧序列

    架构:
    1. 逐帧特征提取 (浅层 CNN)
    2. 双向 ConvLSTM 时序传播
    3. 融合层 + PixelShuffle 上采样
    4. 全局残差 (加上 Bicubic 上采样)
    """
    def __init__(self, scale=4, hidden_dim=32):
        super().__init__()
        self.scale = scale
        self.hidden_dim = hidden_dim

        # 特征提取
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # 双向 ConvLSTM
        self.bi_convlstm = BidirectionalConvLSTM(hidden_dim, hidden_dim)

        # 融合 + 残差处理
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # PixelShuffle 上采样
        upsampling = []
        current_ch = hidden_dim * 2
        if scale == 4:
            upsampling.extend([
                nn.Conv2d(current_ch, current_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
            upsampling.extend([
                nn.Conv2d(current_ch, current_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
        elif scale == 2:
            upsampling.extend([
                nn.Conv2d(current_ch, current_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])
        upsampling.append(nn.Conv2d(current_ch, 3, 3, 1, 1))
        self.upsampler = nn.Sequential(*upsampling)

    def forward(self, lrs):
        B, T, C, H, W = lrs.shape

        # 逐帧特征提取
        feats = []
        for t in range(T):
            f = self.feat_extract(lrs[:, t])  # [B, hidden_dim, H, W]
            feats.append(f)
        feats = torch.stack(feats)  # [T, B, hidden_dim, H, W]

        # 双向 ConvLSTM
        lstm_out = self.bi_convlstm(feats)  # [T, B, 2*hidden_dim, H, W]
        lstm_out = lstm_out.permute(1, 0, 2, 3, 4)  # [B, T, 2*hidden_dim, H, W]

        # 逐帧融合 + 上采样
        outputs = []
        for t in range(T):
            feat = self.fusion(lstm_out[:, t])  # [B, 2*hidden_dim, H, W]
            hr = self.upsampler(feat)  # [B, 3, H*s, W*s]

            # 全局残差
            base = F.interpolate(lrs[:, t], scale_factor=self.scale,
                                 mode='bilinear', align_corners=False)
            outputs.append(hr + base)

        return torch.stack(outputs, dim=1)  # [B, T, 3, H*s, W*s]
