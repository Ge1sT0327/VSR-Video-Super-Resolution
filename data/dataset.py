"""
data/dataset.py — 统一数据加载模块
========================================
支持:
1. TinyDataset — 合成数据 (快速验证流程)
2. Vimeo90KDataset — Vimeo-90K Septuplet 数据集
3. 测试集加载器 (VID4, UDM10 等)

VSR 数据核心:
- HR (高分辨率) = Ground Truth
- LR (低分辨率) = HR Bicubic 4x 下采样
- 模型从 LR 序列恢复 HR 序列
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class TinyDataset(Dataset):
    """
    合成 Tiny Dataset — 快速验证 VSR 流水线。

    生成原理:
    1. 创建随机彩色图案 + 棋盘格纹理作为基础场景
    2. 通过平移模拟视频运动，生成帧序列
    3. Bicubic 下采样得到 LR 帧

    注意: 合成数据只能验证代码正确性，不能训练出好模型。
    """
    def __init__(self, num_sequences=200, num_frames=7, hr_size=256, scale=4):
        super().__init__()
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.scale = scale
        print(f"[TinyDataset] {num_sequences} 序列, 每序列 {num_frames} 帧, "
              f"HR={hr_size}x{hr_size}, LR={self.lr_size}x{self.lr_size}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        torch.manual_seed(idx)

        canvas_size = self.hr_size + 40
        base = torch.zeros(3, canvas_size, canvas_size)

        # 随机色块
        for _ in range(np.random.randint(5, 15)):
            c = torch.rand(3, 1, 1)
            y1 = np.random.randint(0, canvas_size - 20)
            x1 = np.random.randint(0, canvas_size - 20)
            h = np.random.randint(10, canvas_size // 3)
            w = np.random.randint(10, canvas_size // 3)
            y2 = min(y1 + h, canvas_size)
            x2 = min(x1 + w, canvas_size)
            base[:, y1:y2, x1:x2] = c

        # 棋盘格高频纹理
        checker = torch.zeros(1, canvas_size, canvas_size)
        for i in range(0, canvas_size, 8):
            for j in range(0, canvas_size, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    checker[:, i:i+8, j:j+8] = 0.15
        base = (base + checker).clamp(0, 1)

        # 平移生成帧序列
        hr_frames = []
        for t in range(self.num_frames):
            dy = t * 2 + np.random.randint(-1, 2)
            dx = t * 3 + np.random.randint(-1, 2)
            y_start = max(0, min(20 + dy, canvas_size - self.hr_size))
            x_start = max(0, min(20 + dx, canvas_size - self.hr_size))
            frame = base[:, y_start:y_start + self.hr_size,
                            x_start:x_start + self.hr_size]
            hr_frames.append(frame)

        hr_frames = torch.stack(hr_frames)

        lr_frames = torch.nn.functional.interpolate(
            hr_frames, size=(self.lr_size, self.lr_size),
            mode='bicubic', align_corners=False
        ).clamp(0, 1)

        return lr_frames, hr_frames


class Vimeo90KDataset(Dataset):
    """
    Vimeo-90K Septuplet 数据集。

    目录结构:
    vimeo_septuplet/
    ├── sequences/
    │   ├── 00001/
    │   │   ├── 0001/
    │   │   │   ├── im1.png ~ im7.png
    │   │   │   └── ...
    ├── sep_trainlist.txt
    └── sep_testlist.txt
    """
    def __init__(self, root_dir, split='train', num_frames=7,
                 scale=4, patch_size=64):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.scale = scale
        self.patch_size = patch_size
        self.hr_patch_size = patch_size * scale

        list_file = os.path.join(root_dir, f'sep_{split}list.txt')
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.sequences = [line.strip() for line in f if line.strip()]
        else:
            seq_dir = os.path.join(root_dir, 'sequences')
            self.sequences = []
            if os.path.exists(seq_dir):
                for folder in sorted(os.listdir(seq_dir)):
                    folder_path = os.path.join(seq_dir, folder)
                    if os.path.isdir(folder_path):
                        for subfolder in sorted(os.listdir(folder_path)):
                            self.sequences.append(f"{folder}/{subfolder}")

        print(f"[Vimeo-90K] {split}: {len(self.sequences)} 序列, "
              f"{num_frames} 帧/序列, scale=x{scale}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = os.path.join(self.root_dir, 'sequences', self.sequences[idx])

        hr_frames = []
        for i in range(1, self.num_frames + 1):
            img = Image.open(os.path.join(seq_path, f'im{i}.png')).convert('RGB')
            hr_frames.append(TF.to_tensor(img))

        hr_frames = torch.stack(hr_frames)

        if self.split == 'train':
            hr_frames = self._random_crop(hr_frames)
            if np.random.random() > 0.5:
                hr_frames = torch.flip(hr_frames, dims=[-1])
            if np.random.random() > 0.5:
                hr_frames = torch.flip(hr_frames, dims=[-2])

        T, C, H, W = hr_frames.shape
        lr_frames = torch.nn.functional.interpolate(
            hr_frames, size=(H // self.scale, W // self.scale),
            mode='bicubic', align_corners=False
        ).clamp(0, 1)

        return lr_frames, hr_frames

    def _random_crop(self, frames):
        _, _, H, W = frames.shape
        h_start = np.random.randint(0, max(1, H - self.hr_patch_size + 1))
        w_start = np.random.randint(0, max(1, W - self.hr_patch_size + 1))
        return frames[:, :,
                       h_start:h_start + self.hr_patch_size,
                       w_start:w_start + self.hr_patch_size]


class VideoTestDataset(Dataset):
    """
    标准测试数据集加载器 (VID4, UDM10 等)。

    测试集结构:
    dataset_root/
    ├── calendar/
    │   ├── 0001.png ~ 0041.png  (HR 帧序列)
    ├── city/
    ├── foliage/
    └── walk/

    每个子目录是一个视频序列的全部帧。
    """
    def __init__(self, root_dir, scale=4, num_frames=7):
        super().__init__()
        self.root_dir = root_dir
        self.scale = scale
        self.num_frames = num_frames

        self.sequences = []
        if os.path.isdir(root_dir):
            for seq_name in sorted(os.listdir(root_dir)):
                seq_path = os.path.join(root_dir, seq_name)
                if os.path.isdir(seq_path):
                    frames = self._load_frames(seq_path)
                    if frames is not None and len(frames) >= 3:
                        self.sequences.append((seq_name, frames))

        print(f"[TestDataset] 共 {len(self.sequences)} 个测试序列")

    def _load_frames(self, seq_dir):
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        files = []
        for pat in patterns:
            files.extend(sorted(glob.glob(os.path.join(seq_dir, pat))))
        if not files:
            return None
        frames = []
        for f in files:
            img = Image.open(f).convert('RGB')
            frames.append(TF.to_tensor(img))
        return torch.stack(frames)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        name, hr_frames = self.sequences[idx]
        T, C, H, W = hr_frames.shape

        # 裁剪到 num_frames 的整数倍
        if T > self.num_frames:
            hr_frames = hr_frames[:self.num_frames]

        # 生成 LR
        lr_h, lr_w = H // self.scale, W // self.scale
        lr_frames = torch.nn.functional.interpolate(
            hr_frames, size=(lr_h, lr_w),
            mode='bicubic', align_corners=False
        ).clamp(0, 1)

        return name, lr_frames, hr_frames


def get_dataloader(dataset_type='tiny', batch_size=2, num_workers=0,
                   vimeo_root=None, test_root=None, **kwargs):
    """
    获取数据加载器的统一接口。

    返回:
        train_loader, test_loader (test_loader 可能为 None)
    """
    if dataset_type == 'vimeo' and vimeo_root and os.path.isdir(vimeo_root):
        train_dataset = Vimeo90KDataset(vimeo_root, split='train', **kwargs)
        test_dataset = Vimeo90KDataset(vimeo_root, split='test', **kwargs)
    elif dataset_type == 'test' and test_root and os.path.isdir(test_root):
        train_dataset = None
        test_dataset = VideoTestDataset(test_root, **kwargs)
    else:
        num_frames = kwargs.get('num_frames', 7)
        scale = kwargs.get('scale', 4)
        train_dataset = TinyDataset(
            num_sequences=200, num_frames=num_frames, scale=scale)
        test_dataset = TinyDataset(
            num_sequences=20, num_frames=num_frames, scale=scale)

    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)
        print(f"训练集: {len(train_dataset)} 样本, batch_size={batch_size}")

    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"测试集: {len(test_dataset)} 样本")

    return train_loader, test_loader
