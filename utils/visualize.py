"""
utils/visualize.py — 实验可视化工具 (50 Epoch 实验专用)
========================================================
生成实验计划书要求的 12 张可视化图:

4.1 训练过程类:
  图A: 多模型训练曲线 (Loss + PSNR 双Y轴)
  图B: 学习率 vs PSNR 曲线
  图C: 逐Epoch验证SSIM曲线

4.2 模型对比类:
  图D: 多维度雷达图 (PSNR/SSIM/Speed/Params/Memory/Stability)
  图E: PSNR-参数量散点图
  图F: 各序列PSNR分布箱线图

4.3 效率与剖分类:
  图G: 推理速度堆叠柱状图
  图H: 显存占用 vs 帧数曲线
  图I: PSNR-FPS 权衡散点图

4.4 长视频类 (创新拓展):
  图J: 长视频时间一致性指标
  图K: 长视频逐帧PSNR曲线
  图L: 长视频显存占用曲线

使用方法:
  from utils.visualize import plot_all_from_logs
  plot_all_from_logs(logs_dict, results_df, long_video_data, save_root='results/figures')
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


def _load_logs(log_dir, model_name):
    """从保存目录加载 metrics_history.json"""
    path = os.path.join(log_dir, 'metrics_history.json')
    if os.path.exists(path):
        with open(path) as f:
            return model_name, json.load(f)
    return model_name, []


# ============================================================
# 4.1 训练过程类
# ============================================================

def plot_training_curves(all_logs, save_path):
    """
    图A: 多模型训练曲线 (双Y轴: Loss对数 + PSNR)
    all_logs: {model_name: [{epoch, loss, psnr, ssim, lr, time_s}, ...], ...}
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    colors = {'basicvsr_light': '#e74c3c', 'vsr_3dcnn': '#3498db',
              'vsr_convlstm': '#2ecc71',
              'BasicVSR': '#e74c3c', '3D CNN': '#3498db',
              'ConvLSTM': '#2ecc71'}
    markers = {'basicvsr_light': 'o', 'vsr_3dcnn': 's', 'vsr_convlstm': '^',
               'BasicVSR': 'o', '3D CNN': 's', 'ConvLSTM': '^'}

    for model_name, log in all_logs.items():
        if not log:
            continue
        epochs = [e['epoch'] for e in log]
        losses = [e['loss'] for e in log]
        psnrs = [e['psnr'] for e in log]

        c = colors.get(model_name, '#333333')
        m = markers.get(model_name, 'o')

        ax1.plot(epochs, losses, color=c, linestyle='--', alpha=0.5,
                 label=f'{model_name} Loss')
        ax2.plot(epochs, psnrs, color=c, linewidth=2, marker=m, markersize=4,
                 label=f'{model_name} PSNR')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Charbonnier Loss', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_yscale('log')
    ax1.axvline(x=30, color='gray', linestyle=':', alpha=0.5,
                label='Phase I/II Boundary')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               fontsize=9, ncol=2)

    plt.title('Training Curves: Loss & PSNR vs Epoch', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图A 已保存: {save_path}")


def plot_lr_vs_psnr(all_logs, save_path):
    """
    图B: 学习率 vs PSNR 曲线
    展示多阶段学习率调度与PSNR收敛的关系
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    for model_name, log in all_logs.items():
        if not log:
            continue
        epochs = [e['epoch'] for e in log]
        psnrs = [e['psnr'] for e in log]
        lrs = [e['lr'] for e in log]

        ax1.plot(epochs, psnrs, linewidth=2, marker='.', label=f'{model_name} PSNR')
        ax2.plot(epochs, lrs, linestyle='--', alpha=0.6, linewidth=1.5,
                 label=f'{model_name} LR')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_yscale('log')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    plt.title('Learning Rate Schedule & PSNR Convergence', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图B 已保存: {save_path}")


def plot_ssim_curves(all_logs, save_path):
    """
    图C: 逐Epoch验证SSIM曲线
    对比PSNR与SSIM的趋势一致性
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, log in all_logs.items():
        if not log:
            continue
        epochs = [e['epoch'] for e in log]
        psnrs = [e['psnr'] for e in log]
        ssims = [e['ssim'] for e in log]

        axes[0].plot(epochs, psnrs, linewidth=2, marker='.', label=model_name)
        axes[1].plot(epochs, ssims, linewidth=2, marker='.', label=model_name)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    plt.suptitle('Per-Epoch Validation PSNR & SSIM', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图C 已保存: {save_path}")


# ============================================================
# 4.2 模型对比类
# ============================================================

def plot_radar_chart(metrics_dict, save_path):
    """
    图D: 多维度雷达图
    metrics_dict: {model_name: [psnr, ssim, speed, params, memory, stability]}
    所有值归一化到 [0, 1]，越大越好
    """
    categories = ['PSNR', 'SSIM', 'Speed', 'Params\n(lower=better)',
                  'Memory\n(lower=better)', 'Stability']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = {'basicvsr_light': '#e74c3c', 'vsr_3dcnn': '#3498db',
              'vsr_convlstm': '#2ecc71',
              'BasicVSR': '#e74c3c', '3D CNN': '#3498db',
              'ConvLSTM': '#2ecc71'}

    for model_name, values in metrics_dict.items():
        values_norm = values + values[:1]
        c = colors.get(model_name, '#333333')
        ax.plot(angles, values_norm, 'o-', linewidth=2, color=c, label=model_name)
        ax.fill(angles, values_norm, alpha=0.1, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Multi-Dimensional Model Comparison', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图D 已保存: {save_path}")


def plot_psnr_params_scatter(models_info, save_path):
    """
    图E: PSNR-参数量散点图
    models_info: [(name, psnr, params_m, time_s), ...]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [m[0] for m in models_info]
    psnrs = [m[1] for m in models_info]
    params = [m[2] for m in models_info]
    times = [m[3] for m in models_info]

    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i, (n, p, pm, t) in enumerate(models_info):
        ax.scatter(pm, p, s=max(t * 8, 80), c=colors[i % len(colors)],
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(n, (pm, p), textcoords="offset points", xytext=(0, 10),
                    fontsize=10, ha='center')

    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR vs Parameter Count (bubble size = training time)', fontsize=13)

    # Bicubic baseline reference line
    ax.axhline(y=22.0, color='gray', linestyle=':', alpha=0.5, label='Bicubic baseline (~22 dB)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图E 已保存: {save_path}")


def plot_sequence_boxplot(results_df, save_path):
    """
    图F: 各序列PSNR分布箱线图
    results_df: DataFrame with columns ['sequence', 'model', 'psnr']
    """
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=results_df, x='sequence', y='psnr', hue='model',
                    palette='Set2', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_xlabel('Test Sequence', fontsize=12)
        ax.legend(title='Model', loc='lower left')
        ax.set_title('Per-Sequence PSNR Distribution', fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  图F 已保存: {save_path}")
    except ImportError:
        print("  [跳过] 图F 需要 seaborn, 请 `pip install seaborn`")


# ============================================================
# 4.3 效率与剖分类
# ============================================================

def plot_speed_breakdown(speed_data, save_path):
    """
    图G: 推理速度堆叠柱状图
    speed_data: {model: {'feature_extract': s, 'temporal': s, 'reconstruct': s, 'other': s}, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(speed_data.keys())
    components = ['feature_extract', 'temporal', 'reconstruct', 'other']
    labels = ['Feature Extract', 'Temporal Processing', 'Reconstruction', 'Other']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    bottom = np.zeros(len(models))
    for i, comp in enumerate(components):
        values = [speed_data[m].get(comp, 0) for m in models]
        ax.bar(models, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom += np.array(values)

    ax.set_ylabel('Time per Frame (ms)', fontsize=12)
    ax.set_title('Inference Speed Breakdown by Component', fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图G 已保存: {save_path}")


def plot_memory_vs_frames(memory_data, save_path):
    """
    图H: 显存占用 vs 帧数曲线
    memory_data: {model: [(T3, gb3), (T7, gb7), (T11, gb11), (T15, gb15), (T21, gb21)], ...}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, points in memory_data.items():
        ts = [p[0] for p in points]
        gbs = [p[1] for p in points]
        ax.plot(ts, gbs, 'o-', linewidth=2, markersize=8, label=model_name)

    ax.set_xlabel('Number of Input Frames (T)', fontsize=12)
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_title('GPU Memory vs Sequence Length', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图H 已保存: {save_path}")


def plot_psnr_fps_scatter(perf_data, save_path):
    """
    图I: PSNR-FPS 权衡散点图
    perf_data: [(model, psnr, fps, params_m), ...]
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, psnr, fps, params) in enumerate(perf_data):
        ax.scatter(fps, psnr, s=params * 80, alpha=0.7, edgecolors='black',
                   linewidth=0.5, zorder=5)
        ax.annotate(name, (fps, psnr), textcoords="offset points",
                    xytext=(0, 12), fontsize=11, ha='center')

    ax.set_xlabel('FPS (frames/sec, higher=better)', fontsize=12)
    ax.set_ylabel('PSNR (dB, higher=better)', fontsize=12)
    ax.set_title('PSNR-FPS Trade-off (bubble size = parameter count)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图I 已保存: {save_path}")


# ============================================================
# 4.4 长视频类 (创新拓展)
# ============================================================

def plot_long_video_summary(summary_data, save_path):
    """
    图J: 长视频时间一致性汇总指标
    summary_data: {method: {'avg_psnr': x, 'boundary_psnr': x, 'interior_psnr': x, 'lpips': x}, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(summary_data.keys())
    x = np.arange(len(methods))
    width = 0.2

    metrics_keys = ['avg_psnr', 'boundary_psnr', 'interior_psnr']
    labels = ['Avg PSNR', 'Boundary PSNR', 'Interior PSNR']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, (key, label, color) in enumerate(zip(metrics_keys, labels, colors)):
        values = [summary_data[m].get(key, 0) for m in methods]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Long Video Temporal Consistency: Method Comparison', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图J 已保存: {save_path}")


def plot_long_video_frame_psnr(psnr_data, save_path, window_size=7):
    """
    图K: 长视频逐帧PSNR曲线
    psnr_data: {method: [psnr_t0, psnr_t1, ...], ...}
    使用不同颜色标注窗口边界
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {'naive': '#e74c3c', 'stateful': '#2ecc71',
              'bicubic': '#95a5a6',
              '朴素滑动窗口': '#e74c3c', '状态传递滑动窗口': '#2ecc71',
              'Bicubic': '#95a5a6',
              'Naive Sliding': '#e74c3c', 'State Propagation': '#2ecc71'}

    for method, psnrs in psnr_data.items():
        T = len(psnrs)
        c = colors.get(method, '#333333')
        ax.plot(range(T), psnrs, linewidth=1.5, label=method, color=c, alpha=0.9)

    # 标注窗口边界
    stride = max(1, window_size - 2)
    boundaries = list(range(stride, 30, stride))
    for b in boundaries:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Per-Frame PSNR (30-frame Long Video, window={window_size})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图K 已保存: {save_path}")


def plot_long_video_memory(memory_curve, save_path):
    """
    图L: 长视频显存占用曲线
    memory_curve: {method: [(num_frames, gb), ...], ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, points in memory_curve.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, 'o-', linewidth=2, markersize=8, label=method)

    ax.set_xlabel('Video Length (frames)', fontsize=12)
    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_title('Long Video Processing: Memory Scaling', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图L 已保存: {save_path}")


# ============================================================
# 批量生成
# ============================================================

def plot_all_from_logs(save_root='results/figures'):
    """
    批量从 results/ 目录加载各模型日志并生成全部12张图。
    这是实验完成后的"一键可视化"入口。
    """
    os.makedirs(save_root, exist_ok=True)

    model_dirs = {
        'BasicVSR': 'results/basicvsr_light',
        '3D CNN': 'results/vsr_3dcnn',
        'ConvLSTM': 'results/vsr_convlstm',
    }

    all_logs = {}
    for name, d in model_dirs.items():
        _, log = _load_logs(d, name)
        if log:
            all_logs[name] = log
            print(f"  已加载 {name}: {len(log)} epochs, "
                  f"best PSNR={max(e['psnr'] for e in log):.2f} dB")

    if not all_logs:
        print("未找到训练日志。请先完成训练后再生成图表。")
        return

    print(f"\n生成 12 张可视化图表 → {save_root}/")
    print("=" * 60)

    # 4.1 训练过程类
    plot_training_curves(all_logs, os.path.join(save_root, 'A_training_curves.png'))
    plot_lr_vs_psnr(all_logs, os.path.join(save_root, 'B_lr_vs_psnr.png'))
    plot_ssim_curves(all_logs, os.path.join(save_root, 'C_ssim_curves.png'))

    # 4.2 模型对比类 (需要实际数据)
    print("\n  [图D-I, J-L 需要使用实际评估数据填充, 提供了函数接口]")

    print(f"\n全部图表已保存到: {save_root}/")
