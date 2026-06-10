#!/bin/bash
# ============================================================
# AutoDL 运行脚本 — 一键训练+评估
# 使用方法: bash run_autodl.sh [full|standard|evaluate]
# ============================================================
set -e

MODE=${1:-full}

echo "============================================"
echo "  VSR 实验 — AutoDL"
echo "  模式: $MODE"
echo "============================================"

# 检测数据集路径
if [ -d "/root/autodl-tmp/vimeo_septuplet" ]; then
    VIMEO_ROOT="/root/autodl-tmp/vimeo_septuplet"
elif [ -d "/root/autodl-fs/vimeo_septuplet" ]; then
    VIMEO_ROOT="/root/autodl-fs/vimeo_septuplet"
elif [ -d "./data/vimeo_septuplet" ]; then
    VIMEO_ROOT="./data/vimeo_septuplet"
else
    echo "未找到 Vimeo-90K 数据集"
    echo "请先运行: bash setup_autodl.sh"
    echo "或使用合成数据: bash run_autodl.sh standard"
    VIMEO_ROOT=""
fi

case $MODE in
    full)
        echo "[full 模式] 完整训练 (50 epochs, Vimeo-90K)"
        if [ -z "$VIMEO_ROOT" ]; then
            echo "错误: 需要 Vimeo-90K 数据集"
            exit 1
        fi
        python run_experiment.py \
            --mode full \
            --dataset vimeo \
            --vimeo_root "$VIMEO_ROOT" \
            --epochs 50 \
            --batch_size 4 \
            --num_frames 7 \
            --scale 4
        ;;
    standard)
        echo "[standard 模式] 标准实验 (合成数据, 5 epochs)"
        python run_experiment.py --mode standard
        ;;
    evaluate)
        echo "[evaluate 模式] 仅评估已训练的模型"
        python evaluate.py \
            --model_paths results/basicvsr_light/best_model.pth \
                         results/vsr_3dcnn/best_model.pth \
                         results/vsr_convlstm/best_model.pth \
            --model_types basicvsr_light vsr_3dcnn vsr_convlstm \
            --dataset "${VIMEO_ROOT:+vimeo}" \
            ${VIMEO_ROOT:+--vimeo_root "$VIMEO_ROOT"} \
            --output_dir results/final_evaluation
        ;;
    inference)
        echo "[inference 模式] 单模型推理演示"
        python inference.py \
            --model_path results/basicvsr_light/best_model.pth \
            --model_type basicvsr_light \
            --output_dir results/inference_output
        ;;
    *)
        echo "用法: bash run_autodl.sh [full|standard|evaluate|inference]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  实验完成!"
echo "  结果保存在: results/"
echo "============================================"
