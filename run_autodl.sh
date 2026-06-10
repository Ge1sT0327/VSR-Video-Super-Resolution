#!/bin/bash
# ============================================================
# AutoDL 运行脚本 — 一键训练+评估
# 使用方法: bash run_autodl.sh [full|standard|evaluate|inference]
# ============================================================
set -e

MODE=${1:-full}

echo "============================================"
echo "  VSR 实验 — AutoDL"
echo "  模式: $MODE"
echo "============================================"

# 检测数据集路径 (按优先级)
VIMEO_ROOT=""
for path in \
    "/root/autodl-tmp/vimeo_septuplet" \
    "/root/autodl-pub/vimeo_septuplet" \
    "/root/autodl-fs/vimeo_septuplet" \
    "./data/vimeo_septuplet"; do
    if [ -d "$path/sequences" ]; then
        VIMEO_ROOT="$path"
        echo "数据集: $VIMEO_ROOT"
        break
    fi
done

case $MODE in
    full)
        echo ""
        echo "[full 模式] 完整训练 — Vimeo-90K, 50 epochs"
        echo "预计耗时: 8-12 小时 (RTX 3090)"

        if [ -z "$VIMEO_ROOT" ]; then
            echo ""
            echo "*** 错误: 未找到 Vimeo-90K 数据集 ***"
            echo "请先运行: bash setup_autodl.sh"
            echo "或使用合成数据: bash run_autodl.sh standard"
            exit 1
        fi

        python run_experiment.py \
            --mode full \
            --dataset vimeo \
            --vimeo_root "$VIMEO_ROOT" \
            --epochs 50 \
            --batch_size 16 \
            --num_frames 7 \
            --scale 4
        ;;

    standard)
        echo ""
        echo "[standard 模式] 标准实验 — 合成数据, 5 epochs"
        echo "预计耗时: 约 5 分钟"

        python run_experiment.py --mode standard
        ;;

    evaluate)
        echo ""
        echo "[evaluate 模式] 仅评估已训练模型"

        ARGS=""
        if [ -n "$VIMEO_ROOT" ]; then
            ARGS="--dataset vimeo --vimeo_root $VIMEO_ROOT"
        else
            ARGS="--dataset tiny"
        fi

        python evaluate.py \
            --model_paths results/basicvsr_light/best_model.pth \
                         results/vsr_3dcnn/best_model.pth \
                         results/vsr_convlstm/best_model.pth \
            --model_types basicvsr_light vsr_3dcnn vsr_convlstm \
            $ARGS \
            --output_dir results/final_evaluation
        ;;

    inference)
        echo ""
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
