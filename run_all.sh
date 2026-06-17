#!/bin/bash
# 自动串行训练三个模型 (30 epochs each)
# 用法: nohup bash run_all.sh > logs/run_all.log 2>&1 &

VIMEO_ROOT="../vimeo_septuplet"
EPOCHS=30
BATCH=48
WORKERS=8

MODELS=("basicvsr_light" "vsr_3dcnn" "vsr_convlstm")
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

for MODEL in "${MODELS[@]}"; do
    echo "============================================"
    echo "  Starting: $MODEL ($(date))"
    echo "============================================"

    python -u train.py \
        --model "$MODEL" \
        --vimeo_root "$VIMEO_ROOT" \
        --dataset vimeo \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --num_workers "$WORKERS" \
        --save_dir results \
        --save_interval 5 \
        > "$LOGS_DIR/${MODEL}.log" 2>&1

    echo ""
    echo "  [$(date)] $MODEL done (exit=$?)."
    echo ""

    python -c "
import json, os
f='results/${MODEL}/metrics_history.json'
if os.path.exists(f):
    d=json.load(open(f))
    best=max(e['psnr'] for e in d)
    last=[e['psnr'] for e in d[-3:]]
    print(f'  {d[-1][\"epoch\"]} epochs | Best PSNR: {best:.2f} dB | Last 3: {last}')
"
    echo ""
done

echo "=========== ALL DONE ============="
echo "$(date)"
echo ""
python -c "
import json, os
for m in ['basicvsr_light','vsr_3dcnn','vsr_convlstm']:
    f=f'results/{m}/metrics_history.json'
    if os.path.exists(f):
        d=json.load(open(f))
        best=max(e['psnr'] for e in d)
        print(f'  {m}: best={best:.2f} dB ({len(d)} epochs)')
"
