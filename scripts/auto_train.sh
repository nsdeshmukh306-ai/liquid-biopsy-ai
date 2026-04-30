#!/bin/bash
# auto_train.sh — Watch for real_feature_matrix.csv, then re-run training pipeline.
# Polls every 5 minutes. Runs until real data is found and training completes.

set -euo pipefail

REAL_CSV=~/niraj/liquid_biopsy_ai/results/real_feature_matrix.csv
PIPELINE_LOG=~/niraj/liquid_biopsy_ai/.pipeline_log.txt
PROJ=~/niraj/liquid_biopsy_ai
LOGDIR=~/niraj/logs
MIN_ROWS=10   # minimum samples needed before training

export PATH="/apps/compilers/anaconda3-2025/bin:$PATH"

log() { echo "[$(date '+%T')] $*" | tee -a "$LOGDIR/auto_train.log"; }

log "auto_train.sh started — watching for $REAL_CSV"

while true; do
    if [[ -f "$REAL_CSV" ]]; then
        ROWS=$(wc -l < "$REAL_CSV")
        if (( ROWS > MIN_ROWS )); then
            log "Found real_feature_matrix.csv with $ROWS rows. Starting training pipeline."
            echo "CHECKPOINT 8a: real data detected ($ROWS rows) - $(date)" >> "$PIPELINE_LOG"

            cd "$PROJ"

            log "Step 1: preprocess real data"
            conda run -n workshop_nf python3 scripts/01_preprocess.py \
                --data-dir results \
                --out-dir results \
                --schema results/real_feature_matrix.csv \
                >> "$LOGDIR/auto_train.log" 2>&1 && log "Preprocess done"

            log "Step 2: train model on real data"
            conda run -n workshop_nf python3 scripts/03_train_model.py \
                --data-dir results \
                --model-dir model \
                --results-dir results \
                --epochs 100 \
                --batch-size 16 \
                >> "$LOGDIR/auto_train.log" 2>&1 && log "Training done"

            log "Step 3: evaluate"
            conda run -n workshop_nf python3 scripts/04_evaluate.py \
                --data-dir results \
                --model-path model/best_model.pt \
                --results-dir results \
                --figures-dir report/figures \
                >> "$LOGDIR/auto_train.log" 2>&1 && log "Evaluation done"

            log "Step 4: visualise"
            conda run -n workshop_nf python3 scripts/05_visualise.py \
                --results-dir results \
                --figures-dir report/figures \
                --features-csv results/cfDNA_features.csv \
                >> "$LOGDIR/auto_train.log" 2>&1 && log "Visualisation done"

            echo "CHECKPOINT 8: auto_train COMPLETE - $(date)" >> "$PIPELINE_LOG"
            log "All training steps complete!"
            exit 0
        else
            log "CSV found but only $ROWS rows (need >$MIN_ROWS). Waiting..."
        fi
    else
        log "Waiting for $REAL_CSV... (checking every 5 min)"
    fi

    sleep 300
done
