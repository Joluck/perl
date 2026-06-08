#!/bin/bash
# Evaluate the merged model at outputs/test
# Usage: bash scripts/eval_test.sh

set -e

MODEL_PATH="/home/rwkv/jl/outmodel/lora-merge"
TASK_DIR="eval/data"
WORK_DIR="eval/lora-0.05"

mkdir -p "$TASK_DIR" "$WORK_DIR"

# NOTE: Place task jsonl files under $TASK_DIR before running.
# Supported tasks: aime2024, aime2025, aime2026, beyond_aime, amc2023,
#   math500, minerva, hmmt2025, gpqa_diamond, mmlu, mmlu_pro, mmlu_prox,
#   ceval, ifeval, ifbench
#aime2024,aime2025,amc2023,hmmt2025
python modules/eval/run_eval.py \
    --stage all \
    --tasks aime2024@32,aime2025@32,amc2023@32,hmmt2025@32,math500@4,minerva@4 \
    --task-dir "$TASK_DIR" \
    --model-path "$MODEL_PATH" \
    --backend offline \
    --tp-size 1 \
    --dp-size 4 \
    --temperature 1 \
    --max-tokens 32000 \
    --chat-template-model-path none \
    --output "$WORK_DIR/step01.jsonl" \
    --inference-output "$WORK_DIR/step02.jsonl" \
    --score-output "$WORK_DIR/step03_scores.jsonl" \
    --final-eval-output "$WORK_DIR/step03_metrics.jsonl" \
    --n-proc 32

echo "Evaluation complete. Metrics saved to $WORK_DIR/step03_metrics.jsonl"

#    --top-p 0.95 \