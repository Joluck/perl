#!/bin/bash
# Evaluate the merged model at outputs/test
# Usage: bash scripts/eval_test.sh

set -e

MODEL_PATH="outputs/test"
TASK_DIR="outputs/nano_eval"
WORK_DIR="outputs/eval_test"

mkdir -p "$TASK_DIR" "$WORK_DIR"

# NOTE: Place task jsonl files under $TASK_DIR before running.
# Supported tasks: aime2024, aime2025, aime2026, beyond_aime, amc2023,
#   math500, minerva, hmmt2025, gpqa_diamond, mmlu, mmlu_pro, mmlu_prox,
#   ceval, ifeval, ifbench

python modules/eval/run_eval.py \
    --stage all \
    --tasks aime2024 \
    --pass-k 32 \
    --task-dir "$TASK_DIR" \
    --model-path "$MODEL_PATH" \
    --backend offline \
    --tp-size 1 \
    --dp-size 4 \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 32000 \
    --output "$WORK_DIR/step01.jsonl" \
    --inference-output "$WORK_DIR/step02.jsonl" \
    --score-output "$WORK_DIR/step03_scores.jsonl" \
    --final-eval-output "$WORK_DIR/step03_metrics.jsonl" \
    --n-proc 32

echo "Evaluation complete. Metrics saved to $WORK_DIR/step03_metrics.jsonl"
