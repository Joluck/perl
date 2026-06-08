#!/bin/bash
# Simple eval wrapper using modules/eval/run_eval.py
# Usage: MODEL_PATH=outputs/test bash scripts/eval_model.sh

set -e

MODEL_PATH="${MODEL_PATH:-outputs/test}"
TASK_DIR="outputs/nano_eval"
WORK_DIR="outputs/eval_$(basename "${MODEL_PATH}")"

mkdir -p "${TASK_DIR}" "${WORK_DIR}"

python modules/eval/run_eval.py \
    --stage all \
    --tasks aime2024@32,aime2025@32,amc2023@32,hmmt2025@32，math500@4,minerva@4 \
    --task-dir "${TASK_DIR}" \
    --model-path "${MODEL_PATH}" \
    --backend offline \
    --tp-size 1 \
    --dp-size 4 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 31744 \
    --chat-template-model-path none \
    --output "${WORK_DIR}/step01.jsonl" \
    --inference-output "${WORK_DIR}/step02.jsonl" \
    --score-output "${WORK_DIR}/step03_scores.jsonl" \
    --final-eval-output "${WORK_DIR}/step03_metrics.jsonl" \
    --n-proc 32

echo "Done. Metrics: ${WORK_DIR}/step03_metrics.jsonl"
