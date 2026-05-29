#!/bin/bash
# Re-run only step03 (scoring) with updated judge logic
# Usage: bash scripts/eval_rescore.sh

set -e

WORK_DIR="outputs/eval_test_nosp"

python modules/eval/run_eval.py \
    --stage step03 \
    --output "$WORK_DIR/step01.jsonl" \
    --inference-output "$WORK_DIR/step02.jsonl" \
    --eval-input "$WORK_DIR/step02.jsonl" \
    --score-output "$WORK_DIR/step03_scores.jsonl" \
    --final-eval-output "$WORK_DIR/step03_metrics.jsonl" \
    --n-proc 32

echo "Rescoring complete. Metrics saved to $WORK_DIR/step03_metrics.jsonl"
