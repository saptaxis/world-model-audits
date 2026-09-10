#!/bin/bash
# Train DreamerV3 on LunarLander from state vectors.
# Usage: bash dreamerv3/scripts/train.sh [seed] [extra_args...]
#
# Examples:
#   bash dreamerv3/scripts/train.sh 42
#   bash dreamerv3/scripts/train.sh 42 trainer.steps=1000000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/env.sh"

SEED=${1:-42}
shift 2>/dev/null

LOGDIR="${WMA_DATA_ROOT}/world-model-audit-runs/dreamerv3/s${SEED}"

echo "Training DreamerV3 on LunarLander (state vectors)"
echo "  Seed: ${SEED}"
echo "  Logdir: ${LOGDIR}"
echo ""

cd "${REPO_ROOT}/dreamerv3/vendor/r2dreamer"

python train.py \
    env=lunar_lander \
    model.rep_loss=dreamer \
    seed=${SEED} \
    logdir=${LOGDIR} \
    "$@"
