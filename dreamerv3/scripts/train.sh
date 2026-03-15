#!/bin/bash
# Train DreamerV3 on LunarLander from state vectors.
# Usage: bash dreamerv3/scripts/train.sh [seed] [extra_args...]
#
# Examples:
#   bash dreamerv3/scripts/train.sh 42
#   bash dreamerv3/scripts/train.sh 42 trainer.steps=1000000

SEED=${1:-42}
shift 2>/dev/null

LOGDIR="/media/hdd1/physics-priors-latent-space/world-model-audit-runs/dreamerv3/s${SEED}"

echo "Training DreamerV3 on LunarLander (state vectors)"
echo "  Seed: ${SEED}"
echo "  Logdir: ${LOGDIR}"
echo ""

cd "$(dirname "$0")/../vendor/r2dreamer"

python train.py \
    env=lunar_lander \
    model.rep_loss=dreamer \
    seed=${SEED} \
    logdir=${LOGDIR} \
    "$@"
