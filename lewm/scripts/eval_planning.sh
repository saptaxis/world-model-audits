#!/bin/bash
# Launcher for JEPA CEM planning on Lunar Lander.
#
# Usage:
#     bash lewm/scripts/eval_planning.sh <mode> <policy_ckpt_name> [<state_head_path>] [<solver>] [<extra hydra overrides>]
#
# Modes: replay | synthetic
#
# <policy_ckpt_name>: path relative to $STABLEWM_HOME, without `_object.ckpt` suffix.
#   Example: synthetic-heuristic-fs10/lewm_lunarlander_synthetic_heuristic_fs10_epoch_30
#
# <state_head_path>: absolute path to state_head.pt (required for synthetic mode, ignored for replay).
#
# <solver>: cem (default) | icem
#
# Examples:
#     # Replay mode with CEM on heuristic-only ep30
#     bash lewm/scripts/eval_planning.sh replay \
#         synthetic-heuristic-fs10/lewm_lunarlander_synthetic_heuristic_fs10_epoch_30
#
#     # Synthetic (kinematic) mode with iCEM
#     bash lewm/scripts/eval_planning.sh synthetic \
#         synthetic-heuristic-fs10/lewm_lunarlander_synthetic_heuristic_fs10_epoch_30 \
#         /media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/state_head_epoch30/state_head.pt \
#         icem
set -e

MODE=${1:?"Usage: eval_planning.sh <replay|synthetic> <policy_ckpt> [state_head] [solver]"}
POLICY=${2:?"policy ckpt name required"}
STATE_HEAD=${3:-""}
SOLVER=${4:-cem}
shift $(( $# < 4 ? $# : 4 ))

export STABLEWM_HOME=/media/hdd1/physics-priors-latent-space/lunar-lander-data
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /workspace/world-model-audits/lewm/vendor/le-wm

CONFIG_NAME="lunarlander_${MODE}"

OVERRIDES=(policy=${POLICY} solver=${SOLVER})
if [ -n "$STATE_HEAD" ]; then
    OVERRIDES+=(state_head_path=${STATE_HEAD})
fi

# Also accept extra hydra overrides appended as remaining args
OVERRIDES+=("$@")

echo "=== Planning eval ==="
echo "  mode:       $MODE"
echo "  policy:     $POLICY"
echo "  state_head: $STATE_HEAD"
echo "  solver:     $SOLVER"
echo "===================="

python eval.py --config-name=${CONFIG_NAME} "${OVERRIDES[@]}"
