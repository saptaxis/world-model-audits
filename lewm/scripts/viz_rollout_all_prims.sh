#!/bin/bash
# Rollout viz on all primitives datasets using the synthetic-heuristic encoder.
# Tests generalization: encoder/predictor trained on heuristic, evaluated on prims.
set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

MODEL=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/lewm_lunarlander_synthetic_heuristic_fs10_epoch_30_object.ckpt
SH=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/state_head_epoch30/state_head.pt
CACHE=/home/vsr/vsr-tmp/lewm-datasets
OUTBASE=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/rollout_viz_prims
GPU=${1:-1}

for ds in free-fall ground-stationary ground-liftoff ground-side-thrust ground-thrust-sweep impulse-main impulse-side random; do
    echo "=== $ds ==="
    CUDA_VISIBLE_DEVICES=$GPU python lewm/scripts/viz_rollout.py \
        --model $MODEL \
        --state-head $SH \
        --dataset lunarlander_synthetic_${ds} \
        --cache-dir $CACHE \
        --output-dir $OUTBASE/$ds \
        --n-episodes 5 \
        --seq-len 50 \
        --frameskip 10 \
        --start-mode episode_start \
        --device cuda
    echo ""
done

echo "All done. Videos at $OUTBASE/"
