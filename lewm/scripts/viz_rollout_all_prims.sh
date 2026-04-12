#!/bin/bash
# Rollout viz on all primitives datasets using the synthetic-heuristic encoder.
# Tests generalization: encoder/predictor trained on heuristic, evaluated on prims.
# seq_len per dataset chosen based on episode length distribution at frameskip=10.
set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

MODEL=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/lewm_lunarlander_synthetic_heuristic_fs10_epoch_30_object.ckpt
SH=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/state_head_epoch30/state_head.pt
CACHE=/home/vsr/vsr-tmp/lewm-datasets
OUTBASE=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-heuristic-fs10/rollout_viz_prims
GPU=${1:-1}

run_viz() {
    local ds=$1
    local seqlen=$2
    echo "=== $ds (seq_len=$seqlen) ==="
    CUDA_VISIBLE_DEVICES=$GPU python lewm/scripts/viz_rollout.py \
        --model $MODEL \
        --state-head $SH \
        --dataset lunarlander_synthetic_${ds} \
        --cache-dir $CACHE \
        --output-dir $OUTBASE/$ds \
        --n-episodes 5 \
        --seq-len $seqlen \
        --frameskip 10 \
        --start-mode episode_start \
        --device cuda
    echo ""
}

#                dataset                seq_len  (based on p10 episode length / 10)
run_viz          ground-stationary      15
run_viz          ground-side-thrust     15
run_viz          ground-liftoff         8
run_viz          ground-thrust-sweep    8
run_viz          random                 7
run_viz          impulse-main           4
run_viz          free-fall              4
run_viz          impulse-side           4

# Also re-run heuristic for comparison
run_viz          heuristic              15

echo "All done. Videos at $OUTBASE/"
