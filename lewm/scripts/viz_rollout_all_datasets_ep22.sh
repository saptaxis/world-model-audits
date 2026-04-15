#!/bin/bash
# Rollout viz on ALL 13 datasets used for training synthetic-all-fs10 (epoch 22).
# Includes: heuristic + 8 primitives + 4 gym-default agent policies.
# 20 episodes per dataset, seq_len sized to episode length at frameskip=10.
set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

RUN_DIR=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-all-fs10
MODEL=$RUN_DIR/lewm_lunarlander_synthetic_all_fs10_epoch_22_object.ckpt
SH=$RUN_DIR/state_head_epoch22_all/state_head.pt
CACHE=/home/vsr/vsr-tmp/lewm-datasets
OUTBASE=$RUN_DIR/rollout_viz_ep22
GPU=${1:-1}
N_EP=${2:-20}

run_viz() {
    local ds=$1
    local seqlen=$2
    for mode in episode_start episode_mid; do
        echo "=== $ds [$mode] (seq_len=$seqlen, n_episodes=$N_EP) ==="
        CUDA_VISIBLE_DEVICES=$GPU python lewm/scripts/viz_rollout.py \
            --model $MODEL \
            --state-head $SH \
            --dataset lunarlander_synthetic_${ds} \
            --cache-dir $CACHE \
            --output-dir $OUTBASE/$ds/$mode \
            --n-episodes $N_EP \
            --seq-len $seqlen \
            --frameskip 10 \
            --start-mode $mode \
            --device cuda
        echo ""
    done
}

# Long episodes (heuristic + agent policies): seq_len=20
run_viz          heuristic                20
run_viz          gym-default-blind-s42    20
run_viz          gym-default-blind-s123   20
run_viz          gym-default-labeled-s42  20
run_viz          gym-default-labeled-s123 20

# Ground primitives (longer): seq_len=15
run_viz          ground-stationary        15
run_viz          ground-side-thrust       15

# Medium primitives: seq_len=8
run_viz          ground-liftoff           8
run_viz          ground-thrust-sweep      8

# Short primitives: seq_len=4-7
run_viz          random                   7
run_viz          impulse-main             4
run_viz          free-fall                4
run_viz          impulse-side             4

echo "All done. Videos at $OUTBASE/"
