#!/bin/bash
# Rollout viz on all 12 _clean datasets for synthetic-all-clean-fs10-aux (epoch 8).
# Mirrors viz_rollout_all_datasets_ep22.sh (same seq_len tiers) but uses:
#   - _clean HDF5 suffix
#   - synthetic-all-clean-fs10-aux run dir
#   - state_head_epoch8_all state head
#   - 'random' excluded (dropped from training set per the clean+aux spec)
set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

RUN_DIR=/media/hdd1/physics-priors-latent-space/lunar-lander-networks/lewm-runs/synthetic-all-clean-fs10-aux
MODEL=$RUN_DIR/lewm_lunarlander_synthetic_all_clean_fs10_aux_epoch_8_object.ckpt
SH=$RUN_DIR/state_head_epoch8_all/state_head.pt
CACHE=/home/vsr/vsr-tmp/lewm-datasets
OUTBASE=$RUN_DIR/rollout_viz_ep8
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
            --dataset lunarlander_synthetic_${ds}_clean \
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

# Per-dataset seq_len ≈ mean raw episode length / frameskip, so rollouts
# cover the whole episode without rollout_episodes dropping clips as
# too-short-for-span. Derived from truncation_stats_clean.json mean raw
# lengths at fs=10.
run_viz          heuristic                36   # mean 363
run_viz          gym-default-blind-s42    37   # mean 378
run_viz          gym-default-blind-s123   41   # mean 412
run_viz          gym-default-labeled-s42  15   # mean 151
run_viz          gym-default-labeled-s123 17   # mean 177
run_viz          ground-stationary        20   # mean 201
run_viz          ground-side-thrust       20   # mean 200
run_viz          ground-thrust-sweep      13   # mean 130
run_viz          ground-liftoff            9   # mean  93
run_viz          impulse-main              8   # mean  84
run_viz          free-fall                 4   # mean  47
run_viz          impulse-side              4   # mean  46

echo "All done. Videos at $OUTBASE/"
