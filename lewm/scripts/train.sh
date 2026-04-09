#!/bin/bash
# Train LeWorldModel on Lunar Lander
# Usage: bash lewm/scripts/train.sh [extra hydra overrides...]
#
# Output goes to $STABLEWM_HOME (checkpoints) and stdout (logs).
# To log to file: bash lewm/scripts/train.sh 2>&1 | tee ~/vsr-tmp/lewm-train.log
set -e

source ~/virtual_envs/lewm/bin/activate
export STABLEWM_HOME=/media/hdd1/physics-priors-latent-space/lunar-lander-data
export CUDA_VISIBLE_DEVICES=0  # 4090

cd ~/Dropbox/code/world-model-audits/lewm/vendor/le-wm
python train.py --config-name=lewm_lunarlander "$@"
