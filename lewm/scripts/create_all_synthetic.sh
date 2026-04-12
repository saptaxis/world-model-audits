#!/bin/bash
# Create synthetic (triangle) HDF5 datasets for all episode categories.
# CPU-only — runs in parallel, no GPU needed.
set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

DATA=/media/hdd1/physics-priors-latent-space/lunar-lander-data/world_model_data/visual-gym-default-random-heuristic-prims
OUT=/home/vsr/vsr-tmp/lewm-datasets/datasets
LOGDIR=/home/vsr/vsr-tmp/lewm-synthetic-create-logs
mkdir -p "$LOGDIR"

for dir in random free-fall ground-stationary ground-liftoff ground-side-thrust ground-thrust-sweep impulse-main impulse-side; do
    echo "Starting synthetic $dir..."
    python lewm/scripts/create_synthetic_dataset.py \
        --input-dirs "$DATA/$dir" \
        --output "$OUT/lunarlander_synthetic_${dir}.h5" \
        --triangle-radius 35 \
        > "$LOGDIR/synthetic-${dir}.log" 2>&1 &
done

echo "All 8 launched. Monitor with:"
echo "  tail -f $LOGDIR/synthetic-*.log"
echo "  grep -l 'Written' $LOGDIR/synthetic-*.log  # completed ones"
echo ""
echo "Waiting for all to finish..."
wait
echo "All done."
