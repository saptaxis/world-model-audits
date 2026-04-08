#!/bin/bash
# Convert all 9 episode categories to HDF5 in parallel.
# Each process logs to its own file. Monitor with:
#   tail -f ~/vsr-tmp/lewm-convert-Apr082026/*.log
#   # or watch summary:
#   grep -h "Converting\|Written\|Total" ~/vsr-tmp/lewm-convert-Apr082026/*.log
set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

DATA=/media/hdd1/physics-priors-latent-space/lunar-lander-data/world_model_data/visual-gym-default-random-heuristic-prims
OUT=/media/hdd1/physics-priors-latent-space/lunar-lander-data/datasets
LOGDIR=~/vsr-tmp/lewm-convert-Apr082026

mkdir -p "$OUT" "$LOGDIR"

for dir in heuristic random free-fall ground-stationary ground-liftoff ground-side-thrust ground-thrust-sweep impulse-main impulse-side; do
    echo "Starting $dir → $LOGDIR/lewm-convert-${dir}.log"
    python lewm/scripts/convert_npz_to_hdf5.py \
        --input-dirs "$DATA/$dir" \
        --output "$OUT/lunarlander_${dir}.h5" \
        --resize 224 \
        > "$LOGDIR/lewm-convert-${dir}.log" 2>&1 &
done

echo "All 9 launched. Monitor with:"
echo "  tail -f ~/vsr-tmp/lewm-convert-Apr082026/*.log"
echo "  # or check status:"
echo "  grep -l 'Written' ~/vsr-tmp/lewm-convert-Apr082026/*.log  # completed ones"
echo ""
echo "Waiting for all to finish..."
wait
echo "All done. Check logs in $LOGDIR/lewm-convert-*.log"
