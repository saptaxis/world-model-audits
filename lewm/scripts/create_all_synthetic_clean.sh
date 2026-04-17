#!/bin/bash
# Create 12 TRUNCATED synthetic (triangle) HDF5 datasets for the clean-data run.
# Truncates each episode at first frame with y>1.5 or |x|>1.0. Drops episodes
# with kept length < 15. Writes canonical outputs to /media/hdd1 and a merged
# truncation_stats_clean.json next to them.
#
# `random` is deliberately excluded from this run.

set -e

source ~/virtual_envs/lewm/bin/activate
cd ~/Dropbox/code/world-model-audits

DATA_ROOT=/media/hdd1/physics-priors-latent-space/lunar-lander-data/world_model_data
PRIMS_ROOT="$DATA_ROOT/visual-gym-default-random-heuristic-prims"
AGENT_ROOT="$DATA_ROOT/gym-default"
OUT=/media/hdd1/physics-priors-latent-space/lunar-lander-data/datasets
LOGDIR=~/vsr-tmp/lewm-synthetic-clean-logs
STATS_DIR="$LOGDIR/stats"

mkdir -p "$OUT" "$LOGDIR" "$STATS_DIR"

# Each entry: "<label>|<input_dir>"
# Label becomes the HDF5 suffix after `lunarlander_synthetic_`.
DATASETS=(
  "heuristic|$PRIMS_ROOT/heuristic"
  "free-fall|$PRIMS_ROOT/free-fall"
  "ground-stationary|$PRIMS_ROOT/ground-stationary"
  "ground-liftoff|$PRIMS_ROOT/ground-liftoff"
  "ground-side-thrust|$PRIMS_ROOT/ground-side-thrust"
  "ground-thrust-sweep|$PRIMS_ROOT/ground-thrust-sweep"
  "impulse-main|$PRIMS_ROOT/impulse-main"
  "impulse-side|$PRIMS_ROOT/impulse-side"
  "gym-default-blind-s42|$AGENT_ROOT/gym-default-blind-s42"
  "gym-default-blind-s123|$AGENT_ROOT/gym-default-blind-s123"
  "gym-default-labeled-s42|$AGENT_ROOT/gym-default-labeled-s42"
  "gym-default-labeled-s123|$AGENT_ROOT/gym-default-labeled-s123"
)

launch_one() {
    local label="$1"
    local indir="$2"
    local outpath="$OUT/lunarlander_synthetic_${label}_clean.h5"
    local statspath="$STATS_DIR/${label}.json"
    local logpath="$LOGDIR/${label}.log"
    echo "  launching $label"
    python lewm/scripts/create_synthetic_dataset.py \
        --input-dirs "$indir" \
        --output "$outpath" \
        --triangle-radius 35 \
        --truncate \
        --min-length 15 \
        --stats-out "$statspath" \
        --stats-label "$label" \
        > "$logpath" 2>&1 &
}

# Two waves of 6 each to cap concurrent CPU load.
WAVE_SIZE=6
echo "Running ${#DATASETS[@]} regenerations in waves of $WAVE_SIZE. Logs: $LOGDIR/"

echo "=== Wave 1 ==="
for entry in "${DATASETS[@]:0:WAVE_SIZE}"; do
    launch_one "${entry%%|*}" "${entry##*|}"
done
wait
echo "Wave 1 done."

echo "=== Wave 2 ==="
for entry in "${DATASETS[@]:WAVE_SIZE}"; do
    launch_one "${entry%%|*}" "${entry##*|}"
done
wait
echo "Wave 2 done."

# Merge per-dataset stats files into a single truncation_stats_clean.json.
MERGED="$OUT/truncation_stats_clean.json"
python - <<PY
import json, pathlib
stats = {}
for p in sorted(pathlib.Path("$STATS_DIR").glob("*.json")):
    with p.open() as f:
        entry = json.load(f)
    # Each per-run file is {"<label>": {...}}; merge all.
    stats.update(entry)
out = pathlib.Path("$MERGED")
out.write_text(json.dumps(stats, indent=2, sort_keys=True))
print(f"Merged {len(stats)} entries -> {out}")
# Print a quick summary line per dataset.
for label, e in sorted(stats.items()):
    pct_kept = 100*e["frames_kept"]/max(e["frames_in"], 1)
    print(f"  {label:32s} eps {e['episodes_kept']:5d}/{e['episodes_in']:5d}  frames {e['frames_kept']:8d}/{e['frames_in']:8d} ({pct_kept:5.1f}% kept)")
PY

echo ""
echo "Done. Canonical outputs:"
echo "  $OUT/lunarlander_synthetic_*_clean.h5"
echo "  $MERGED"
echo ""
echo "Next: copy to SSD training cache:"
echo "  rsync -ahv --progress $OUT/lunarlander_synthetic_*_clean.h5 ~/vsr-tmp/lewm-datasets/datasets/"
