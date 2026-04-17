#!/usr/bin/env python3
"""Analyze action-kinematic dynamics in rollout episodes.

Per-episode and pooled analysis:
- Pearson correlation: action vs kinematic delta (pred + GT)
- Cross-correlation with lag: peak lag and correlation
- Pred vs GT trajectory correlation per kinematic dim
- Summary report

Usage:
    CUDA_VISIBLE_DEVICES=0 python lewm/scripts/analyze_rollout_dynamics.py \
        --model /path/to/lewm_object.ckpt \
        --state-head /path/to/state_head.pt \
        --dataset lunarlander_synthetic_heuristic \
        --cache-dir ~/vsr-tmp/lewm-datasets \
        --n-episodes 20 --seq-len 15 \
        --output-dir /path/to/report/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
LEWM_ROOT = str(Path(__file__).resolve().parent.parent)
VENDOR_ROOT = str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, LEWM_ROOT)
sys.path.insert(0, VENDOR_ROOT)

from scipy.stats import pearsonr
from scipy.signal import correlate


KIN_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def cross_corr_peak(sig_a, sig_b, max_lag=5):
    """Normalized cross-correlation. Returns (lags, xcorr, peak_lag, peak_r)."""
    a = (sig_a - sig_a.mean()) / (sig_a.std() + 1e-8)
    b = (sig_b - sig_b.mean()) / (sig_b.std() + 1e-8)
    T = len(a)
    xcorr = correlate(a, b, mode='full') / T
    lags = np.arange(-(T - 1), T)
    # Restrict to [-max_lag, +max_lag]
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lags_r = lags[mask]
    xcorr_r = xcorr[mask]
    peak_idx = np.argmax(np.abs(xcorr_r))
    return lags_r, xcorr_r, int(lags_r[peak_idx]), float(xcorr_r[peak_idx])


def analyze_episode(pred_states, actual_states, actions, ep_idx):
    """Analyze one episode. Returns dict of metrics."""
    T = min(len(pred_states) - 1, len(actions))
    if T < 5:
        return None

    main_a = np.array([float(actions[t][0]) for t in range(T)])
    side_a = np.array([float(actions[t][1]) for t in range(T)])

    pred_deltas = np.diff(pred_states[:T + 1], axis=0)  # (T, 6)
    gt_deltas = np.diff(actual_states[:T + 1], axis=0)

    result = {"ep_idx": ep_idx, "n_steps": T}

    # 1. Pearson correlation: action vs kinematic delta
    for act_name, act_sig, kin_idx, kin_name in [
        ("main", main_a, 3, "dvy"),
        ("side", side_a, 4, "dangle"),
        ("side", side_a, 5, "dang_vel"),
    ]:
        if act_sig.std() < 1e-6:
            result[f"pearson_{act_name}_{kin_name}_pred"] = float("nan")
            result[f"pearson_{act_name}_{kin_name}_gt"] = float("nan")
            continue
        r_pred, p_pred = pearsonr(act_sig, pred_deltas[:, kin_idx])
        r_gt, p_gt = pearsonr(act_sig, gt_deltas[:, kin_idx])
        result[f"pearson_{act_name}_{kin_name}_pred"] = float(r_pred)
        result[f"pearson_{act_name}_{kin_name}_gt"] = float(r_gt)
        result[f"pearson_{act_name}_{kin_name}_pred_p"] = float(p_pred)

    # 2. Cross-correlation with lag
    for act_name, act_sig, kin_idx, kin_name in [
        ("main", main_a, 3, "dvy"),
        ("side", side_a, 4, "dangle"),
    ]:
        if act_sig.std() < 1e-6 or pred_deltas[:, kin_idx].std() < 1e-6:
            result[f"xcorr_{act_name}_{kin_name}_pred_peak_lag"] = float("nan")
            result[f"xcorr_{act_name}_{kin_name}_pred_peak_r"] = float("nan")
            continue
        _, _, lag_p, r_p = cross_corr_peak(act_sig, pred_deltas[:, kin_idx])
        _, _, lag_g, r_g = cross_corr_peak(act_sig, gt_deltas[:, kin_idx])
        result[f"xcorr_{act_name}_{kin_name}_pred_peak_lag"] = lag_p
        result[f"xcorr_{act_name}_{kin_name}_pred_peak_r"] = float(r_p)
        result[f"xcorr_{act_name}_{kin_name}_gt_peak_lag"] = lag_g
        result[f"xcorr_{act_name}_{kin_name}_gt_peak_r"] = float(r_g)

    # 3. Pred vs GT trajectory correlation
    for dim, name in enumerate(KIN_NAMES):
        r, p = pearsonr(pred_states[:T, dim], actual_states[:T, dim])
        result[f"traj_corr_{name}"] = float(r)

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--state-head", required=True)
    p.add_argument("--dataset", default="lunarlander_synthetic_heuristic")
    p.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=15)
    p.add_argument("--frameskip", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from lewm.eval.rollout import rollout_episodes

    print("Running rollouts...")
    results = rollout_episodes(
        args.model, args.state_head,
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        n_episodes=args.n_episodes,
        seq_len=args.seq_len,
        frameskip=args.frameskip,
        start_mode="episode_start",
        device=args.device, seed=42)

    print(f"Analyzing {len(results)} episodes...")

    # Per-episode analysis
    ep_reports = []
    for i, r in enumerate(results):
        report = analyze_episode(
            r["predicted_states"], r["actual_states"], r["actions"], i
        )
        if report is not None:
            ep_reports.append(report)

    # Pooled analysis
    all_main, all_side = [], []
    all_pred_dvy, all_gt_dvy = [], []
    all_pred_da, all_gt_da = [], []
    all_pred_dav, all_gt_dav = [], []

    for r in results:
        pred = r["predicted_states"]
        actual = r["actual_states"]
        actions = r["actions"]
        T = min(len(pred) - 1, len(actions))
        if T < 5:
            continue
        for t in range(T):
            all_main.append(float(actions[t][0]))
            all_side.append(float(actions[t][1]))
            all_pred_dvy.append(pred[t + 1, 3] - pred[t, 3])
            all_gt_dvy.append(actual[t + 1, 3] - actual[t, 3])
            all_pred_da.append(pred[t + 1, 4] - pred[t, 4])
            all_gt_da.append(actual[t + 1, 4] - actual[t, 4])
            all_pred_dav.append(pred[t + 1, 5] - pred[t, 5])
            all_gt_dav.append(actual[t + 1, 5] - actual[t, 5])

    all_main = np.array(all_main)
    all_side = np.array(all_side)
    all_pred_dvy = np.array(all_pred_dvy)
    all_gt_dvy = np.array(all_gt_dvy)
    all_pred_da = np.array(all_pred_da)
    all_gt_da = np.array(all_gt_da)

    pooled = {}
    for act_name, act, pred_d, gt_d, kin_name in [
        ("main", all_main, all_pred_dvy, all_gt_dvy, "dvy"),
        ("side", all_side, all_pred_da, all_gt_da, "dangle"),
        ("side", all_side, np.array(all_pred_dav), np.array(all_gt_dav), "dang_vel"),
    ]:
        r_p, p_p = pearsonr(act, pred_d)
        r_g, p_g = pearsonr(act, gt_d)
        pooled[f"{act_name}_vs_{kin_name}_pred"] = f"r={r_p:+.3f} (p={p_p:.4f})"
        pooled[f"{act_name}_vs_{kin_name}_gt"] = f"r={r_g:+.3f} (p={p_g:.4f})"

    # Lagged correlation pooled
    lagged = {}
    for lag in range(4):
        if lag == 0:
            m, pdv, gdv = all_main, all_pred_dvy, all_gt_dvy
        else:
            m = all_main[:-lag]
            pdv = all_pred_dvy[lag:]
            gdv = all_gt_dvy[lag:]
        r_p, _ = pearsonr(m, pdv)
        r_g, _ = pearsonr(m, gdv)
        lagged[f"lag{lag}"] = f"pred r={r_p:+.3f}, GT r={r_g:+.3f}"

    # Print report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ROLLOUT DYNAMICS ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nEpisodes: {len(ep_reports)}, steps per episode: ~{args.seq_len}")

    # Per-episode table
    print("\n--- Per-episode: main thrust vs Δvy ---")
    print(f"{'ep':>4s}  {'pred_r':>8s}  {'gt_r':>8s}  {'xcorr_lag':>10s}  {'xcorr_peak':>11s}  {'traj_y':>7s}  {'traj_vy':>8s}")
    for r in ep_reports:
        pr = r.get("pearson_main_dvy_pred", float("nan"))
        gr = r.get("pearson_main_dvy_gt", float("nan"))
        xl = r.get("xcorr_main_dvy_pred_peak_lag", float("nan"))
        xr = r.get("xcorr_main_dvy_pred_peak_r", float("nan"))
        ty = r.get("traj_corr_y", float("nan"))
        tv = r.get("traj_corr_vy", float("nan"))
        print(f"{r['ep_idx']:>4d}  {pr:>+8.3f}  {gr:>+8.3f}  {xl:>10}  {xr:>+11.3f}  {ty:>+7.3f}  {tv:>+8.3f}")

    print("\n--- Per-episode: side thrust vs Δangle ---")
    print(f"{'ep':>4s}  {'pred_r':>8s}  {'gt_r':>8s}  {'xcorr_lag':>10s}  {'xcorr_peak':>11s}  {'traj_angle':>10s}")
    for r in ep_reports:
        pr = r.get("pearson_side_dangle_pred", float("nan"))
        gr = r.get("pearson_side_dangle_gt", float("nan"))
        xl = r.get("xcorr_side_dangle_pred_peak_lag", float("nan"))
        xr = r.get("xcorr_side_dangle_pred_peak_r", float("nan"))
        ta = r.get("traj_corr_angle", float("nan"))
        print(f"{r['ep_idx']:>4d}  {pr:>+8.3f}  {gr:>+8.3f}  {xl:>10}  {xr:>+11.3f}  {ta:>+10.3f}")

    print("\n--- Pooled correlations ---")
    for k, v in pooled.items():
        print(f"  {k:>30s}: {v}")

    print("\n--- Lagged correlation (main thrust → Δvy) ---")
    for k, v in lagged.items():
        print(f"  {k}: {v}")

    # Pred vs GT trajectory (pooled mean)
    print("\n--- Pred vs GT trajectory correlation (mean across episodes) ---")
    for name in KIN_NAMES:
        vals = [r[f"traj_corr_{name}"] for r in ep_reports if not np.isnan(r.get(f"traj_corr_{name}", float("nan")))]
        print(f"  {name:>8s}: mean r={np.mean(vals):+.3f} ± {np.std(vals):.3f}")

    # Save JSON
    report = {
        "per_episode": ep_reports,
        "pooled": pooled,
        "lagged": lagged,
        "config": vars(args),
    }
    json_path = out_dir / "dynamics_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {json_path}")


if __name__ == "__main__":
    main()
