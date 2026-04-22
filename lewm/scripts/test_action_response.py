#!/usr/bin/env python3
"""Action-response direction test for LeWM JEPA predictor.

Tests whether the predictor understands that:
- Main thrust → vy increases (upward acceleration)
- Side thrust → angular_vel / angle changes (sign convention: action[1]=+1
  fires the right engine which produces NEGATIVE Δang_vel by Box2D physics;
  action[1]=-1 produces POSITIVE Δang_vel — verified empirically across all
  training datasets on 2026-04-19)
- No action → vy decreases (gravity pulls down)

Does NOT use CEM or planning. Just:
1. Pick frames from dataset
2. Encode → z history (3 frames)
3. Predict z_{t+1} with different actions
4. Decode via state head → kinematics
5. Compare: does the predicted kinematic change in the right (physics-correct) direction?

Usage:
    CUDA_VISIBLE_DEVICES=0 python lewm/scripts/test_action_response.py \
        --model /path/to/lewm_object.ckpt \
        --state-head /path/to/state_head.pt \
        --dataset lunarlander_synthetic_heuristic \
        --cache-dir ~/vsr-tmp/lewm-datasets \
        --output-dir /path/to/run-dir/action_response_heuristic/ \
        --n-frames 200
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch


class Tee(io.TextIOBase):
    """Write to multiple streams simultaneously. Used when --output-dir is set
    to mirror stdout into an action_response_report.txt file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

# Repo root (for lewm.*), lewm/ (for eval.*), vendor/le-wm/ (for pickle: jepa, module)
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
LEWM_ROOT = str(Path(__file__).resolve().parent.parent)
VENDOR_ROOT = str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, LEWM_ROOT)
sys.path.insert(0, VENDOR_ROOT)

from lewm.eval.state_head import load_state_head


KIN_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def encode_frames(model, pixels_uint8, device):
    """Encode (N, H, W, 3) uint8 frames → (N, D) z embeddings."""
    pix = torch.from_numpy(pixels_uint8).float().permute(0, 3, 1, 2) / 255.0
    pix = (pix - MEAN) / STD
    pix = pix.to(device)
    with torch.no_grad():
        out = model.encoder(pix, interpolate_pos_encoding=True)
        z = model.projector(out.last_hidden_state[:, 0])
    return z


def predict_one_step(model, z_history, action_2d, frameskip, device,
                     act_mean=None, act_std=None):
    """Given z history (N, HS, D) and a 2D action, predict z_{t+1}.

    Action is z-score normalized per raw-action dim (if act_mean/act_std
    provided, mirroring training for norm-trained checkpoints), then repeated
    `frameskip` times to match training action_dim = frameskip * 2 = 20.
    """
    N, HS, D = z_history.shape
    act_vec = np.asarray(action_2d, dtype=np.float32)
    if act_mean is not None:
        act_vec = (act_vec - act_mean) / act_std
    act_single = torch.from_numpy(act_vec).to(device)
    act_repeated = act_single.repeat(frameskip)  # (fs * action_dim,)
    act = act_repeated.unsqueeze(0).unsqueeze(0).expand(N, HS, -1)  # (N, HS, 20)

    with torch.no_grad():
        act_emb = model.action_encoder(act)
        pred = model.predict(z_history, act_emb)  # (N, HS, D)
        z_next = pred[:, -1]  # (N, D) — prediction at last history position
    return z_next


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--state-head", required=True)
    p.add_argument("--dataset", default="lunarlander_synthetic_heuristic")
    p.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    p.add_argument("--n-frames", type=int, default=200,
                   help="Number of test transitions")
    p.add_argument("--frameskip", type=int, default=10)
    p.add_argument("--ctx-len", type=int, default=3,
                   help="Training history_size / context length (must match training config).")
    p.add_argument("--n-preds", type=int, default=1,
                   help="Training num_preds / prediction horizon (must match training config). "
                        "The predictor's pred[:, -1] is n_preds steps ahead of the last history "
                        "position, so GT is state[t + n_preds]. Default 1.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None,
                   help="If set, writes action_response_report.txt here "
                        "in addition to stdout.")
    p.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="z-score normalize actions before action_encoder. Default ON. "
                        "Pass --no-normalize-actions for broken-regime multi-dataset "
                        "checkpoints (pre ConcatDataset transform-propagation fix; see "
                        "e5-05). Report filename includes _normZ vs _normRaw so runs "
                        "don't overwrite.")
    p.add_argument("--action-norm-ref", default="lunarlander_synthetic_heuristic_clean",
                   help="Reference dataset for reproducing training's action normalizer. "
                        "Only consulted when --normalize-actions is set.")
    args = p.parse_args()

    # Resolve action normalizer up-front so the report can state it.
    act_mean = act_std = None
    norm_tag = "Raw"
    if args.normalize_actions:
        from lewm.eval.action_norm import compute_action_normalizer
        act_mean, act_std = compute_action_normalizer(args.action_norm_ref, args.cache_dir)
        norm_tag = "Z"

    # Tee stdout to a report file if requested. Report filename encodes
    # normalization mode so norm-ON and norm-OFF runs in the same dir don't
    # overwrite each other.
    _report_file = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"action_response_report_norm{norm_tag}.txt"
        _report_file = open(report_path, "w")
        sys.stdout = Tee(sys.__stdout__, _report_file)
        print(f"Writing report to {report_path}")
        print(f"Model: {args.model}")
        print(f"State head: {args.state_head}")
        print(f"Dataset: {args.dataset}")
        print(f"N frames: {args.n_frames}, frameskip: {args.frameskip}, ctx_len: {args.ctx_len}, n_preds: {args.n_preds}")
        print(f"Action normalization: {'ON (ref=' + args.action_norm_ref + ')' if args.normalize_actions else 'OFF (raw actions)'}")
        if args.normalize_actions:
            print(f"  mean={act_mean.tolist()}  std={act_std.tolist()}")
        print()

    device = torch.device(args.device)

    # Load model + state head. The state head may have been trained with --z-dims
    # (e.g. 0:16 for v2's dedicated z_kin block). load_state_head surfaces the
    # z_slice in metrics; we must slice z before feeding to the head.
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    state_head, sh_metrics = load_state_head(args.state_head, device=args.device)
    state_head.eval()
    z_slice = sh_metrics.get("z_slice")  # (start, end) or None
    if z_slice is not None:
        print(f"State head expects sliced z [{z_slice[0]}:{z_slice[1]}] — slicing predictions accordingly")

    def decode(z):
        """Apply state head to (B, D) predicted z, honoring z_slice if set."""
        z_in = z if z_slice is None else z[..., z_slice[0]:z_slice[1]]
        return state_head(z_in).cpu().numpy()

    # Load dataset frames (need 3 consecutive frames at frameskip for history)
    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    datasets_dir = swm.data.utils.get_cache_dir(args.cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{args.dataset}.h5"

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        pixels_all = f["pixels"]
        state_all = f["state"]

        HS = args.ctx_len  # history size (matches training)
        fs = args.frameskip
        NP = args.n_preds  # prediction horizon; predictor's last output is NP steps ahead
        # Need HS history frames + NP lookahead frames.
        min_len = (HS + NP) * fs

        # Sample random transitions from episodes long enough.
        # Read contiguous episode slices (not fancy indexing) for speed.
        rng = np.random.default_rng(42)
        valid_eps = np.where(ep_len >= min_len)[0]
        rng.shuffle(valid_eps)
        n = args.n_frames

        history_pixels = []  # (N, HS, H, W, 3)
        gt_states_t = []     # (N, 6) state at t (last history frame)
        gt_states_tp1 = []   # (N, 6) state at t + NP (what predictor targets)

        for ep in valid_eps:
            if len(history_pixels) >= n:
                break
            base = int(ep_offset[ep])
            raw_len = int(ep_len[ep])
            n_output = raw_len // fs
            if n_output < HS + NP:
                continue

            # Contiguous slice: all output-step frames for this episode
            ep_pixels = pixels_all[base:base + n_output * fs:fs]  # (n_output, H, W, 3)
            ep_states = state_all[base:base + n_output * fs:fs]   # (n_output, state_dim)

            # Pick random transitions: need t_start+HS-1 (last history) and
            # t_start+HS-1+NP (target state) to exist.
            n_avail = n_output - (HS - 1) - NP
            n_pick = min(n_avail, n - len(history_pixels))
            t_starts = rng.choice(n_avail, size=n_pick, replace=False)

            for t_start in t_starts:
                history_pixels.append(ep_pixels[t_start:t_start + HS])
                gt_states_t.append(ep_states[t_start + HS - 1, :6])
                gt_states_tp1.append(ep_states[t_start + HS - 1 + NP, :6])

    history_pixels = np.array(history_pixels)  # (N, HS, H, W, 3)
    gt_states_t = np.array(gt_states_t)        # (N, 6)
    gt_states_tp1 = np.array(gt_states_tp1)    # (N, 6)
    N = len(history_pixels)
    print(f"Loaded {N} transitions from {args.dataset}")

    # Encode all history frames → z_history (N, HS, D)
    z_list = []
    for h in range(HS):
        z_h = encode_frames(model, history_pixels[:, h], device)
        z_list.append(z_h)
    z_history = torch.stack(z_list, dim=1)  # (N, HS, D)

    # Define test actions
    actions = {
        "no_action":    [0.0, 0.0],
        "main_thrust":  [1.0, 0.0],
        "side_left":    [0.0, -1.0],
        "side_right":   [0.0, +1.0],
    }

    # Predict z_{t+1} for each action, decode via state head
    decoded = {}
    for name, act in actions.items():
        z_next = predict_one_step(model, z_history, act, args.frameskip, device,
                                  act_mean=act_mean, act_std=act_std)
        with torch.no_grad():
            kin = decode(z_next)  # (N, 6); honors z_slice if state head is sliced
        decoded[name] = kin

    # Also decode z_t (last history frame) for reference
    with torch.no_grad():
        z_t = z_history[:, -1]
        kin_t = decode(z_t)

    # ============================================================
    # Direction tests
    # ============================================================
    print("\n" + "=" * 70)
    print("ACTION-RESPONSE DIRECTION TEST")
    print("=" * 70)

    def direction_accuracy(pred_a, pred_b, dim, expected_sign):
        """Fraction of samples where (pred_a - pred_b)[dim] has expected sign."""
        diff = pred_a[:, dim] - pred_b[:, dim]
        if expected_sign > 0:
            return (diff > 0).mean()
        else:
            return (diff < 0).mean()

    def mean_diff(pred_a, pred_b, dim):
        return (pred_a[:, dim] - pred_b[:, dim]).mean()

    # Test 1: Main thrust → vy increases relative to no-action
    acc = direction_accuracy(decoded["main_thrust"], decoded["no_action"], 3, +1)
    md = mean_diff(decoded["main_thrust"], decoded["no_action"], 3)
    print(f"\n1. Main thrust → vy increases (vs no_action)")
    print(f"   Direction accuracy: {acc:.1%}  (mean Δvy: {md:+.4f})")

    # Test 2: No action → vy decreases relative to current (gravity)
    acc = direction_accuracy(decoded["no_action"], kin_t, 3, -1)
    md = mean_diff(decoded["no_action"], kin_t, 3)
    print(f"\n2. No action → vy decreases (gravity)")
    print(f"   Direction accuracy: {acc:.1%}  (mean Δvy: {md:+.4f})")

    # Test 3: Side right thrust (action[1]=+1) → angular_vel DECREASES (vs no_action)
    # Sign convention: action[1]=+1 fires the right engine, which by Box2D
    # physics applies a counter-torque that makes Δang_vel NEGATIVE.
    # Verified empirically on all 7 training datasets with side-thrust frames.
    acc_av = direction_accuracy(decoded["side_right"], decoded["no_action"], 5, -1)
    md_av = mean_diff(decoded["side_right"], decoded["no_action"], 5)
    acc_a = direction_accuracy(decoded["side_right"], decoded["no_action"], 4, -1)
    md_a = mean_diff(decoded["side_right"], decoded["no_action"], 4)
    print(f"\n3. Side right thrust (action[1]=+1) → angular_vel DECREASES (vs no_action)")
    print(f"   ang_vel direction accuracy: {acc_av:.1%}  (mean Δang_vel: {md_av:+.4f})")
    print(f"   angle direction accuracy:   {acc_a:.1%}  (mean Δangle: {md_a:+.4f})")

    # Test 4: Side left thrust (action[1]=-1) → angular_vel INCREASES (opposite of right)
    acc_av = direction_accuracy(decoded["side_left"], decoded["no_action"], 5, +1)
    md_av = mean_diff(decoded["side_left"], decoded["no_action"], 5)
    print(f"\n4. Side left thrust (action[1]=-1) → angular_vel INCREASES (vs no_action)")
    print(f"   ang_vel direction accuracy: {acc_av:.1%}  (mean Δang_vel: {md_av:+.4f})")

    # Test 5: Symmetry — left vs right produce opposite angle changes
    diff_right = decoded["side_right"][:, 5] - decoded["no_action"][:, 5]
    diff_left = decoded["side_left"][:, 5] - decoded["no_action"][:, 5]
    opposite = ((diff_right > 0) & (diff_left < 0)) | ((diff_right < 0) & (diff_left > 0))
    print(f"\n5. Symmetry: left and right produce opposite ang_vel changes")
    print(f"   Symmetry accuracy: {opposite.mean():.1%}")

    # Test 6: Main thrust → x doesn't change much (vs side thrust)
    md_main_x = mean_diff(decoded["main_thrust"], decoded["no_action"], 0)
    md_right_x = mean_diff(decoded["side_right"], decoded["no_action"], 0)
    md_left_x = mean_diff(decoded["side_left"], decoded["no_action"], 0)
    print(f"\n6. Lateral specificity: side thrust moves x more than main")
    print(f"   main→Δx:  {md_main_x:+.4f}")
    print(f"   right→Δx: {md_right_x:+.4f}")
    print(f"   left→Δx:  {md_left_x:+.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("FULL DECODED STATE COMPARISON (mean over N frames)")
    print("=" * 70)
    print(f"{'':20s} {'x':>8s} {'y':>8s} {'vx':>8s} {'vy':>8s} {'angle':>8s} {'ang_vel':>8s}")
    print(f"{'current (z_t)':20s}", end="")
    for d in range(6):
        print(f" {kin_t[:, d].mean():+.4f}", end="")
    print()
    for name in actions:
        print(f"{name:20s}", end="")
        for d in range(6):
            print(f" {decoded[name][:, d].mean():+.4f}", end="")
        print()
    print(f"\n{'Δ main-none':20s}", end="")
    for d in range(6):
        print(f" {mean_diff(decoded['main_thrust'], decoded['no_action'], d):+.4f}", end="")
    print()
    print(f"{'Δ right-none':20s}", end="")
    for d in range(6):
        print(f" {mean_diff(decoded['side_right'], decoded['no_action'], d):+.4f}", end="")
    print()
    print(f"{'Δ left-none':20s}", end="")
    for d in range(6):
        print(f" {mean_diff(decoded['side_left'], decoded['no_action'], d):+.4f}", end="")
    print()

    # GT reference: what actually happens in the dataset
    gt_delta = gt_states_tp1 - gt_states_t
    print(f"\n{'GT Δ (dataset)':20s}", end="")
    for d in range(6):
        print(f" {gt_delta[:, d].mean():+.4f}", end="")
    print()
    print(f"\n(GT is average delta across all sampled transitions, not action-specific)")

    # ============================================================
    # SUSTAINED + DELAYED ACTION TEST
    # ============================================================
    print("\n\n" + "=" * 70)
    print("SUSTAINED & DELAYED ACTION TEST (5-step rollout)")
    print("=" * 70)
    print("\nRolling out 5 steps with constant action. If the predictor has a")
    print("delayed response, the effect should appear at steps 2-5 even if")
    print("step 1 shows nothing.\n")

    N_ROLLOUT = 5

    def rollout_n_steps(model, z_hist, action_2d, n_steps, frameskip, device, state_head):
        """Autoregressive rollout: predict n_steps ahead with constant action."""
        z_h = z_hist.clone()  # (N, HS, D)
        decoded_per_step = []
        for _ in range(n_steps):
            z_next = predict_one_step(model, z_h, action_2d, frameskip, device,
                                      act_mean=act_mean, act_std=act_std)
            with torch.no_grad():
                kin = decode(z_next)
            decoded_per_step.append(kin)
            # Shift history: drop oldest, append new
            z_h = torch.cat([z_h[:, 1:, :], z_next.unsqueeze(1)], dim=1)
        return decoded_per_step  # list of (N, 6), one per step

    rollouts = {}
    for name, act in actions.items():
        rollouts[name] = rollout_n_steps(
            model, z_history, act, N_ROLLOUT, args.frameskip, device, state_head
        )

    # Compare main_thrust vs no_action at each step
    print("Main thrust vs no_action — vy direction accuracy per step:")
    print(f"  {'step':>5s}  {'accuracy':>8s}  {'mean Δvy':>10s}")
    for step in range(N_ROLLOUT):
        acc = direction_accuracy(rollouts["main_thrust"][step],
                                rollouts["no_action"][step], 3, +1)
        md = mean_diff(rollouts["main_thrust"][step],
                      rollouts["no_action"][step], 3)
        print(f"  {step+1:>5d}  {acc:>8.1%}  {md:>+10.4f}")

    # Compare side_right vs no_action at each step (expected: Δang_vel NEGATIVE)
    print("\nSide right vs no_action — ang_vel direction accuracy per step (expect Δang_vel<0):")
    print(f"  {'step':>5s}  {'accuracy':>8s}  {'mean Δang_vel':>14s}")
    for step in range(N_ROLLOUT):
        acc = direction_accuracy(rollouts["side_right"][step],
                                rollouts["no_action"][step], 5, -1)
        md = mean_diff(rollouts["side_right"][step],
                      rollouts["no_action"][step], 5)
        print(f"  {step+1:>5d}  {acc:>8.1%}  {md:>+14.4f}")

    # Delayed action test: thrust at step 0 only, then no-action for steps 1-4
    print("\n\nDELAYED ACTION TEST: thrust at step 0 only, then no-action")
    print("If predictor has a 1-step delay, effect appears at step 2+\n")

    def rollout_impulse(model, z_hist, impulse_action, n_steps, frameskip, device, state_head):
        """Step 0: impulse_action. Steps 1+: no_action."""
        z_h = z_hist.clone()
        no_act = [0.0, 0.0]
        decoded_per_step = []
        for step in range(n_steps):
            act = impulse_action if step == 0 else no_act
            z_next = predict_one_step(model, z_h, act, frameskip, device,
                                      act_mean=act_mean, act_std=act_std)
            with torch.no_grad():
                kin = decode(z_next)
            decoded_per_step.append(kin)
            z_h = torch.cat([z_h[:, 1:, :], z_next.unsqueeze(1)], dim=1)
        return decoded_per_step

    impulse_main = rollout_impulse(
        model, z_history, [1.0, 0.0], N_ROLLOUT, args.frameskip, device, state_head
    )
    baseline_none = rollout_n_steps(
        model, z_history, [0.0, 0.0], N_ROLLOUT, args.frameskip, device, state_head
    )

    print("Impulse main thrust (step 0 only) vs sustained no_action — vy per step:")
    print(f"  {'step':>5s}  {'accuracy':>8s}  {'mean Δvy':>10s}  {'note':>20s}")
    for step in range(N_ROLLOUT):
        acc = direction_accuracy(impulse_main[step], baseline_none[step], 3, +1)
        md = mean_diff(impulse_main[step], baseline_none[step], 3)
        note = "← impulse step" if step == 0 else "← coasting (no action)"
        print(f"  {step+1:>5d}  {acc:>8.1%}  {md:>+10.4f}  {note:>20s}")

    # Close the report file if we opened one.
    if _report_file is not None:
        sys.stdout = sys.__stdout__
        _report_file.close()


if __name__ == "__main__":
    main()
