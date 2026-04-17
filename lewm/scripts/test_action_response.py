#!/usr/bin/env python3
"""Action-response direction test for LeWM JEPA predictor.

Tests whether the predictor understands that:
- Main thrust → vy increases (upward acceleration)
- Side thrust → angle/angular_vel changes
- No action → vy decreases (gravity pulls down)

Does NOT use CEM or planning. Just:
1. Pick frames from dataset
2. Encode → z history (3 frames)
3. Predict z_{t+1} with different actions
4. Decode via state head → kinematics
5. Compare: does the predicted kinematic change in the right direction?

Usage:
    CUDA_VISIBLE_DEVICES=0 python lewm/scripts/test_action_response.py \
        --model /path/to/lewm_object.ckpt \
        --state-head /path/to/state_head.pt \
        --dataset lunarlander_synthetic_heuristic \
        --cache-dir ~/vsr-tmp/lewm-datasets \
        --n-frames 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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


def predict_one_step(model, z_history, action_2d, frameskip, device):
    """Given z history (N, HS, D) and a 2D action, predict z_{t+1}.

    Action is repeated `frameskip` times and concatenated to match training
    action_dim = frameskip * 2 = 20.
    """
    N, HS, D = z_history.shape
    # Build action: repeat the 2D action `frameskip` times → (N, HS, frameskip*2)
    act_single = torch.tensor(action_2d, dtype=torch.float32, device=device)
    act_repeated = act_single.repeat(frameskip)  # (20,)
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
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load model + state head
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    state_head, _ = load_state_head(args.state_head, device=args.device)
    state_head.eval()

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

        HS = 3  # history size
        fs = args.frameskip
        min_len = (HS + 1) * fs  # need HS frames + 1 next frame

        # Sample random transitions from episodes long enough
        rng = np.random.default_rng(42)
        valid_eps = np.where(ep_len >= min_len)[0]
        n = min(args.n_frames, len(valid_eps) * 5)  # cap

        # For each sample: pick episode, pick start offset, read HS+1 frames
        history_pixels = []  # (N, HS, H, W, 3)
        gt_states_t = []     # (N, 6) state at t (last history frame)
        gt_states_tp1 = []   # (N, 6) state at t+1

        for _ in range(n):
            ep = rng.choice(valid_eps)
            base = int(ep_offset[ep])
            max_start = int(ep_len[ep]) // fs - (HS + 1)
            if max_start <= 0:
                continue
            t_start = rng.integers(0, max_start)

            # Read HS+1 frames at frameskip
            indices = [base + (t_start + h) * fs for h in range(HS + 1)]
            frames = pixels_all[indices]  # (HS+1, H, W, 3)
            states = state_all[indices]   # (HS+1, state_dim)

            history_pixels.append(frames[:HS])
            gt_states_t.append(states[HS - 1, :6])
            gt_states_tp1.append(states[HS, :6])

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
        z_next = predict_one_step(model, z_history, act, args.frameskip, device)
        with torch.no_grad():
            kin = state_head(z_next).cpu().numpy()  # (N, 6)
        decoded[name] = kin

    # Also decode z_t (last history frame) for reference
    with torch.no_grad():
        z_t = z_history[:, -1]
        kin_t = state_head(z_t).cpu().numpy()

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

    # Test 3: Side right thrust → angle changes (angular_vel increases)
    acc_av = direction_accuracy(decoded["side_right"], decoded["no_action"], 5, +1)
    md_av = mean_diff(decoded["side_right"], decoded["no_action"], 5)
    acc_a = direction_accuracy(decoded["side_right"], decoded["no_action"], 4, +1)
    md_a = mean_diff(decoded["side_right"], decoded["no_action"], 4)
    print(f"\n3. Side right thrust → angular_vel / angle change (vs no_action)")
    print(f"   ang_vel direction accuracy: {acc_av:.1%}  (mean Δang_vel: {md_av:+.4f})")
    print(f"   angle direction accuracy:   {acc_a:.1%}  (mean Δangle: {md_a:+.4f})")

    # Test 4: Side left thrust → opposite of side right
    acc_av = direction_accuracy(decoded["side_left"], decoded["no_action"], 5, -1)
    md_av = mean_diff(decoded["side_left"], decoded["no_action"], 5)
    print(f"\n4. Side left thrust → angular_vel decreases (vs no_action)")
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
        HS = z_h.shape[1]
        decoded_per_step = []
        for step in range(n_steps):
            z_next = predict_one_step(model, z_h, action_2d, frameskip, device)
            with torch.no_grad():
                kin = state_head(z_next).cpu().numpy()
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

    # Compare side_right vs no_action at each step
    print("\nSide right vs no_action — ang_vel direction accuracy per step:")
    print(f"  {'step':>5s}  {'accuracy':>8s}  {'mean Δang_vel':>14s}")
    for step in range(N_ROLLOUT):
        acc = direction_accuracy(rollouts["side_right"][step],
                                rollouts["no_action"][step], 5, +1)
        md = mean_diff(rollouts["side_right"][step],
                      rollouts["no_action"][step], 5)
        print(f"  {step+1:>5d}  {acc:>8.1%}  {md:>+14.4f}")

    # Delayed action test: thrust at step 0 only, then no-action for steps 1-4
    print("\n\nDELAYED ACTION TEST: thrust at step 0 only, then no-action")
    print("If predictor has a 1-step delay, effect appears at step 2+\n")

    def rollout_impulse(model, z_hist, impulse_action, n_steps, frameskip, device, state_head):
        """Step 0: impulse_action. Steps 1+: no_action."""
        z_h = z_hist.clone()
        HS = z_h.shape[1]
        no_act = [0.0, 0.0]
        decoded_per_step = []
        for step in range(n_steps):
            act = impulse_action if step == 0 else no_act
            z_next = predict_one_step(model, z_h, act, frameskip, device)
            with torch.no_grad():
                kin = state_head(z_next).cpu().numpy()
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


if __name__ == "__main__":
    main()
