#!/usr/bin/env python3
"""Diagnose where the action signal dies in the LeWM JEPA predictor.

Part 1: Action Encoder Analysis
  - Embedding geometry: cosine similarity matrix across action types
  - Magnitude scaling: does thrust level affect embedding magnitude?
  - Direction structure: PCA of action embeddings

Part 2: adaLN Modulation Analysis
  - Gate magnitudes per block: how much does each block's gate open?
  - Gate sensitivity: how much do gates change between actions?
  - Shift/scale analysis: what modulation does the action produce?

Part 3: Transformer Attention Analysis
  - Attention patterns: does the predictor attend to history tokens differently
    based on action? (requires hooking into attention layers)

Part 4: Residual Stream Analysis
  - Track the z-vector through each predictor block
  - Measure per-block change in z for different actions
  - Identify which block(s) absorb vs propagate the action signal

Usage:
    CUDA_VISIBLE_DEVICES=0 python lewm/scripts/diagnose_action_signal.py \
        --model /path/to/lewm_object.ckpt \
        --state-head /path/to/state_head.pt \
        --dataset lunarlander_synthetic_heuristic \
        --cache-dir ~/vsr-tmp/lewm-datasets \
        --n-frames 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
LEWM_ROOT = str(Path(__file__).resolve().parent.parent)
VENDOR_ROOT = str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, LEWM_ROOT)
sys.path.insert(0, VENDOR_ROOT)

from lewm.eval.state_head import load_state_head

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def encode_frames(model, pixels_uint8, device):
    pix = torch.from_numpy(pixels_uint8).float().permute(0, 3, 1, 2) / 255.0
    pix = (pix - MEAN) / STD
    pix = pix.to(device)
    with torch.no_grad():
        out = model.encoder(pix, interpolate_pos_encoding=True)
        z = model.projector(out.last_hidden_state[:, 0])
    return z


def make_action_embedding(model, action_2d, frameskip, device):
    """Get action embedding for a single 2D action (repeated across frameskip)."""
    act = torch.tensor(action_2d * frameskip, dtype=torch.float32, device=device)
    act = act.unsqueeze(0).unsqueeze(0)  # (1, 1, 20)
    with torch.no_grad():
        emb = model.action_encoder(act)
    return emb.squeeze()  # (192,)


def load_test_frames(args):
    """Load N frames with 3-frame history from dataset."""
    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    datasets_dir = swm.data.utils.get_cache_dir(args.cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{args.dataset}.h5"

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        pixels_all = f["pixels"]

        HS = 3
        fs = args.frameskip
        min_len = (HS + 1) * fs
        rng = np.random.default_rng(42)
        valid_eps = np.where(ep_len >= min_len)[0]

        history_pixels = []
        for _ in range(args.n_frames):
            ep = rng.choice(valid_eps)
            base = int(ep_offset[ep])
            max_start = int(ep_len[ep]) // fs - (HS + 1)
            if max_start <= 0:
                continue
            t_start = rng.integers(0, max_start)
            indices = [base + (t_start + h) * fs for h in range(HS)]
            frames = pixels_all[indices]
            history_pixels.append(frames)

    return np.array(history_pixels)  # (N, HS, H, W, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--state-head", required=True)
    p.add_argument("--dataset", default="lunarlander_synthetic_heuristic")
    p.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument("--frameskip", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    fs = args.frameskip

    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    state_head, _ = load_state_head(args.state_head, device=args.device)
    state_head.eval()

    # ================================================================
    # PART 1: ACTION ENCODER GEOMETRY
    # ================================================================
    print("=" * 70)
    print("PART 1: ACTION ENCODER EMBEDDING GEOMETRY")
    print("=" * 70)

    # Define a grid of actions to test
    action_grid = {
        "none":            [0.0, 0.0],
        "main_low":        [0.3, 0.0],
        "main_mid":        [0.6, 0.0],
        "main_high":       [1.0, 0.0],
        "left_low":        [0.0, -0.3],
        "left_mid":        [0.0, -0.6],
        "left_high":       [0.0, -1.0],
        "right_low":       [0.0, +0.3],
        "right_mid":       [0.0, +0.6],
        "right_high":      [0.0, +1.0],
        "main+left":       [1.0, -1.0],
        "main+right":      [1.0, +1.0],
        "main_low+left":   [0.3, -0.5],
        "main_low+right":  [0.3, +0.5],
    }

    embeddings = {}
    for name, act in action_grid.items():
        embeddings[name] = make_action_embedding(model, act, fs, device)

    names = list(embeddings.keys())
    emb_stack = torch.stack([embeddings[n] for n in names])  # (N_actions, 192)

    # Cosine similarity matrix
    print("\nCosine similarity matrix:")
    cosine_matrix = torch.cosine_similarity(
        emb_stack.unsqueeze(0), emb_stack.unsqueeze(1), dim=-1
    ).cpu().numpy()

    # Print header
    short_names = [n[:8] for n in names]
    print(f"{'':>14s}", end="")
    for sn in short_names:
        print(f" {sn:>8s}", end="")
    print()
    for i, name in enumerate(names):
        print(f"{name:>14s}", end="")
        for j in range(len(names)):
            print(f" {cosine_matrix[i, j]:>8.3f}", end="")
        print()

    # Norms
    print("\nEmbedding norms:")
    for name in names:
        print(f"  {name:>14s}: {embeddings[name].norm():.3f}")

    # Key comparisons
    print("\nKey comparisons (L2 distance):")
    pairs = [
        ("none", "main_high", "no action vs full main thrust"),
        ("none", "left_high", "no action vs full left"),
        ("none", "right_high", "no action vs full right"),
        ("left_high", "right_high", "full left vs full right (should be far)"),
        ("main_low", "main_high", "low vs high main (should scale)"),
        ("left_low", "left_high", "low vs high left (should scale)"),
        ("main+left", "main+right", "combined thrust: left vs right"),
        ("left_high", "left_low", "left high vs low (magnitude sensitivity)"),
    ]
    for a, b, desc in pairs:
        dist = (embeddings[a] - embeddings[b]).norm().item()
        cos = torch.cosine_similarity(
            embeddings[a].unsqueeze(0), embeddings[b].unsqueeze(0)
        ).item()
        print(f"  {a:>14s} vs {b:<14s}: L2={dist:.3f}, cos={cos:.3f}  ({desc})")

    # PCA of action embeddings
    print("\nPCA of action embeddings (first 3 components):")
    emb_np = emb_stack.cpu().numpy()
    emb_centered = emb_np - emb_np.mean(axis=0)
    U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
    pca = emb_centered @ Vt[:3].T
    explained = (S[:3] ** 2) / (S ** 2).sum()
    print(f"  Explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, PC3={explained[2]:.1%}")
    for i, name in enumerate(names):
        print(f"  {name:>14s}: ({pca[i, 0]:+.3f}, {pca[i, 1]:+.3f}, {pca[i, 2]:+.3f})")

    # ================================================================
    # PART 2: adaLN MODULATION PER BLOCK
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: adaLN MODULATION — GATE/SHIFT/SCALE PER BLOCK")
    print("=" * 70)

    test_actions = {
        "none": embeddings["none"],
        "main_high": embeddings["main_high"],
        "left_high": embeddings["left_high"],
        "right_high": embeddings["right_high"],
    }

    print("\nAbsolute gate magnitudes (how open is the gate?):")
    print(f"{'block':>6s}", end="")
    for name in test_actions:
        print(f"  {name:>12s}_msa  {name:>12s}_mlp", end="")
    print()

    for bi, block in enumerate(model.predictor.transformer.layers):
        print(f"{bi:>6d}", end="")
        for name, emb in test_actions.items():
            with torch.no_grad():
                mod = block.adaLN_modulation(emb.unsqueeze(0))
                chunks = mod.chunk(6, dim=-1)
                gate_msa = chunks[2].abs().mean().item()
                gate_mlp = chunks[5].abs().mean().item()
            print(f"  {gate_msa:>16.4f}  {gate_mlp:>16.4f}", end="")
        print()

    print("\nGate DIFFERENCE from no_action (action sensitivity per block):")
    print(f"{'block':>6s}  {'main_msa':>10s}  {'main_mlp':>10s}  {'left_msa':>10s}  {'left_mlp':>10s}  {'right_msa':>10s}  {'right_mlp':>10s}")
    for bi, block in enumerate(model.predictor.transformer.layers):
        with torch.no_grad():
            mod_none = block.adaLN_modulation(embeddings["none"].unsqueeze(0))
            diffs = {}
            for name in ["main_high", "left_high", "right_high"]:
                mod_act = block.adaLN_modulation(embeddings[name].unsqueeze(0))
                # gate is chunks 2 (msa) and 5 (mlp)
                diff_msa = (mod_act.chunk(6, dim=-1)[2] - mod_none.chunk(6, dim=-1)[2]).abs().mean().item()
                diff_mlp = (mod_act.chunk(6, dim=-1)[5] - mod_none.chunk(6, dim=-1)[5]).abs().mean().item()
                diffs[name] = (diff_msa, diff_mlp)
        print(f"{bi:>6d}  {diffs['main_high'][0]:>10.4f}  {diffs['main_high'][1]:>10.4f}  "
              f"{diffs['left_high'][0]:>10.4f}  {diffs['left_high'][1]:>10.4f}  "
              f"{diffs['right_high'][0]:>10.4f}  {diffs['right_high'][1]:>10.4f}")

    # ================================================================
    # PART 3: RESIDUAL STREAM — WHERE DOES ACTION SIGNAL GET ABSORBED?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: RESIDUAL STREAM — PER-BLOCK Z CHANGE BY ACTION")
    print("=" * 70)

    # Load some real frames
    print("\nLoading test frames...")
    history_pixels = load_test_frames(args)
    N = len(history_pixels)
    print(f"Loaded {N} frames")

    # Encode history
    z_list = []
    for h in range(3):
        z_h = encode_frames(model, history_pixels[:, h], device)
        z_list.append(z_h)
    z_history = torch.stack(z_list, dim=1)  # (N, 3, 192)

    # For a subset of frames, track the residual stream through each block
    N_track = min(50, N)
    z_sub = z_history[:N_track]

    actions_to_track = {
        "none": [0.0, 0.0],
        "main_high": [1.0, 0.0],
        "right_high": [0.0, 1.0],
    }

    print("\nPer-block z-change (L2 norm of block output difference: action vs no_action)")
    print("This shows WHERE the action signal enters and WHERE it gets absorbed.\n")

    # Hook into each ConditionalBlock to capture per-block outputs
    for act_name, act_2d in actions_to_track.items():
        if act_name == "none":
            continue

        act_emb_act = make_action_embedding(model, act_2d, fs, device).unsqueeze(0).unsqueeze(0).expand(N_track, 3, -1)
        act_emb_none = make_action_embedding(model, [0.0, 0.0], fs, device).unsqueeze(0).unsqueeze(0).expand(N_track, 3, -1)

        # Manual forward through predictor blocks
        with torch.no_grad():
            # Initial: z + pos_embedding
            T = z_sub.size(1)
            x_act = z_sub + model.predictor.pos_embedding[:, :T]
            x_act = model.predictor.dropout(x_act)
            x_none = x_act.clone()

            c_act = act_emb_act
            c_none = act_emb_none

            print(f"  {act_name} vs none:")
            print(f"  {'block':>6s}  {'z_diff (L2)':>12s}  {'z_diff/z_norm':>14s}  {'gate_contribution':>18s}")

            for bi, block in enumerate(model.predictor.transformer.layers):
                x_act_out = block(x_act, c_act)
                x_none_out = block(x_none, c_none)

                # Diff at the last history position (what becomes the prediction)
                diff = (x_act_out[:, -1] - x_none_out[:, -1]).norm(dim=-1).mean().item()
                z_norm = x_none_out[:, -1].norm(dim=-1).mean().item()
                ratio = diff / (z_norm + 1e-8)

                # What was the gate contribution?
                # gate_msa * attn_diff + gate_mlp * mlp_diff
                mod_act = block.adaLN_modulation(c_act[:, -1:])
                mod_none = block.adaLN_modulation(c_none[:, -1:])
                gate_diff = (mod_act - mod_none).abs().mean().item()

                print(f"  {bi:>6d}  {diff:>12.4f}  {ratio:>14.4f}  {gate_diff:>18.4f}")

                x_act = x_act_out
                x_none = x_none_out

            # Final output difference
            final_diff = (x_act[:, -1] - x_none[:, -1]).norm(dim=-1).mean().item()
            final_z_norm = x_none[:, -1].norm(dim=-1).mean().item()
            print(f"  {'FINAL':>6s}  {final_diff:>12.4f}  {final_diff/final_z_norm:>14.4f}")

            # Decode final and compare kinematics
            kin_act = state_head(x_act[:, -1]).cpu().numpy()
            kin_none = state_head(x_none[:, -1]).cpu().numpy()
            kin_diff = (kin_act - kin_none).mean(axis=0)
            print(f"\n  Decoded kinematic Δ ({act_name} - none):")
            for d, name in enumerate(["x", "y", "vx", "vy", "angle", "ang_vel"]):
                print(f"    {name:>8s}: {kin_diff[d]:+.4f}")
        print()

    # ================================================================
    # PART 4: ATTENTION PATTERN ANALYSIS
    # ================================================================
    print("=" * 70)
    print("PART 4: ATTENTION PATTERNS — DO THEY CHANGE WITH ACTION?")
    print("=" * 70)

    # Hook into attention layers to capture attention weights
    attn_weights = {}

    def make_attn_hook(block_idx, action_name):
        def hook(module, input, output):
            # Attention module's forward: x → q, k, v → attn_weights → output
            # We need to capture the attention weights. The Attention class
            # uses F.scaled_dot_product_attention which doesn't expose weights.
            # Instead, compute them manually from q, k.
            pass
        return hook

    # Since the Attention class uses F.scaled_dot_product_attention (no weight output),
    # we'll measure attention sensitivity indirectly:
    # Compare attention OUTPUT (not weights) for different actions.
    print("\nAttention output sensitivity per block (L2 norm of attn output diff):")
    print("If attention output changes with action → attention IS action-sensitive.\n")

    for act_name, act_2d in [("main_high", [1.0, 0.0]), ("right_high", [0.0, 1.0])]:
        act_emb_act = make_action_embedding(model, act_2d, fs, device).unsqueeze(0).unsqueeze(0).expand(N_track, 3, -1)
        act_emb_none = make_action_embedding(model, [0.0, 0.0], fs, device).unsqueeze(0).unsqueeze(0).expand(N_track, 3, -1)

        print(f"  {act_name} vs none:")
        print(f"  {'block':>6s}  {'attn_diff':>10s}  {'mlp_diff':>10s}  {'total_diff':>11s}")

        with torch.no_grad():
            T = z_sub.size(1)
            x_act = z_sub + model.predictor.pos_embedding[:, :T]
            x_none = x_act.clone()
            c_act = act_emb_act
            c_none = act_emb_none

            for bi, block in enumerate(model.predictor.transformer.layers):
                # Decompose block: x_out = x + gate_msa * attn(...) + gate_mlp * mlp(...)
                # Compute attn part and mlp part separately for each action

                mod_act = block.adaLN_modulation(c_act)
                sh_msa_a, sc_msa_a, g_msa_a, sh_mlp_a, sc_mlp_a, g_mlp_a = mod_act.chunk(6, dim=-1)

                mod_none = block.adaLN_modulation(c_none)
                sh_msa_n, sc_msa_n, g_msa_n, sh_mlp_n, sc_mlp_n, g_mlp_n = mod_none.chunk(6, dim=-1)

                # Attention path
                from stable_worldmodel.wm.lewm.module import modulate
                attn_in_a = modulate(block.norm1(x_act), sh_msa_a, sc_msa_a)
                attn_in_n = modulate(block.norm1(x_none), sh_msa_n, sc_msa_n)
                attn_out_a = block.attn(attn_in_a)
                attn_out_n = block.attn(attn_in_n)

                attn_diff = (g_msa_a * attn_out_a - g_msa_n * attn_out_n)[:, -1].norm(dim=-1).mean().item()

                # MLP path
                x_mid_a = x_act + g_msa_a * attn_out_a
                x_mid_n = x_none + g_msa_n * attn_out_n
                mlp_in_a = modulate(block.norm2(x_mid_a), sh_mlp_a, sc_mlp_a)
                mlp_in_n = modulate(block.norm2(x_mid_n), sh_mlp_n, sc_mlp_n)
                mlp_out_a = block.mlp(mlp_in_a)
                mlp_out_n = block.mlp(mlp_in_n)

                mlp_diff = (g_mlp_a * mlp_out_a - g_mlp_n * mlp_out_n)[:, -1].norm(dim=-1).mean().item()

                total = (attn_diff ** 2 + mlp_diff ** 2) ** 0.5

                print(f"  {bi:>6d}  {attn_diff:>10.4f}  {mlp_diff:>10.4f}  {total:>11.4f}")

                # Update x for next block
                x_act = x_mid_a + g_mlp_a * mlp_out_a
                x_none = x_mid_n + g_mlp_n * mlp_out_n

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
If action encoder produces distinct embeddings          → signal enters
If adaLN gates differ between actions                   → signal reaches predictor
If per-block z-diff is small AND doesn't grow            → signal absorbed by residual
If per-block z-diff grows through blocks                 → signal propagates
If attn_diff >> mlp_diff                                → action modulates attention
If mlp_diff >> attn_diff                                → action modulates MLP path
If final z-diff is small despite per-block diffs         → cancellation across blocks
If final decoded kinematic diff is small                 → action signal doesn't affect output
""")


if __name__ == "__main__":
    main()
