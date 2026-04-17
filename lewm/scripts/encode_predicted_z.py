#!/usr/bin/env python3
"""Generate predicted z embeddings (from predictor, not encoder) for state head training.

For each transition in the dataset: encodes 3-frame history, runs the predictor
with the ACTUAL recorded action, produces z_{t+1}_pred. Pairs with GT state_{t+1}.

Output npz has same format as encode_dataset.py (z, state) so it can be
fed directly to train_state_head.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python lewm/scripts/encode_predicted_z.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic \
        --cache-dir ~/vsr-tmp/lewm-datasets \
        --output /path/to/predicted_z.npz \
        --max-frames 50000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
LEWM_ROOT = str(Path(__file__).resolve().parent.parent)
VENDOR_ROOT = str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, LEWM_ROOT)
sys.path.insert(0, VENDOR_ROOT)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="lunarlander_synthetic_heuristic")
    p.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    p.add_argument("--output", required=True, help="Output npz path")
    p.add_argument("--max-frames", type=int, default=50000, help="0=all")
    p.add_argument("--frameskip", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    fs = args.frameskip
    HS = 3  # history size

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()

    datasets_dir = swm.data.utils.get_cache_dir(args.cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{args.dataset}.h5"

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        n_episodes = len(ep_len)

        # Build list of valid transitions: need HS frames + 1 next frame at frameskip
        min_raw_len = (HS + 1) * fs
        transitions = []  # (ep_idx, t_output) where t_output is in output-step space
        for ep in range(n_episodes):
            raw_len = int(ep_len[ep])
            n_output = raw_len // fs
            if n_output < HS + 1:
                continue
            base = int(ep_offset[ep])
            for t in range(HS - 1, n_output - 1):
                # history: t-2, t-1, t; next: t+1 (in output steps)
                transitions.append((ep, base, t))

        rng = np.random.default_rng(args.seed)
        rng.shuffle(transitions)
        if args.max_frames > 0:
            transitions = transitions[:args.max_frames]
        n_trans = len(transitions)
        print(f"Processing {n_trans} transitions from {n_episodes} episodes")

        # Determine z_dim from a test forward pass
        test_pix = f["pixels"][0:1]
        test_pix_t = torch.from_numpy(test_pix).float().permute(0, 3, 1, 2) / 255.0
        test_pix_t = (test_pix_t - MEAN) / STD
        with torch.no_grad():
            test_out = model.encoder(test_pix_t.to(device), interpolate_pos_encoding=True)
            test_z = model.projector(test_out.last_hidden_state[:, 0])
        z_dim = test_z.shape[1]
        state_dim = f["state"].shape[1]
        print(f"z_dim={z_dim}, state_dim={state_dim}")

        all_z = np.zeros((n_trans, z_dim), dtype=np.float32)
        all_state = np.zeros((n_trans, state_dim), dtype=np.float32)

        # Process in batches
        for b_start in tqdm(range(0, n_trans, args.batch_size), desc="predicting"):
            b_end = min(b_start + args.batch_size, n_trans)
            batch = transitions[b_start:b_end]
            B = len(batch)

            # Gather pixel indices for history (HS frames) and action + next state
            hist_indices = []  # (B, HS) raw indices into pixels
            action_indices = []  # (B,) raw index for action at t
            next_state_indices = []  # (B,) raw index for state at t+1

            for ep, base, t in batch:
                for h in range(HS):
                    hist_indices.append(base + (t - (HS - 1) + h) * fs)
                # Action: concatenated raw actions between t and t+1
                action_indices.append(base + t * fs)
                # Next state at t+1
                next_state_indices.append(base + (t + 1) * fs)

            hist_indices = np.array(hist_indices).reshape(B, HS)
            action_indices = np.array(action_indices)
            next_state_indices = np.array(next_state_indices)

            # Read pixels for history (B*HS frames)
            flat_hist = hist_indices.flatten()
            sort_order = np.argsort(flat_hist)
            sorted_idx = flat_hist[sort_order]
            pixels_flat = f["pixels"][sorted_idx]
            unsort = np.argsort(sort_order)
            pixels_flat = pixels_flat[unsort]  # back to original order
            pixels_hist = pixels_flat.reshape(B, HS, *pixels_flat.shape[1:])

            # Read actions: for each transition, get fs raw actions and concatenate
            actions_batch = np.zeros((B, fs * 2), dtype=np.float32)
            for i, aidx in enumerate(action_indices):
                end_idx = min(aidx + fs, f["action"].shape[0])
                raw_acts = f["action"][aidx:end_idx]  # (fs, 2) or shorter
                if len(raw_acts) < fs:
                    pad = np.zeros((fs - len(raw_acts), 2), dtype=np.float32)
                    raw_acts = np.concatenate([raw_acts, pad], axis=0)
                actions_batch[i] = raw_acts.flatten()  # (fs*2,)

            # Read next states
            sort_ns = np.argsort(next_state_indices)
            sorted_ns = next_state_indices[sort_ns]
            states_next = f["state"][sorted_ns]
            unsort_ns = np.argsort(sort_ns)
            states_next = states_next[unsort_ns]

            # Encode history frames
            z_hist_list = []
            for h in range(HS):
                pix = torch.from_numpy(pixels_hist[:, h]).float().permute(0, 3, 1, 2) / 255.0
                pix = (pix - MEAN) / STD
                pix = pix.to(device)
                with torch.no_grad():
                    out = model.encoder(pix, interpolate_pos_encoding=True)
                    z = model.projector(out.last_hidden_state[:, 0])
                z_hist_list.append(z)
            z_history = torch.stack(z_hist_list, dim=1)  # (B, HS, D)

            # Build action embedding: (B, HS, fs*2) → action_encoder
            # Each history position gets the same action (the action at time t)
            act_t = torch.from_numpy(actions_batch).float().to(device)
            act_expanded = act_t.unsqueeze(1).expand(B, HS, -1)  # (B, HS, 20)
            with torch.no_grad():
                act_emb = model.action_encoder(act_expanded)
                pred = model.predict(z_history, act_emb)
                z_pred = pred[:, -1]  # (B, D)

            all_z[b_start:b_end] = z_pred.cpu().numpy()
            all_state[b_start:b_end] = states_next

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, z=all_z, state=all_state)
    print(f"Saved {n_trans} predicted z's to {args.output}")
    print(f"  z: {all_z.shape}, state: {all_state.shape}")


if __name__ == "__main__":
    main()
