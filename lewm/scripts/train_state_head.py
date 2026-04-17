#!/usr/bin/env python3
"""Train state head probe on LeWorldModel latents.

Usage:
    # Single dataset (backward compatible):
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_heuristic \
        --output-dir /path/to/state_head/ \
        --max-frames 50000

    # Multiple datasets, N frames per dataset:
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic lunarlander_synthetic_free-fall ... \
        --output-dir /path/to/state_head/ \
        --max-frames-per-dataset 5000

    # Multiple datasets, N frames total (split evenly):
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic lunarlander_synthetic_free-fall ... \
        --output-dir /path/to/state_head/ \
        --max-frames 50000

    # No limit — all frames from all datasets:
    python lewm/scripts/train_state_head.py \
        --model /path/to/lewm_object.ckpt \
        --dataset lunarlander_synthetic_heuristic lunarlander_synthetic_free-fall ... \
        --output-dir /path/to/state_head/

Encodes frames (or loads cached z's), trains MLP probe, reports R² per dim.
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.encode_dataset import encode_dataset
from eval.state_head import train_state_head, save_state_head, KIN_DIM_NAMES


def encode_predicted(
    model_path: str,
    dataset_name: str,
    cache_dir: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 256,
    max_frames: int = 0,
    seed: int = 42,
    read_chunk_size: int = 10000,
):
    """Generate predicted z's from the predictor (not encoder) for state head training.

    For each transition: encode 3-frame history → run predictor with recorded
    action → produce z_{t+1}_pred. Pair with GT state_{t+1}.

    Output npz has same (z, state) keys as encode_dataset for compatibility.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from tqdm import tqdm

    LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
    sys.path.insert(0, str(LEWM_DIR))
    import stable_worldmodel as swm

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    device_t = torch.device(device)
    model = torch.load(model_path, map_location=device_t, weights_only=False)
    model.eval()

    HS = 3  # history size
    fs = 10  # frameskip (hardcoded to match training)

    datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{dataset_name}.h5"

    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        n_episodes = len(ep_len)

        # Count valid transitions and select episodes
        valid_eps = []
        total_trans = 0
        for ep in range(n_episodes):
            raw_len = int(ep_len[ep])
            n_output = raw_len // fs
            n_trans_ep = max(0, n_output - HS)  # transitions per episode
            if n_trans_ep > 0:
                valid_eps.append((ep, n_trans_ep))
                total_trans += n_trans_ep

        rng = np.random.default_rng(seed)
        rng.shuffle(valid_eps)

        # Select episodes until we have enough transitions
        if max_frames > 0:
            selected_eps = []
            count = 0
            for ep, n_t in valid_eps:
                selected_eps.append((ep, n_t))
                count += n_t
                if count >= max_frames:
                    break
            n_trans = min(count, max_frames)
        else:
            selected_eps = valid_eps
            n_trans = total_trans
        # Sort by episode offset for contiguous HDF5 reads
        selected_eps.sort(key=lambda x: int(ep_offset[x[0]]))
        print(f"  {n_trans} transitions from {len(selected_eps)} episodes")

        state_dim = f["state"].shape[1]
        # Get z_dim from test forward
        test_pix = torch.from_numpy(f["pixels"][0:1]).float().permute(0, 3, 1, 2) / 255.0
        test_pix = (test_pix - MEAN) / STD
        with torch.no_grad():
            test_out = model.encoder(test_pix.to(device_t), interpolate_pos_encoding=True)
            z_dim = model.projector(test_out.last_hidden_state[:, 0]).shape[1]

        all_z_parts = []
        all_state_parts = []

        pbar = tqdm(total=n_trans, desc="predicting z", unit="trans")

        for ep_idx, n_trans_ep in selected_eps:
            base = int(ep_offset[ep_idx])
            raw_len = int(ep_len[ep_idx])
            n_output = raw_len // fs

            # Contiguous read: all output-step frames for this episode
            # Need frames at indices: base, base+fs, base+2*fs, ..., base+(n_output-1)*fs
            frame_indices = np.arange(n_output) * fs + base
            ep_pixels = f["pixels"][frame_indices[0]:frame_indices[-1]+1:fs]  # contiguous slice
            ep_states = f["state"][frame_indices[0]:frame_indices[-1]+1:fs]
            ep_actions = f["action"][base:base+raw_len]  # all raw actions for episode

            # For each valid transition t (where t >= HS-1 and t+1 < n_output):
            # history frames: t-2, t-1, t; next state at t+1
            for t_start in range(0, n_trans_ep, batch_size):
                t_end = min(t_start + batch_size, n_trans_ep)
                B = t_end - t_start
                t_indices = np.arange(t_start, t_end) + (HS - 1)  # offset by history

                # History pixels: (B, HS, H, W, 3)
                pixels_hist = np.stack([
                    ep_pixels[t_indices - (HS - 1) + h] for h in range(HS)
                ], axis=1)

                # Actions: for each transition, concatenate fs raw actions starting at t*fs
                actions_batch = np.zeros((B, fs * 2), dtype=np.float32)
                for i, t in enumerate(t_indices):
                    act_start = t * fs  # relative to episode start
                    act_end = min(act_start + fs, len(ep_actions))
                    raw = ep_actions[act_start:act_end]
                    if len(raw) < fs:
                        raw = np.concatenate([raw, np.zeros((fs - len(raw), 2), dtype=np.float32)])
                    actions_batch[i] = raw.flatten()

                # Next states at t+1
                states_next = ep_states[t_indices + 1]

                # Encode history frames
                z_hist_list = []
                for h in range(HS):
                    pix = torch.from_numpy(pixels_hist[:, h]).float().permute(0, 3, 1, 2) / 255.0
                    pix = (pix - MEAN) / STD
                    with torch.no_grad():
                        out = model.encoder(pix.to(device_t), interpolate_pos_encoding=True)
                        z = model.projector(out.last_hidden_state[:, 0])
                    z_hist_list.append(z)
                z_history = torch.stack(z_hist_list, dim=1)  # (B, HS, D)

                # Run predictor with recorded action
                act_t = torch.from_numpy(actions_batch).float().to(device_t)
                act_expanded = act_t.unsqueeze(1).expand(B, HS, -1)
                with torch.no_grad():
                    act_emb = model.action_encoder(act_expanded)
                    pred = model.predict(z_history, act_emb)
                    z_pred = pred[:, -1]

                all_z_parts.append(z_pred.cpu().numpy())
                all_state_parts.append(states_next)
                pbar.update(B)

        pbar.close()

        all_z = np.concatenate(all_z_parts, axis=0)[:n_trans]
        all_state = np.concatenate(all_state_parts, axis=0)[:n_trans]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=all_z, state=all_state)
    print(f"Saved {n_trans} predicted z's to {output_path}")
    print(f"  z: {all_z.shape}, state: {all_state.shape}")


def resolve_max_frames(datasets, max_frames, max_frames_per_dataset, cache_dir):
    """Return per-dataset max_frames list.

    Scenarios:
        - Both flags set: error
        - --max-frames-per-dataset N: N per dataset
        - --max-frames N (single dataset): N for that dataset (backward compat)
        - --max-frames N (multi-dataset): N total, distributed proportionally
          by dataset size (number of frames in each HDF5)
        - Neither flag: 0 (all frames) per dataset
    """
    n = len(datasets)
    if max_frames is not None and max_frames_per_dataset is not None:
        raise ValueError("Cannot specify both --max-frames and --max-frames-per-dataset")
    if max_frames_per_dataset is not None:
        return [max_frames_per_dataset] * n
    if max_frames is not None:
        if n == 1:
            return [max_frames]
        # Distribute proportionally by dataset size
        import h5py
        import hdf5plugin  # noqa: F401
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "le-wm"))
        import stable_worldmodel as swm
        datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
        sizes = []
        for ds_name in datasets:
            h5_path = Path(datasets_dir) / f"{ds_name}.h5"
            with h5py.File(h5_path, "r") as f:
                sizes.append(int(f["ep_len"][:].sum()))
        total = sum(sizes)
        per_ds = [min(s, int(max_frames * s / total)) for s in sizes]
        print(f"Distributing {max_frames} frames proportionally across {n} datasets:")
        for ds_name, s, alloc in zip(datasets, sizes, per_ds):
            capped = " (all)" if alloc >= s else ""
            print(f"  {ds_name}: {s} total -> {alloc} sampled{capped}")
        return per_ds
    return [0] * n  # 0 = all frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LeWorldModel _object.ckpt path")
    parser.add_argument("--dataset", nargs="+", default=["lunarlander_heuristic"],
                        help="One or more HDF5 dataset names")
    parser.add_argument("--cache-dir", default="/media/hdd1/physics-priors-latent-space/lunar-lander-data")
    parser.add_argument("--output-dir", required=True, help="Where to save state head + results")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Total max frames across all datasets (split evenly for multi-dataset). 0=all.")
    parser.add_argument("--max-frames-per-dataset", type=int, default=None,
                        help="Max frames per dataset (multi-dataset only). 0=all.")
    parser.add_argument("--encode-batch-size", type=int, default=1024, help="Batch size for ViT encoding")
    parser.add_argument("--read-chunk-size", type=int, default=10000,
                        help="Frames to read from HDF5 at a time (controls memory usage)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction held out for validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--linear", action="store_true",
                        help="Use linear probe instead of MLP")
    parser.add_argument("--predicted-z", action="store_true",
                        help="Train on predictor z's (action-conditioned) instead of encoder z's")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    datasets = args.dataset
    per_ds_limits = resolve_max_frames(datasets, args.max_frames, args.max_frames_per_dataset,
                                       args.cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.predicted_z:
        # Predicted-z mode: use predictor(z_history, action) → z_{t+1}
        z_cache = output_dir / "predicted_z.npz"
        if z_cache.exists():
            print(f"Loading cached predicted embeddings from {z_cache}")
            data = np.load(z_cache)
            z_all, state_all = data["z"], data["state"]
        else:
            # For predicted-z, max_frames applies to total transitions
            total_limit = args.max_frames if args.max_frames is not None else 0
            if args.max_frames_per_dataset is not None:
                total_limit = args.max_frames_per_dataset * len(datasets)
            z_parts, state_parts = [], []
            for ds_name in datasets:
                ds_cache = output_dir / f"predicted_z_{ds_name}.npz"
                ds_limit = total_limit // len(datasets) if total_limit > 0 else 0
                print(f"\n--- Predicting z for {ds_name} (max_frames={ds_limit or 'all'}) ---")
                encode_predicted(
                    model_path=args.model,
                    dataset_name=ds_name,
                    cache_dir=args.cache_dir,
                    output_path=str(ds_cache),
                    device=args.device,
                    batch_size=args.encode_batch_size,
                    max_frames=ds_limit,
                )
                data = np.load(ds_cache)
                z_parts.append(data["z"])
                state_parts.append(data["state"])
                print(f"  {ds_name}: {data['z'].shape[0]} transitions predicted")
            z_all = np.concatenate(z_parts, axis=0)
            state_all = np.concatenate(state_parts, axis=0)
            np.savez(z_cache, z=z_all, state=state_all)
            print(f"\nCombined: {z_all.shape[0]} predicted z's -> {z_cache}")
    else:
        # Standard encoder-z mode
        z_cache = output_dir / "encoded_z.npz"
        if z_cache.exists():
            print(f"Loading cached embeddings from {z_cache}")
            data = np.load(z_cache)
            z_all, state_all = data["z"], data["state"]
        else:
            z_parts, state_parts = [], []
            for ds_name, ds_limit in zip(datasets, per_ds_limits):
                ds_cache = output_dir / f"encoded_z_{ds_name}.npz"
                print(f"\n--- Encoding {ds_name} (max_frames={ds_limit or 'all'}) ---")
                encode_dataset(
                    model_path=args.model,
                    dataset_name=ds_name,
                    cache_dir=args.cache_dir,
                    output_path=str(ds_cache),
                    device=args.device,
                    batch_size=args.encode_batch_size,
                    max_frames=ds_limit,
                    read_chunk_size=args.read_chunk_size,
                )
                data = np.load(ds_cache)
                z_parts.append(data["z"])
                state_parts.append(data["state"])
                print(f"  {ds_name}: {data['z'].shape[0]} frames encoded")

            z_all = np.concatenate(z_parts, axis=0)
            state_all = np.concatenate(state_parts, axis=0)
            np.savez(z_cache, z=z_all, state=state_all)
            print(f"\nCombined: {z_all.shape[0]} frames from {len(datasets)} datasets -> {z_cache}")

    # Use first 6 dims only (kinematics)
    state_kin = state_all[:, :6]

    n = len(z_all)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    split = int((1 - args.val_split) * n)
    z_train, z_val = z_all[perm[:split]], z_all[perm[split:]]
    s_train, s_val = state_kin[perm[:split]], state_kin[perm[split:]]
    print(f"Train: {len(z_train)}, Val: {len(z_val)}")

    # Step 2: Train
    probe_type = "linear" if args.linear else "MLP"
    print(f"Training state head ({probe_type})...")
    head, metrics = train_state_head(
        z_train, s_train, z_val, s_val,
        epochs=args.epochs,
        device=args.device,
        linear=args.linear,
    )

    # Step 3: Report
    print(f"\n=== State Head R² ({probe_type}) ===")
    for name in KIN_DIM_NAMES:
        r2 = metrics["r2_per_dim"][name]
        print(f"  {name:8s}: R² = {r2:.4f}")
    print(f"  {'mean':8s}: R² = {metrics['r2_mean']:.4f}")
    print(f"  val MSE: {metrics['val_mse']:.6f}")

    # Step 4: Save
    suffix = "_linear" if args.linear else ""
    save_state_head(head, metrics, str(output_dir / f"state_head{suffix}.pt"))

    # Save report as text
    ds_str = ", ".join(datasets)
    report_path = output_dir / f"r2_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(f"State Head R² Report ({probe_type})\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Datasets ({len(datasets)}): {ds_str}\n")
        f.write(f"Frames: {n}\n\n")
        for name in KIN_DIM_NAMES:
            f.write(f"  {name:8s}: R² = {metrics['r2_per_dim'][name]:.4f}\n")
        f.write(f"\n  mean R²: {metrics['r2_mean']:.4f}\n")
        f.write(f"  val MSE: {metrics['val_mse']:.6f}\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
