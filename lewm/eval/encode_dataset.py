#!/usr/bin/env python3
"""Encode frames from HDF5 dataset to z embeddings using frozen LeWorldModel encoder.

Caches (z, state) pairs to npz for fast state head training.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# le-wm imports need path setup
LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
sys.path.insert(0, str(LEWM_DIR))

import stable_worldmodel as swm
from utils import get_img_preprocessor


def encode_dataset(
    model_path: str,
    dataset_name: str,
    cache_dir: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 256,
    max_frames: int = 0,
    seed: int = 42,
):
    """Encode frames and save (z, state) pairs.

    Samples random episodes (all frames per episode) up to max_frames total.
    If max_frames=0, encodes everything.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    device = torch.device(device)

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)

    # Open HDF5 directly to get episode structure
    datasets_dir = swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets")
    h5_path = Path(datasets_dir) / f"{dataset_name}.h5"
    with h5py.File(h5_path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
    n_episodes = len(ep_len)
    total_frames = int(ep_len.sum())

    # Select random episodes up to max_frames
    rng = np.random.default_rng(seed)
    ep_order = rng.permutation(n_episodes)

    if max_frames > 0:
        selected_eps = []
        frame_count = 0
        for ep_idx in ep_order:
            selected_eps.append(ep_idx)
            frame_count += ep_len[ep_idx]
            if frame_count >= max_frames:
                break
        selected_eps = sorted(selected_eps)
        n_frames = frame_count
    else:
        selected_eps = list(range(n_episodes))
        n_frames = total_frames

    print(f"Encoding {n_frames} frames from {len(selected_eps)} random episodes "
          f"(of {n_episodes} total, {total_frames} total frames)")

    # Load via HDF5Dataset for proper transforms
    ds = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels", "state"],
        cache_dir=cache_dir,
        transform=transform,
    )

    # Build flat frame indices from selected episodes
    frame_indices = []
    for ep_idx in selected_eps:
        start = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        frame_indices.extend(range(start, start + length))

    n = len(frame_indices)
    print(f"  {n} frames to encode")

    # Pre-allocate arrays to avoid growing lists in RAM
    # Detect z_dim from a single forward pass
    with torch.no_grad():
        sample = ds[frame_indices[0]]
        px = sample["pixels"].squeeze(0).unsqueeze(0).to(device)
        out = model.encoder(px, interpolate_pos_encoding=True)
        z_dim = model.projector(out.last_hidden_state[:, 0]).shape[1]
    state_dim = 15

    z_array = np.empty((n, z_dim), dtype=np.float32)
    state_array = np.empty((n, state_dim), dtype=np.float32)

    with torch.no_grad():
        for batch_start in tqdm(range(0, n, batch_size)):
            batch_end = min(batch_start + batch_size, n)
            pixels_batch = []
            states_batch = []
            for i in range(batch_start, batch_end):
                sample = ds[frame_indices[i]]
                pixels_batch.append(sample["pixels"].squeeze(0))
                states_batch.append(sample["state"].squeeze(0))

            pixels = torch.stack(pixels_batch).to(device)
            states = torch.stack(states_batch)

            output = model.encoder(pixels, interpolate_pos_encoding=True)
            z = model.projector(output.last_hidden_state[:, 0])

            z_array[batch_start:batch_end] = z.cpu().numpy()
            state_array[batch_start:batch_end] = states.numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=z_array, state=state_array)
    print(f"Saved {z_array.shape[0]} embeddings to {output_path}")
    print(f"  z: {z_array.shape}, state: {state_array.shape}")
