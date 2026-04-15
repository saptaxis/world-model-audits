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


def encode_dataset(
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
    """Encode frames and save (z, state) pairs.

    Samples random episodes (all frames per episode) up to max_frames total.
    If max_frames=0, encodes everything.

    Reads from HDF5 in chunks of read_chunk_size frames to limit memory usage.
    Each chunk is read, transformed, encoded, then freed before the next.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    device = torch.device(device)

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # ImageNet normalization constants (same as get_img_preprocessor)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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
        else:
            selected_eps = list(range(n_episodes))

        # Build contiguous (start, length) slices per episode, sorted by offset
        ep_slices = sorted(
            [(int(ep_offset[ep_idx]), int(ep_len[ep_idx])) for ep_idx in selected_eps]
        )
        n = sum(length for _, length in ep_slices)
        print(f"Encoding {n} frames from {len(selected_eps)} episodes "
              f"(of {n_episodes} total, {total_frames} total frames)")

        all_z = []
        all_states = []
        pbar = tqdm(total=n, desc="encoding", unit="frames")

        # Accumulate contiguous episode slices into chunks up to read_chunk_size,
        # then read each chunk as one contiguous HDF5 slice for maximum speed.
        def flush_chunk(start, end):
            """Read a contiguous HDF5 range [start, end) and encode it."""
            pixels_chunk = f["pixels"][start:end]
            states_chunk = f["state"][start:end]
            chunk_n = end - start

            for b_start in range(0, chunk_n, batch_size):
                b_end = min(b_start + batch_size, chunk_n)
                pixels = torch.from_numpy(pixels_chunk[b_start:b_end]).float()
                pixels = pixels.permute(0, 3, 1, 2) / 255.0
                pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
                pixels = pixels.to(device)

                output = model.encoder(pixels, interpolate_pos_encoding=True)
                z = model.projector(output.last_hidden_state[:, 0])

                all_z.append(z.cpu().numpy())
                all_states.append(states_chunk[b_start:b_end])
                pbar.update(b_end - b_start)

            del pixels_chunk, states_chunk

        with torch.no_grad():
            # Merge adjacent episodes into contiguous reads where possible
            chunk_start = ep_slices[0][0]
            chunk_end = ep_slices[0][0] + ep_slices[0][1]

            for ep_start, ep_length in ep_slices[1:]:
                # If this episode is contiguous with current chunk and chunk is small enough
                if ep_start == chunk_end and (chunk_end - chunk_start + ep_length) <= read_chunk_size:
                    chunk_end = ep_start + ep_length
                else:
                    # Flush current chunk
                    flush_chunk(chunk_start, chunk_end)
                    chunk_start = ep_start
                    chunk_end = ep_start + ep_length

            # Flush last chunk
            flush_chunk(chunk_start, chunk_end)

        pbar.close()

    z_array = np.concatenate(all_z, axis=0)
    state_array = np.concatenate(all_states, axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=z_array, state=state_array)
    print(f"Saved {z_array.shape[0]} embeddings to {output_path}")
    print(f"  z: {z_array.shape}, state: {state_array.shape}")
