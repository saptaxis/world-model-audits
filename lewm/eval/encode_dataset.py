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
):
    """Encode frames and save (z, state) pairs."""
    device = torch.device(device)

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    ds = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels", "state"],
        cache_dir=cache_dir,
        transform=transform,
    )

    n = len(ds) if max_frames <= 0 else min(len(ds), max_frames)
    print(f"Encoding {n} frames from {dataset_name}...")

    all_z = []
    all_states = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            pixels_batch = []
            states_batch = []
            for i in range(start, end):
                sample = ds[i]
                pixels_batch.append(sample["pixels"].squeeze(0))
                states_batch.append(sample["state"].squeeze(0))

            pixels = torch.stack(pixels_batch).to(device)
            states = torch.stack(states_batch)

            output = model.encoder(pixels, interpolate_pos_encoding=True)
            z = model.projector(output.last_hidden_state[:, 0])

            all_z.append(z.cpu().numpy())
            all_states.append(states.numpy())

    z_array = np.concatenate(all_z, axis=0)
    state_array = np.concatenate(all_states, axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=z_array, state=state_array)
    print(f"Saved {z_array.shape[0]} embeddings to {output_path}")
    print(f"  z: {z_array.shape}, state: {state_array.shape}")
