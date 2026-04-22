"""Shared helper for reproducing training's action column normalizer.

Training sets up the normalizer in `le-wm/train.py:120` via
`get_column_normalizer(ref_dataset, 'action', 'action')` — a z-score normalizer
computed from the reference dataset's raw per-frame actions (shape (N, action_dim)).
Single-dataset training applies it; multi-dataset training does NOT (the transform
set on swm.data.ConcatDataset is never read — see e5-05).

For offline probes / action-response tests that need to mirror training's
action input distribution, use `compute_action_normalizer` to get (mean, std)
as numpy arrays of shape (action_dim,), then apply per-frame before flattening
to (fs * action_dim,).
"""

from pathlib import Path
import sys

import numpy as np


def compute_action_normalizer(ref_dataset_name: str, cache_dir: str,
                              ctx_len: int = 3, n_preds: int = 1):
    """Return (mean, std) for action column of ref_dataset_name, shape (action_dim,).

    Replicates `le-wm/utils.py:get_column_normalizer` exactly: reads raw
    per-frame action via HDF5Dataset.get_col_data, drops NaN rows, computes
    per-dim mean/std. ctx_len/n_preds only affect the HDF5Dataset's num_steps
    param (which doesn't change the underlying col_data result but is passed
    for consistency with training).
    """
    import torch
    LEWM_DIR = Path(__file__).resolve().parent.parent / "vendor" / "le-wm"
    sys.path.insert(0, str(LEWM_DIR))
    import stable_worldmodel as swm

    ds = swm.data.HDF5Dataset(
        name=ref_dataset_name,
        num_steps=ctx_len + n_preds,
        frameskip=10,
        keys_to_load=["action", "state"],
        keys_to_cache=["action", "state"],
        cache_dir=cache_dir,
        transform=None,
    )
    a = np.array(ds.get_col_data("action"))
    t = torch.from_numpy(a)
    t = t[~torch.isnan(t).any(dim=1)]
    mean = t.mean(0).numpy().astype(np.float32)
    std = t.std(0).numpy().astype(np.float32)
    return mean, std
