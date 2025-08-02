"""data_cache.py
Utility functions to cache LR patches derived from HR files so that the HR→LR degradation
is computed only once per sample.

Stored as .npy files in a sibling directory `<data_dir>/lr_cache/` to keep things simple.
The cache filename scheme is `<hr_stem>_lr.npy`.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple


def get_cache_path(hr_file: Path) -> Path:
    """Return path where LR cache for given HR file should be stored."""
    # Place all LR caches in a single 'lr_cache' folder adjacent to patch directory.
    # If hr_file is already inside an lr_cache directory, avoid nesting.
    if hr_file.parent.name == "lr_cache":
        cache_dir = hr_file.parent  # reuse existing
    else:
        cache_dir = hr_file.parent / "lr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hr_file.stem}_lr.npy"
    return cache_path


def load_or_compute_lr(
    hr_file: Path,
    compute_fn,
    *fn_args,
    **fn_kwargs,
) -> np.ndarray:
    """Load LR patch from cache or compute and save it.

    Args:
        hr_file: path to original HR complex patch (.npy)
        compute_fn: function producing LR ndarray when cache miss
        *fn_args, **fn_kwargs: forwarded to compute_fn
    Returns:
        lr ndarray (np.complex64) of shape (2, h, w)
    """
    cache_path = get_cache_path(hr_file)
    if cache_path.exists():
        try:
            lr = np.load(cache_path, mmap_mode="r")  # zero-copy mmap
            return lr
        except Exception:
            # Corrupted cache: fall back to recompute
            cache_path.unlink(missing_ok=True)

    # Cache miss -> compute
    lr = compute_fn(*fn_args, **fn_kwargs)
    # Ensure contiguous array before saving to avoid pickle overhead
    np.save(cache_path, np.ascontiguousarray(lr))
    return lr
