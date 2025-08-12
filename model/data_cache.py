"""data_cache.py
Utility functions to cache LR patches derived from HR files so that the HR→LR degradation
is computed only once per sample.

Stored as .npy files in a sibling directory `<data_dir>/lr_cache/` to keep things simple.
The cache filename scheme is `<hr_stem>_lr.npy`. A sidecar JSON `<hr_stem>_lr_meta.json`
stores degradation parameters. Cache is reused only if metadata matches.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import json


def get_cache_path(hr_file: Path) -> Path:
    """Return path where LR cache for given HR file should be stored."""
    # Place all LR caches in a single 'lr_cache' folder adjacent to patch directory.
    # If hr_file is already inside an lr_cache directory, avoid nesting.
    if "lr_cache" in hr_file.parts:
        cache_dir = next(p for p in hr_file.parents if p.name == "lr_cache")
    else:
        cache_dir = hr_file.parent / "lr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hr_file.stem}_lr.npy"
    return cache_path


def get_cache_meta_path(hr_file: Path) -> Path:
    cache_path = get_cache_path(hr_file)
    return cache_path.with_suffix('').with_name(cache_path.stem + "_meta.json")


def load_or_compute_lr(
    hr_file: Path,
    compute_fn,
    *fn_args,
    meta: Dict[str, Any] | None = None,
    **fn_kwargs,
) -> np.ndarray:
    """Load LR patch from cache or compute and save it with metadata.

    Args:
        hr_file: path to original HR complex patch (.npy)
        compute_fn: function producing LR ndarray when cache miss
        *fn_args, **fn_kwargs: forwarded to compute_fn
        meta: optional dict of degradation parameters; if provided, cache is reused
              only when stored metadata equals this dict
    Returns:
        lr ndarray (np.complex64) of shape (2, h, w)
    """
    cache_path = get_cache_path(hr_file)
    meta_path = get_cache_meta_path(hr_file)

    def _meta_matches() -> bool:
        if meta is None:
            return True
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                stored = json.load(f)
            return stored == meta
        except Exception:
            return False

    if cache_path.exists() and _meta_matches():
        try:
            lr = np.load(cache_path, mmap_mode="r")  # zero-copy mmap
            return lr
        except Exception:
            # Corrupted cache: fall back to recompute
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    # Cache miss or metadata mismatch -> compute and overwrite
    lr = compute_fn(*fn_args, **fn_kwargs)
    # Ensure contiguous array before saving to avoid pickle overhead
    np.save(cache_path, np.ascontiguousarray(lr))
    if meta is not None:
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            # Non-fatal metadata write failure
            pass
    return lr
