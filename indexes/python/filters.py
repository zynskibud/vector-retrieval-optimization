"""Metadata filter masks for Phase 3 (CONTRACT 11).

bench sets DATA_DIR on the index module; the module calls mask() on first use of a filter.
A mask is bool (N,) from <data>/filter_<name>.npy, cut to the N rows of the index (--limit).
"""

from pathlib import Path

import numpy as np

NAMES = ("none", "top50", "top10", "top1", "top01")
_cache: dict = {}


def mask(data_dir, name: str, n: int) -> np.ndarray | None:
    """Return the bool mask of rows that pass filter `name`, or None for "none"."""
    name = str(name)
    if name not in NAMES:
        raise ValueError(f"unknown filter {name!r}; known: {list(NAMES)}")
    if name == "none":
        return None
    if data_dir is None:
        raise ValueError("filter needs the data directory (module DATA_DIR is not set)")
    key = (str(data_dir), name, n)
    if key not in _cache:
        m = np.load(Path(data_dir) / f"filter_{name}.npy")
        if m.dtype != np.bool_ or m.ndim != 1 or len(m) < n:
            raise ValueError(f"filter_{name}.npy: want bool (>= {n},), got {m.dtype} {m.shape}")
        _cache[key] = np.ascontiguousarray(m[:n])
    return _cache[key]


_counts: dict = {}


def count(m: np.ndarray) -> int:
    """Number of passing rows in mask m (cached by object id)."""
    key = id(m)
    if key not in _counts:
        _counts[key] = int(m.sum())
    return _counts[key]
