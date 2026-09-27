"""Phase 5 change sets (CONTRACT 13.1): delete masks and update sets, cut to the first n rows.

bench reads these files; it never reads ground_truth_del*.npy or ground_truth_upd*.npy.
"""

from pathlib import Path

import numpy as np

DELETES = ("del10", "del30", "del50")
UPDATES = ("upd10",)
COMPACT_MODES = ("rebuild", "repair")


def delete_mask(data_dir, name: str, n: int) -> np.ndarray:
    """bool (n,): row i is deleted. With --limit, the mask is cut to the first n rows."""
    if name not in DELETES:
        raise ValueError(f"unknown delete set {name!r}; known: {list(DELETES)}")
    m = np.load(Path(data_dir) / f"delete_{name}.npy")
    if m.dtype != np.bool_ or m.ndim != 1 or len(m) < n:
        raise ValueError(f"delete_{name}.npy: want bool (>= {n},), got {m.dtype} {m.shape}")
    return np.ascontiguousarray(m[:n])


def update_set(data_dir, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(ids int64 (K,), vectors float32 (K, d)). With --limit, only IDs < n are kept."""
    if name not in UPDATES:
        raise ValueError(f"unknown update set {name!r}; known: {list(UPDATES)}")
    ids = np.load(Path(data_dir) / f"update_{name}_ids.npy")
    vecs = np.load(Path(data_dir) / f"update_{name}_vectors.npy")
    if ids.dtype != np.int64 or vecs.dtype != np.float32 or len(ids) != len(vecs):
        raise ValueError(f"update_{name}: want int64 ids and float32 vectors of the same length")
    keep = ids < n
    return np.ascontiguousarray(ids[keep]), np.ascontiguousarray(vecs[keep])


def tombstone_bytes(n: int) -> int:
    """CONTRACT 13.3: the tombstone bit set counts N/8 bytes (Python holds it as a bool array,
    one byte per row, so that NumPy can index with it; the reported size is the bit set's)."""
    return (n + 7) // 8
