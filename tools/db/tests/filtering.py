"""Shared Phase 3 filter check for the database tests (CONTRACT section 11.5).

The truth is ground_truth_<name>.npy restricted to the loaded rows: the exact top-k among
the first N rows that pass filter_<name>.npy, computed here by brute force.
"""

from pathlib import Path

import numpy as np

K = 10


def filtered_truth(data: Path, vectors: np.ndarray, queries: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(mask over the loaded rows, exact top-K IDs among passing rows)."""
    mask = np.load(data / f"filter_{name}.npy")[: len(vectors)]
    passing = np.flatnonzero(mask)
    order = np.argsort(-(queries @ vectors[passing].T), axis=1, kind="stable")[:, :K]
    return mask, passing[order]


def check(client, data: Path, vectors, queries, params: dict) -> float:
    """Recall@10 of a filtered search; asserts that every returned ID passes the filter."""
    client.data_dir = data
    mask, truth = filtered_truth(data, vectors, queries, params["filter"])
    hits = 0
    for q, t in zip(queries, truth):
        ids, _ = client.search(q, K, params)
        ids = [i for i in ids if i >= 0]
        bad = [i for i in ids if not mask[i]]
        assert not bad, f"IDs that fail filter {params['filter']}: {bad[:5]}"
        hits += len(set(ids[:K]) & set(t.tolist()))
    r = hits / (len(queries) * K)
    print(f"{client.name} {params} recall@10={r:.4f}")
    return r
