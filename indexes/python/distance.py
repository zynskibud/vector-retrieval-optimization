"""Dot-product scoring and top-k selection. Higher score is better."""

import numpy as np


def scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Dot product of one query (d,) with every row of matrix (n, d) -> (n,)."""
    return matrix @ query


def scores_batch(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Dot products of queries (q, d) with rows of matrix (n, d) -> (q, n)."""
    return queries @ matrix.T


def top_k(s: np.ndarray, k: int, ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, scores) of the k highest scores, best first, padded with -1 / -inf.

    `ids` maps positions in `s` to row IDs; default is the position itself.
    """
    n = len(s)
    kk = min(k, n)
    if kk == 0:
        pos = np.empty(0, dtype=np.int64)
    elif kk < n:
        # Take everything at or above the k-th best score, so ties at the boundary are
        # broken by ID below and not by argpartition's arbitrary choice.
        threshold = s[np.argpartition(-s, kk - 1)[kk - 1]]
        pos = np.flatnonzero(s >= threshold)
    else:
        pos = np.arange(n)
    row_ids = pos if ids is None else ids[pos]
    pos = pos[np.lexsort((row_ids, -s[pos]))][:kk]  # best score first, lower ID first on ties
    out_ids = np.full(k, -1, dtype=np.int64)
    out_scores = np.full(k, -np.inf, dtype=np.float32)
    out_ids[:kk] = pos if ids is None else ids[pos]
    out_scores[:kk] = s[pos]
    return out_ids, out_scores
