"""Compute the exact top-k neighbors of each query by brute force.

Every index is scored against this file. Vectors are L2-normalized, so the
dot product equals cosine similarity. Higher is better.

Outputs in the data directory:
  ground_truth.npy         int64 (Q, k). Row IDs into vectors.npy, best first.
  ground_truth_scores.npy  float32 (Q, k). The matching dot products.

Run: uv run python -m tools.data.ground_truth [--data data/processed]
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def exact_topk(vectors, queries, k: int, chunk: int = 100_000, progress: bool = True):
    """Return (ids, scores), each (Q, k), best first, by scanning the corpus in chunks.

    The running top-k per query keeps memory at (Q, chunk) scores instead of (Q, N).
    """
    q = len(queries)
    best_scores = np.full((q, k), -np.inf, dtype=np.float32)
    best_ids = np.zeros((q, k), dtype=np.int64)
    starts = range(0, len(vectors), chunk)
    for start in tqdm(starts, desc="scan", disable=not progress):
        scores = queries @ np.asarray(vectors[start : start + chunk]).T
        ids = np.arange(start, start + scores.shape[1], dtype=np.int64)
        all_scores = np.concatenate([best_scores, scores], axis=1)
        all_ids = np.concatenate([best_ids, np.broadcast_to(ids, scores.shape)], axis=1)
        top = np.argpartition(-all_scores, k, axis=1)[:, :k]
        best_scores = np.take_along_axis(all_scores, top, axis=1)
        best_ids = np.take_along_axis(all_ids, top, axis=1)
    order = np.argsort(-best_scores, axis=1)
    return np.take_along_axis(best_ids, order, axis=1), np.take_along_axis(best_scores, order, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    vectors = np.load(args.data / "vectors.npy", mmap_mode="r")
    queries = np.load(args.data / "queries.npy")
    ids, scores = exact_topk(vectors, queries, args.k, args.chunk)
    np.save(args.data / "ground_truth.npy", ids)
    np.save(args.data / "ground_truth_scores.npy", scores)
    print(f"ground truth: {len(queries)} queries x top-{args.k}")


if __name__ == "__main__":
    main()
