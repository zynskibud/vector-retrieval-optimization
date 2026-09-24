"""Check the Phase 0 outputs. Prints one line per check and exits non-zero on a failure.

Run: uv run python -m tools.data.verify [--data data/processed]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    DATA = ap.parse_args().data
    vectors = np.load(DATA / "vectors.npy", mmap_mode="r")
    queries = np.load(DATA / "queries.npy")
    gt = np.load(DATA / "ground_truth.npy")
    gt_scores = np.load(DATA / "ground_truth_scores.npy")
    meta_rows = pq.read_metadata(DATA / "metadata.parquet").num_rows
    qmeta_rows = pq.read_metadata(DATA / "query_meta.parquet").num_rows
    n, dim = vectors.shape
    rng = np.random.default_rng(0)
    sample = np.asarray(vectors[rng.choice(n, min(n, 10_000), replace=False)])

    # Recompute 20 queries with a full sort, independent of the chunked method.
    check_q = rng.choice(len(queries), 20, replace=False)
    full = np.asarray(vectors) @ queries[check_q].T
    exact = np.argsort(-full, axis=0)[:10].T

    checks = [
        ("dtypes are float32 / int64", vectors.dtype == np.float32 and queries.dtype == np.float32 and gt.dtype == np.int64),
        ("dimension is 384", dim == 384 and queries.shape[1] == 384),
        ("metadata rows match vectors", meta_rows == n and qmeta_rows == len(queries)),
        ("corpus vectors have length 1", np.allclose(np.linalg.norm(sample, axis=1), 1, atol=1e-4)),
        ("query vectors have length 1", np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-4)),
        ("ground truth shape is (Q, 100)", gt.shape == (len(queries), 100)),
        ("ground truth IDs are in range", gt.min() >= 0 and gt.max() < n),
        ("ground truth scores sorted best first", bool(np.all(np.diff(gt_scores, axis=1) <= 1e-6))),
        ("no query is in the corpus (top score < 0.9999)", bool(gt_scores[:, 0].max() < 0.9999)),
        ("top-10 matches a full sort on 20 queries", bool(np.all(np.sort(exact, 1) == np.sort(gt[check_q, :10], 1)))),
    ]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"corpus {n:,} x {dim}   queries {len(queries):,}   "
          f"top-1 score min/median/max {gt_scores[:, 0].min():.3f}/{np.median(gt_scores[:, 0]):.3f}/{gt_scores[:, 0].max():.3f}")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
