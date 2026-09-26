"""Make the Phase 3 filter masks and their ground truth (CONTRACT section 11.1).

For each selectivity s in {0.5, 0.1, 0.01, 0.001}, pick the views threshold t so
that about s of the corpus rows have views >= t, write the bool mask, and compute
the exact top-100 among passing rows for every query.

Outputs in the data directory: filters.json, filter_<name>.npy,
ground_truth_<name>.npy, ground_truth_<name>_scores.npy.

Run: uv run python -m tools.data.filters [--data data/processed]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.data.ground_truth import exact_topk

FILTERS = {"top50": 0.5, "top10": 0.1, "top1": 0.01, "top01": 0.001}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    views = pq.read_table(args.data / "metadata.parquet", columns=["views"]).column("views").to_numpy()
    vectors = np.load(args.data / "vectors.npy", mmap_mode="r")
    queries = np.load(args.data / "queries.npy")
    n = len(views)
    info = {}
    for name, s in FILTERS.items():
        t = float(np.quantile(views, 1.0 - s))
        mask = views >= t
        rows = int(mask.sum())
        np.save(args.data / f"filter_{name}.npy", mask)
        # Exact search over the passing rows only, then map back to corpus row IDs.
        passing = np.flatnonzero(mask)
        ids, scores = exact_topk(np.asarray(vectors[passing]), queries, k=min(args.k, rows), progress=False)
        ids = passing[ids]
        np.save(args.data / f"ground_truth_{name}.npy", ids.astype(np.int64))
        np.save(args.data / f"ground_truth_{name}_scores.npy", scores)
        info[name] = {"views_min": t, "selectivity": rows / n, "rows": rows}
        print(f"{name}: views >= {t:.1f}, {rows:,} of {n:,} rows ({rows / n:.4%})")
    (args.data / "filters.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
