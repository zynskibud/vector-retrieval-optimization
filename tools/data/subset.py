"""Make a small dev dataset for fast tests during index development.

Takes a random sample of the full corpus, keeps the same 1,000 queries, and
computes ground truth against the sample only. Output has the same file names
and formats as data/processed/, in data/processed/dev/.

  dev_ids.npy  int64 (N_dev,). Row i of the dev corpus is row dev_ids[i] of the full corpus.

Run: uv run python -m tools.data.subset [--size 100000]
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.ground_truth import exact_topk

FULL = Path("data/processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=FULL / "dev")
    args = ap.parse_args()

    vectors = np.load(FULL / "vectors.npy", mmap_mode="r")
    queries = np.load(FULL / "queries.npy")
    rng = np.random.default_rng(args.seed)
    dev_ids = np.sort(rng.choice(len(vectors), size=args.size, replace=False))

    dev_vectors = np.asarray(vectors[dev_ids])
    ids, scores = exact_topk(dev_vectors, queries, k=100, progress=False)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "vectors.npy", dev_vectors)
    np.save(args.out / "queries.npy", queries)
    np.save(args.out / "dev_ids.npy", dev_ids)
    np.save(args.out / "ground_truth.npy", ids)
    np.save(args.out / "ground_truth_scores.npy", scores)
    meta = pq.read_table(FULL / "metadata.parquet")
    pq.write_table(meta.take(pa.array(dev_ids)), args.out / "metadata.parquet")
    pq.write_table(pq.read_table(FULL / "query_meta.parquet"), args.out / "query_meta.parquet")
    print(f"dev corpus: {args.size:,} x {dev_vectors.shape[1]}  queries: {len(queries):,}  -> {args.out}")


if __name__ == "__main__":
    main()
