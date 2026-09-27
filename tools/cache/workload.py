"""Request streams for the cache bench (CONTRACT section 14.1).

Pool: the Q query texts (query_meta.parquet, pool_id = query ID) plus pool - Q corpus texts
(metadata.parquet `text`, rows chosen with the seed), pool_id Q..pool-1.
zipf: request i draws rank r in 1..pool with P(r) ~ 1 / r^s; rank r maps to a pool text through
one fixed random permutation of the pool (the "ranking"), so the popular texts are a random mix
of queries and corpus paragraphs. uniform: every pool text with the same probability.
Output: <data>/workload_<name>.parquet with columns request_id, pool_id, text.

Run: uv run python -m tools.cache.workload --data data/processed/dev [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NAMES = ("zipf", "uniform")


def pool_texts(data_dir, pool: int = 5000, seed: int = 11) -> list[str]:
    data_dir = Path(data_dir)
    queries = pq.read_table(data_dir / "query_meta.parquet", columns=["text"]).column("text").to_pylist()
    corpus = pq.read_table(data_dir / "metadata.parquet", columns=["text"]).column("text")
    if pool < len(queries):
        raise ValueError(f"pool {pool} is smaller than the {len(queries)} queries")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(corpus), size=pool - len(queries), replace=False))
    return queries + corpus.take(pa.array(rows)).to_pylist()


def draw(name: str, pool: int, requests: int, zipf_s: float, seed: int) -> np.ndarray:
    """pool_id of each request."""
    rng = np.random.default_rng(seed)
    ranking = rng.permutation(pool)           # ranking[r] = pool_id of the text at rank r (0 = most popular)
    if name == "uniform":
        return rng.integers(0, pool, size=requests)
    if name != "zipf":
        raise ValueError(f"unknown workload {name!r}; known: {NAMES}")
    p = 1.0 / np.arange(1, pool + 1, dtype=np.float64) ** zipf_s
    ranks = rng.choice(pool, size=requests, p=p / p.sum())
    return ranking[ranks]


def build(data_dir, name: str, pool: int = 5000, requests: int = 50000, zipf_s: float = 1.1,
          seed: int = 11, out_dir=None) -> Path:
    texts = pool_texts(data_dir, pool, seed)
    ids = draw(name, pool, requests, zipf_s, seed)
    table = pa.table({"request_id": pa.array(np.arange(requests, dtype=np.int64)),
                      "pool_id": pa.array(ids.astype(np.int64)),
                      "text": pa.array([texts[i] for i in ids])})
    out = Path(out_dir or data_dir) / f"workload_{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    ap.add_argument("--out-dir", type=Path, default=None, help="default: results/raw/<data-name>/workloads (the data dir is read-only in the container)")
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--requests", type=int, default=50000)
    ap.add_argument("--zipf-s", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out_dir = args.out_dir or default_dir(args.data)
    for name in NAMES:
        print(build(args.data, name, args.pool, args.requests, args.zipf_s, args.seed, out_dir))


def default_dir(data_dir) -> Path:
    return Path("results/raw") / Path(data_dir).name / "workloads"


if __name__ == "__main__":
    main()
