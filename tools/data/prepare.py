"""Download Wikipedia embedding shards and prepare them for the benchmarks.

Outputs in data/processed/:
  vectors.npy       float32 (N, 384), L2-normalized. The corpus. Row index = vector ID.
  queries.npy       float32 (Q, 384), L2-normalized. Held-out vectors used as queries.
  metadata.parquet  One row per corpus vector, same order as vectors.npy. Includes the text.
  query_meta.parquet  One row per query, same order as queries.npy.

Queries are removed from the corpus, so a query never finds itself.

Run: uv run python -m tools.data.prepare
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

DATASET = "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2"
SHARD_URL = f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/train/{{i}}.parquet"
META_COLUMNS = ["id", "title", "text", "url", "wiki_id", "views", "paragraph_id", "langs"]

RAW = Path("data/raw")
OUT = Path("data/processed")


def download(i: int) -> Path:
    """Download shard i to data/raw/ unless it is already there."""
    path = RAW / f"{i}.parquet"
    if path.exists():
        return path
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.get(SHARD_URL.format(i=i), stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"shard {i}") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(path)  # rename only after a complete download
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    table = pa.concat_tables(
        pq.read_table(download(i), columns=META_COLUMNS + ["emb"]) for i in range(args.shards)
    )
    n = table.num_rows

    # The emb column is a list of floats per row. Flatten it into one (n, dim) array.
    emb = table.column("emb").combine_chunks()
    dim = len(emb[0])
    vectors = emb.values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, dim)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    rng = np.random.default_rng(args.seed)
    is_query = np.zeros(n, dtype=bool)
    is_query[rng.choice(n, size=args.queries, replace=False)] = True

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "vectors.npy", vectors[~is_query])
    np.save(OUT / "queries.npy", vectors[is_query])
    meta = table.select(META_COLUMNS)
    pq.write_table(meta.filter(pa.array(~is_query)), OUT / "metadata.parquet")
    pq.write_table(meta.filter(pa.array(is_query)), OUT / "query_meta.parquet")
    print(f"rows: {n:,}  corpus: {n - args.queries:,} x {dim}  queries: {args.queries:,}")


if __name__ == "__main__":
    main()
