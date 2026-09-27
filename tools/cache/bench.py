"""Embedding cache bench (CONTRACT section 14.3).

Builds the FAISS HNSW reference (m=16, ef_construct=100, inner product) on the data set once,
then replays a workload one request at a time: cache get; on a miss, embed and put; then
search (k = 10, efSearch = 64). Per request it records hit or miss, embed time (0 on a hit),
search time, and end-to-end time. The output is CONTRACT section 3 JSON with language "cache".

- latency_ms: end-to-end time of every request (R values, not q).
- ids / scores: the results of the first q = min(1000, R) requests; extra.request_pool_ids holds
  their pool_id, so the report scores rows with pool_id < Q (the query texts) against
  ground_truth.npy[pool_id].
- extra.recall_on_queries: the same check over all R requests, computed here only as a check
  that the embedding path reproduces the stored query vectors. The report's recall comes from
  tools/bench/metrics.py.
- --invalidate-at R0: from request R0 on, the model version is "<v>-2", so every old key misses.

Run (inside dbbench for --backend redis):
  uv run python -m tools.cache.bench --data data/processed/dev --workload zipf --backend lru \
      --capacity 2000 --out results/raw/dev/cache-lru-c2000-zipf.json
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from tools.cache import cache as cache_mod
from tools.cache import workload as workload_mod

K, EF = 10, 64
BUILD_PARAMS = {"m": 16, "ef_construct": 100}
IDS_ROWS = 1000


def replay(texts: list[str], cache, embed_one: Callable[[str], np.ndarray], search: Callable | None,
           model_name: str, model_version: str, invalidate_at: int | None = None) -> dict:
    """Replay texts one at a time. Returns per-request arrays: hit (bool), embed_ms, search_ms,
    e2e_ms, and ids / scores (R x K) when search is given."""
    r = len(texts)
    hit = np.zeros(r, dtype=bool)
    embed_ms, search_ms, e2e_ms = np.zeros(r), np.zeros(r), np.zeros(r)
    ids = np.full((r, K), -1, dtype=np.int64) if search else None
    scores = np.zeros((r, K), dtype=np.float32) if search else None
    version = model_version
    for i, text in enumerate(texts):
        if invalidate_at is not None and i == invalidate_at:
            version = f"{model_version}-2"
        t0 = time.perf_counter()
        key = cache_mod.key(text, model_name, version)
        vec = cache.get(key)
        if vec is None:
            t1 = time.perf_counter()
            vec = embed_one(text)
            embed_ms[i] = (time.perf_counter() - t1) * 1000
            cache.put(key, vec)
        else:
            hit[i] = True
        if search:
            t2 = time.perf_counter()
            s, j = search(vec)
            search_ms[i] = (time.perf_counter() - t2) * 1000
        e2e_ms[i] = (time.perf_counter() - t0) * 1000
        if search:
            ids[i], scores[i] = j, s
    return {"hit": hit, "embed_ms": embed_ms, "search_ms": search_ms, "e2e_ms": e2e_ms, "ids": ids, "scores": scores}


def load_workload(data: Path, name: str, requests: int, workload_dir: Path) -> pq.ParquetFile:
    path = workload_dir / f"workload_{name}.parquet"
    if path.exists():
        t = pq.read_table(path)
        if t.num_rows == requests:
            return t
    print(f"building {path} ({requests} requests)", file=sys.stderr)
    return pq.read_table(workload_mod.build(data, name, requests=requests, out_dir=workload_dir))


def main() -> None:
    from tools.bench.faiss_ref import build as faiss_build
    from tools.bench.faiss_ref import index_bytes, machine_info, peak_rss_mb

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workload", choices=workload_mod.NAMES, default="zipf")
    ap.add_argument("--backend", choices=("none", "lru", "redis"), default="lru")
    ap.add_argument("--capacity", type=int, default=2000, help="lru: entries; redis: label only (the server's maxmemory is the limit)")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--invalidate-at", type=int, default=None)
    ap.add_argument("--requests", type=int, default=50000)
    ap.add_argument("--workload-dir", type=Path, default=None, help="default: results/raw/<data-name>/workloads")
    ap.add_argument("--redis-host", default="redis")
    ap.add_argument("--ttl", type=int, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("VRO_THREADS", os.cpu_count())), help="HNSW build threads")
    ap.add_argument("--embed-threads", type=int, default=1, help="torch threads for the embedding (1: CONTRACT section 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=20, help="embeds and searches before the replay, not timed")
    args = ap.parse_args()
    if args.backend == "lru" and args.capacity < 1:
        sys.stderr.write("--capacity must be >= 1 for lru\n")
        sys.exit(2)

    import faiss
    import torch

    from tools.cache.embed import MODEL_NAME, Embedder

    torch.set_num_threads(args.embed_threads)
    wl = load_workload(args.data, args.workload, args.requests, args.workload_dir or workload_mod.default_dir(args.data))
    texts, pool_ids = wl.column("text").to_pylist(), np.asarray(wl.column("pool_id").to_numpy(), dtype=np.int64)
    n_queries = len(np.load(args.data / "queries.npy", mmap_mode="r"))

    x = np.ascontiguousarray(np.load(args.data / "vectors.npy", mmap_mode="r"), dtype=np.float32)
    n, d = x.shape
    faiss.omp_set_num_threads(args.threads)
    index, train_s, add_s = faiss_build("hnsw", x, BUILD_PARAMS, args.seed)
    build_info = {"train_s": train_s, "add_s": add_s, "total_s": train_s + add_s,
                  "peak_rss_mb": peak_rss_mb(), "index_bytes": index_bytes("hnsw", index, n, d, BUILD_PARAMS)}
    print(f"built hnsw n={n} in {train_s + add_s:.2f}s", file=sys.stderr)
    faiss.omp_set_num_threads(1)
    index.hnsw.efSearch = EF

    def search(vec):
        s, j = index.search(np.asarray(vec, dtype=np.float32).reshape(1, d), K)
        return s[0], j[0]

    embedder = Embedder(MODEL_NAME, args.model_version)
    embed_one = lambda t: embedder.embed([t])[0]
    for t in texts[-args.warmup:] if args.warmup else []:  # warm the model and the index, outside the cache
        search(embed_one(t))

    cache = cache_mod.make(args.backend, args.capacity, host=args.redis_host, ttl=args.ttl)
    t_loop = time.perf_counter()
    res = replay(texts, cache, embed_one, search, MODEL_NAME, args.model_version, args.invalidate_at)
    total_s = time.perf_counter() - t_loop
    st = cache.stats()

    hit = res["hit"]
    is_q = pool_ids < n_queries
    gt = np.load(args.data / "ground_truth.npy")[:, :K]
    rec = [len(set(res["ids"][i].tolist()) & set(gt[pool_ids[i]].tolist())) / K for i in np.flatnonzero(is_q)]
    q = min(IDS_ROWS, len(texts))
    extra = {
        "requests": len(texts), "hit_rate": float(hit.mean()),
        "embed_p50_ms": float(np.median(res["embed_ms"][~hit])) if (~hit).any() else 0.0,
        "search_p50_ms": float(np.median(res["search_ms"])),
        "e2e_p50_ms": float(np.percentile(res["e2e_ms"], 50)), "e2e_p99_ms": float(np.percentile(res["e2e_ms"], 99)),
        "hit_e2e_p50_ms": float(np.median(res["e2e_ms"][hit])) if hit.any() else None,
        "miss_e2e_p50_ms": float(np.median(res["e2e_ms"][~hit])) if (~hit).any() else None,
        "entries": st["entries"], "cache_bytes": st["bytes"], "evictions": st["evictions"],
        "hits": st["hits"], "misses": st["misses"],
        "recall_on_queries": float(np.mean(rec)) if rec else None, "query_requests": int(is_q.sum()),
        "request_pool_ids": pool_ids[:q].tolist(), "n_queries": n_queries,
    }
    if args.invalidate_at is not None:
        extra["hit_rate_after_invalidate"] = float(hit[args.invalidate_at:].mean())
        extra["hit_rate_before_invalidate"] = float(hit[:args.invalidate_at].mean())
        extra["misses_first_100_after_invalidate"] = int((~hit[args.invalidate_at:args.invalidate_at + 100]).sum())
    sp = {"backend": args.backend, "capacity": args.capacity, "workload": args.workload, "model_version": args.model_version}
    if args.invalidate_at is not None:
        sp["invalidate_at"] = args.invalidate_at
    doc = {
        "contract_version": 1, "language": "cache", "index": "hnsw", "data_dir": str(args.data),
        "n": n, "dim": d, "q": q, "k": K, "threads": args.threads, "seed": args.seed,
        "build_params": BUILD_PARAMS, "build": build_info,
        "searches": [{
            "search_params": sp, "ids": res["ids"][:q].tolist(),
            "scores": [[float(s) for s in row] for row in res["scores"][:q]],
            "latency_ms": res["e2e_ms"].tolist(), "total_s": total_s, "qps": len(texts) / total_s,
            "distance_computations": None, "extra": extra,
        }],
        "machine": machine_info(),
        "extra": {"model": MODEL_NAME, "ef": EF, "embed_threads": args.embed_threads, "faiss_version": faiss.__version__,
                  "workload_requests": len(texts)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc))
    print(f"{args.backend} c={args.capacity} {args.workload}: hit {extra['hit_rate']:.3f} e2e p50 {extra['e2e_p50_ms']:.3f} ms"
          f" recall_on_queries {extra['recall_on_queries']}", file=sys.stderr)


if __name__ == "__main__":
    main()
