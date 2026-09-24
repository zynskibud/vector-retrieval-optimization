"""FAISS reference bench. Same command line (CONTRACT section 2) and output JSON (section 3)
as the hand-built benches, with language = "faiss".

Mapping of contract params to FAISS:
  flat    IndexFlatIP
  ivf     IndexIVFFlat(IndexFlatIP, METRIC_INNER_PRODUCT), spherical k-means (normalized centers, 6.2)
  pq      IndexPQ(d, m, nbits, metric); ip -> METRIC_INNER_PRODUCT, l2 -> METRIC_L2
  ivf_pq  IndexIVFPQ(IndexFlatIP, d, nlist, m, nbits, metric), by_residual (FAISS default)
  hnsw    IndexHNSWFlat(d, m, METRIC_INNER_PRODUCT)
  diskann not in FAISS: exit 2.

FAISS k-means and HNSW use their own RNG and their own level draw, not SplitMix64 (section 5),
so recall differs slightly from the hand-built versions at the same params.
FAISS returns squared distances for METRIC_L2; scores are the negative squared distance (6.4.1).

Run: uv run python -m tools.bench.faiss_ref --index hnsw --data data/processed/dev --out X.json --search ef=64
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import faiss
import numpy as np

BUILD_DEFAULTS = {
    "flat": {},
    "ivf": {"nlist": 1024, "train_size": None, "iters": 20},
    "pq": {"m": 48, "nbits": 8, "metric": "ip", "train_size": 100000, "iters": 20},
    "ivf_pq": {"nlist": 1024, "m": 48, "nbits": 8, "metric": "ip", "train_size": 100000, "iters": 20},
    "hnsw": {"m": 16, "ef_construct": 100},
}
SEARCH_DEFAULTS = {
    "flat": {},
    "ivf": {"nprobe": 8},
    "pq": {"rerank": 0},
    "ivf_pq": {"nprobe": 8, "rerank": 0},
    "hnsw": {"ef": 64},
}
STRING_PARAMS = {"metric": {"ip", "l2"}}


def usage_error(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def parse_params(pairs: list[str], defaults: dict, what: str) -> dict:
    """Parse KEY=VALUE items (each may hold several comma-separated pairs) over the defaults."""
    params = dict(defaults)
    for item in pairs:
        for pair in item.split(","):
            key, sep, value = pair.partition("=")
            if not sep or key not in defaults:
                usage_error(f"unknown {what} parameter '{pair}'; known: {sorted(defaults)}")
            if key in STRING_PARAMS:
                if value not in STRING_PARAMS[key]:
                    usage_error(f"{key} must be one of {sorted(STRING_PARAMS[key])}, got '{value}'")
                params[key] = value
            else:
                try:
                    params[key] = int(value)
                except ValueError:
                    usage_error(f"{what} parameter {key} must be an integer, got '{value}'")
    return params


def faiss_metric(name: str) -> int:
    return faiss.METRIC_INNER_PRODUCT if name == "ip" else faiss.METRIC_L2


def build(name: str, x: np.ndarray, p: dict, seed: int) -> tuple:
    """Return (index, train_s, add_s)."""
    d = x.shape[1]
    if name == "flat":
        index = faiss.IndexFlatIP(d)
    elif name == "ivf":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, p["nlist"], faiss.METRIC_INNER_PRODUCT)
        index.cp.spherical = True
    elif name == "pq":
        index = faiss.IndexPQ(d, p["m"], p["nbits"], faiss_metric(p["metric"]))
    elif name == "ivf_pq":
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, p["nlist"], p["m"], p["nbits"], faiss_metric(p["metric"]))
        index.cp.spherical = True
    else:
        index = faiss.IndexHNSWFlat(d, p["m"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = p["ef_construct"]
    if name in ("ivf", "ivf_pq"):
        index.cp.niter, index.cp.seed = p["iters"], seed
        index.cp.max_points_per_centroid = 1 << 30  # the contract sets the training size, not FAISS
    if name in ("pq", "ivf_pq"):
        index.pq.cp.niter, index.pq.cp.seed = p["iters"], seed
        index.pq.cp.max_points_per_centroid = 1 << 30
    t0 = time.perf_counter()
    if not index.is_trained:
        index.train(x[: p["train_size"]])
    t1 = time.perf_counter()
    index.add(x)
    return index, t1 - t0, time.perf_counter() - t1


def set_search_params(index, name: str, sp: dict, k: int) -> None:
    if name in ("ivf", "ivf_pq"):
        index.nprobe = sp["nprobe"]
    if name == "hnsw":
        index.hnsw.efSearch = max(sp["ef"], k)


def search_one(index, x: np.ndarray, q: np.ndarray, k: int, rerank: int, metric: str):
    """One query. With rerank > 0, take rerank candidates, re-score with full vectors, keep top k."""
    kk = max(k, rerank)
    dist, ids = index.search(q, kk)
    if rerank <= 0:
        return ids[0, :k], dist[0, :k]
    cand = ids[0][ids[0] >= 0]
    vecs = x[cand]
    scores = vecs @ q[0] if metric == "ip" else -((vecs - q[0]) ** 2).sum(1)
    top = np.argsort(-scores)[:k]
    out_ids = np.full(k, -1, dtype=np.int64)
    out_scores = np.full(k, -np.inf, dtype=np.float32)
    out_ids[: len(top)], out_scores[: len(top)] = cand[top], scores[top]
    return out_ids, out_scores


def reset_stats() -> None:
    faiss.cvar.indexIVF_stats.reset()
    faiss.cvar.hnsw_stats.reset()


def distance_count(name: str, index, n: int, q: int, sp: dict, rerank: int) -> float:
    if name in ("flat", "pq"):
        return float(n + rerank)
    if name == "hnsw":
        return faiss.cvar.hnsw_stats.ndis / q
    return index.nlist + faiss.cvar.indexIVF_stats.ndis / q + rerank


def index_bytes(name: str, index, n: int, d: int, p: dict) -> int:
    if name == "flat":
        return 0
    ivf = p.get("nlist", 0) * d * 4 + n * 8  # FAISS list IDs are int64
    pq = p.get("m", 0) * 256 * (d // max(p.get("m", 1), 1)) * 4 + n * p.get("m", 0)
    if name == "ivf":
        return ivf
    if name == "pq":
        return pq
    if name == "ivf_pq":
        return ivf + pq
    hnsw = index.hnsw
    return int(hnsw.neighbors.size() * 4 + hnsw.levels.size() * 4 + hnsw.offsets.size() * 8)


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return {"os": sys.platform, "arch": platform.machine(), "cpu": cpu, "cores": os.cpu_count()}


def score_list(row) -> list:
    return [float(s) if np.isfinite(s) else None for s in row]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build", action="append", default=[])
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    name = args.index
    if name == "diskann":
        usage_error("faiss has no diskann; compare with Milvus in Phase 2")
    if name not in BUILD_DEFAULTS:
        usage_error(f"unknown index '{name}'; known: {sorted(BUILD_DEFAULTS)}")
    bp = parse_params(args.build, BUILD_DEFAULTS[name], "build")
    sps = [parse_params([s], SEARCH_DEFAULTS[name], "search") for s in args.search] or [dict(SEARCH_DEFAULTS[name])]

    x = np.ascontiguousarray(np.load(args.data / "vectors.npy", mmap_mode="r")[: args.limit], dtype=np.float32)
    queries = np.load(args.data / "queries.npy").astype(np.float32)
    n, d = x.shape
    if name == "ivf" and bp["train_size"] is None:
        bp["train_size"] = max(min(n, 256 * bp["nlist"]), bp["nlist"])
    if "m" in bp and name != "hnsw" and d % bp["m"]:
        usage_error(f"dim {d} is not divisible by m={bp['m']}")
    if bp.get("nbits", 8) != 8:
        usage_error("only nbits=8 is supported")

    faiss.omp_set_num_threads(args.threads)
    index, train_s, add_s = build(name, x, bp, args.seed)
    build_info = {"train_s": train_s, "add_s": add_s, "total_s": train_s + add_s,
                  "peak_rss_mb": peak_rss_mb(), "index_bytes": index_bytes(name, index, n, d, bp)}
    print(f"built {name} n={n} in {train_s + add_s:.2f}s", file=sys.stderr)

    faiss.omp_set_num_threads(1)
    metric = bp.get("metric", "ip")
    set_search_params(index, name, sps[0], args.k)
    for i in range(min(args.warmup, len(queries))):
        search_one(index, x, queries[i : i + 1], args.k, sps[0].get("rerank", 0), metric)

    searches = []
    for sp in sps:
        set_search_params(index, name, sp, args.k)
        rerank = sp.get("rerank", 0)
        ids, scores, lat = [], [], []
        reset_stats()
        t_loop = time.perf_counter()
        for i in range(len(queries)):
            t0 = time.perf_counter()
            r_ids, r_scores = search_one(index, x, queries[i : i + 1], args.k, rerank, metric)
            lat.append((time.perf_counter() - t0) * 1000)
            ids.append(r_ids)
            scores.append(r_scores)
        total_s = time.perf_counter() - t_loop
        ids = np.asarray(ids)
        scores = np.asarray(scores, dtype=np.float64)
        if metric == "l2" and rerank <= 0:
            scores = -scores
        scores[ids < 0] = -np.inf
        searches.append({
            "search_params": sp, "ids": ids.tolist(), "scores": [score_list(r) for r in scores],
            "latency_ms": lat, "total_s": total_s, "qps": len(queries) / total_s,
            "distance_computations": distance_count(name, index, n, len(queries), sp, rerank),
            "extra": {},
        })
        print(f"search {sp}: p50 {np.median(lat):.3f} ms", file=sys.stderr)

    doc = {
        "contract_version": 1, "language": "faiss", "index": name, "data_dir": str(args.data),
        "n": n, "dim": d, "q": len(queries), "k": args.k, "threads": args.threads, "seed": args.seed,
        "build_params": bp, "build": build_info, "searches": searches, "machine": machine_info(),
        "extra": {"faiss_version": faiss.__version__, "id_type": "int64"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc))


if __name__ == "__main__":
    main()
