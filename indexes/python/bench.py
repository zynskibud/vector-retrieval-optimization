"""Build one index, run the queries, write one result JSON (CONTRACT 2-4).

Run: uv run python -m indexes.python.bench --index flat --data data/processed/dev --out r.json
"""

import os

# CONTRACT 4: search runs on one thread. NumPy's BLAS (Accelerate on macOS, OpenBLAS
# elsewhere) reads these before it loads, so they must be set before `import numpy`.
for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import kmeans
from .npy import read_npy

INDEXES = ("flat", "ivf", "pq", "ivf_pq", "hnsw", "diskann")


class UsageError(Exception):
    """Unknown index, unknown parameter, or bad value: exit code 2."""


def parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_pairs(items: list[str], allowed: dict, what: str) -> dict:
    """Parse KEY=VALUE items (each may hold several, comma-separated) over the defaults."""
    out = dict(allowed)
    for item in items:
        for pair in filter(None, item.split(",")):
            key, sep, value = pair.partition("=")
            if not sep or not value:
                raise UsageError(f"bad {what} parameter {pair!r}, want KEY=VALUE")
            if key not in allowed:
                raise UsageError(f"unknown {what} parameter {key!r}; known: {sorted(allowed)}")
            out[key] = parse_value(value)
    return out


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if out.stdout.strip():
                return out.stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def machine() -> dict:
    return {"os": sys.platform, "arch": platform.machine(), "cpu": cpu_name(), "cores": os.cpu_count()}


def run_search(mod, index, queries: np.ndarray, k: int, params: dict) -> dict:
    """Timed loop: one query at a time, in order, on one thread.

    Index modules set index["distance_computations"] and, optionally, index["search_extra"]
    (a dict of per-query counters) inside search(); bench averages them over the queries.
    """
    q = len(queries)
    ids = np.empty((q, k), dtype=np.int64)
    scores = np.empty((q, k), dtype=np.float32)
    latency = []
    dist = []
    counters: dict[str, list[float]] = {}
    t_start = time.perf_counter()
    for i in range(q):
        t0 = time.perf_counter()
        row_ids, row_scores = mod.search(index, queries[i], k, params)
        latency.append((time.perf_counter() - t0) * 1000.0)
        ids[i], scores[i] = row_ids, row_scores
        dist.append(index.get("distance_computations"))
        for key, value in index.get("search_extra", {}).items():  # per-query counters, e.g. disk_reads
            counters.setdefault(key, []).append(value)
    total = time.perf_counter() - t_start
    return {
        "search_params": params,
        "ids": ids.tolist(),
        "scores": [[None if math.isinf(s) else s for s in row] for row in scores.tolist()],
        "latency_ms": latency,
        "total_s": total,
        "qps": q / total,
        "distance_computations": None if None in dist else float(np.mean(dist)),
        "extra": {key: float(np.mean(values)) for key, values in counters.items()},
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="bench")
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build", action="append", default=[])
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args(argv)


def resolve_params(args):
    """Return (module, build_params, [search_params, ...]) or raise UsageError."""
    if args.index not in INDEXES:
        raise UsageError(f"unknown index {args.index!r}; known: {list(INDEXES)}")
    mod = importlib.import_module(f".{args.index}", __package__)
    build_params = parse_pairs(args.build, mod.BUILD_PARAMS, "build")
    searches = [parse_pairs([s], mod.SEARCH_PARAMS, "search") for s in args.search]
    return mod, build_params, searches or [dict(mod.SEARCH_PARAMS)]


def run(args) -> dict:
    mod, build_params, searches = resolve_params(args)  # validate before the slow load
    vectors = read_npy(args.data / "vectors.npy", args.limit)
    queries = read_npy(args.data / "queries.npy")
    n, dim = vectors.shape
    if build_params.get("train_size", 0) is None:  # ivf: 6.2 default depends on nlist and N
        build_params["train_size"] = kmeans.default_train_size(n, build_params["nlist"])

    if hasattr(mod, "OUT_PATH"):  # diskann writes <out>.diskann next to the output JSON (CONTRACT 6.7)
        mod.OUT_PATH = args.out
    index = mod.build(vectors, build_params, args.threads, args.seed)
    build = {
        "train_s": index["train_s"],
        "add_s": index["add_s"],
        "total_s": index["train_s"] + index["add_s"],
        "peak_rss_mb": peak_rss_mb(),
        "index_bytes": int(mod.index_bytes(index)),
    }
    for i in range(min(args.warmup, len(queries))):
        mod.search(index, queries[i], args.k, searches[0])
    results = [run_search(mod, index, queries, args.k, p) for p in searches]
    return {
        "contract_version": 1,
        "language": "python",
        "index": args.index,
        "data_dir": str(args.data),
        "n": n,
        "dim": dim,
        "q": len(queries),
        "k": args.k,
        "threads": args.threads,
        "seed": args.seed,
        "build_params": build_params,
        "build": build,
        "searches": results,
        "machine": machine(),
        "extra": index.get("extra", {}),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:  # argparse errors are usage errors
        return 2 if e.code else 0
    try:
        result = run(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result))
    except UsageError as e:
        print(f"bench: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"bench: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
