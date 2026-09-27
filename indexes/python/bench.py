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
import threading
import time
from pathlib import Path

import numpy as np

from . import changes, kmeans
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


def cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def run_load(mod, index, queries, k, params, clients, duration, inserter=None) -> dict:
    """Load run (CONTRACT 12.1): `clients` threads, closed loop over the queries for `duration`
    seconds (0 = one pass each). Worker 0's first pass supplies ids and scores; worker 0
    finishes that pass even after the deadline (the other workers stop at the deadline).

    `inserter`, if given, is a callable(deadline) run in one more thread during the loop.
    """
    q = len(queries)
    ids = np.full((q, k), -1, dtype=np.int64)
    scores = np.full((q, k), -np.inf, dtype=np.float32)
    lat: list[list[float]] = [[] for _ in range(clients)]
    errors = [0] * clients
    start = threading.Barrier(clients + 1 + (inserter is not None))
    box = {}

    def worker(w):
        mine = lat[w]
        start.wait()
        deadline = box["deadline"]
        first = w == 0
        while True:
            for i in range(q):
                # Worker 0 always finishes its first pass (it supplies ids and scores), so
                # the loop can run past `duration`; the wall time used for qps includes that.
                if duration > 0 and not first and time.perf_counter() >= deadline:
                    return
                t0 = time.perf_counter()
                try:
                    row_ids, row_scores = mod.search(index, queries[i], k, params)
                except Exception:
                    errors[w] += 1
                    continue
                mine.append((time.perf_counter() - t0) * 1000.0)
                if first:
                    ids[i], scores[i] = row_ids, row_scores
            first = False
            if duration <= 0:
                return

    def run_inserter():
        start.wait()
        inserter(box["deadline"])

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(clients)]
    if inserter is not None:
        threads.append(threading.Thread(target=run_inserter))
    for t in threads:
        t.start()
    cpu0 = cpu_seconds()
    t_start = time.perf_counter()
    box["deadline"] = t_start + duration
    start.wait()
    for t in threads:
        t.join()
    total = time.perf_counter() - t_start
    cpu = cpu_seconds() - cpu0
    latency = [x for w in lat for x in w]
    done = len(latency)
    return {
        "search_params": params,
        "ids": ids.tolist(),
        "scores": [[None if math.isinf(s) else s for s in row] for row in scores.tolist()],
        "latency_ms": latency,
        "total_s": total,
        "qps": done / total,
        "distance_computations": None,  # a per-query counter shared by threads is not kept
        "extra": {
            "errors": sum(errors),
            "cpu_pct": cpu / total * 100.0,
            "clients": clients,
            "duration_s": duration,
            "queries_done": done,
        },
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
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--insert-rate", type=float, default=0.0)
    ap.add_argument("--delete", default=None)
    ap.add_argument("--update", default=None)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--compact-mode", default="rebuild")
    return ap.parse_args(argv)


def resolve_params(args):
    """Return (module, build_params, [search_params, ...]) or raise UsageError."""
    if args.index not in INDEXES:
        raise UsageError(f"unknown index {args.index!r}; known: {list(INDEXES)}")
    mod = importlib.import_module(f".{args.index}", __package__)
    build_params = parse_pairs(args.build, mod.BUILD_PARAMS, "build")
    searches = [parse_pairs([s], mod.SEARCH_PARAMS, "search") for s in args.search]
    return mod, build_params, searches or [dict(mod.SEARCH_PARAMS)]


def run_load_all(mod, index, vectors, queries, args, searches, n_build) -> list:
    """One load run per search setting. With inserts, the inserter runs during the first
    setting's loop only; then the repair pass and a one-thread after-inserts pass."""
    n = len(vectors)
    stats = {"rows": 0, "batch_ms": []}

    stats["next"] = n_build

    def add_batch():
        row = stats["next"]
        batch = list(range(row, min(row + 100, n)))
        b0 = time.perf_counter()
        mod.insert(index, batch, vectors[batch[0] : batch[-1] + 1])
        stats["batch_ms"].append((time.perf_counter() - b0) * 1000.0)
        stats["next"] = batch[-1] + 1

    def inserter(deadline):
        rate = args.insert_rate
        t0 = time.perf_counter()
        while stats["next"] < n:
            row = stats["next"]
            due = t0 + (row - n_build) / rate  # paced: batch j starts at t0 + 100 j / R
            now = time.perf_counter()
            if now >= deadline:
                return
            if due > now:
                time.sleep(min(due - now, deadline - now))
                continue
            add_batch()
        stats["rows"] = stats["next"] - n_build

    results = []
    for j, p in enumerate(searches):
        ins = inserter if (n_build is not None and j == 0) else None
        results.append(run_load(mod, index, queries, args.k, p, args.clients, args.duration, ins))
    if n_build is not None:
        # CONTRACT 12.2 insert tail: untimed, no searches running, until every row is in.
        during = stats["next"] - n_build
        t0 = time.perf_counter()
        while stats["next"] < n:
            add_batch()
        tail_s = time.perf_counter() - t0
        stats["rows"] = stats["next"] - n_build
        t0 = time.perf_counter()
        added_a, added_b = mod.repair(index)
        repair_s = time.perf_counter() - t0
        final = run_search(mod, index, queries, args.k, {**searches[0], "phase": "after_inserts"})
        final["extra"].update({
            "inserted_rows": stats["rows"],
            "inserted_during_loop": during,
            "insert_tail_s": tail_s,
            "insert_p50_ms": float(np.median(stats["batch_ms"])) if stats["batch_ms"] else None,
            "insert_batches": len(stats["batch_ms"]),
            "repair_added": added_a,
            "repair_added_unreachable": added_b,
            "repair_s": repair_s,
        })
        results[0]["extra"].update({"inserted_rows": stats["rows"], "inserted_during_loop": during,
                                    "insert_p50_ms": final["extra"]["insert_p50_ms"]})
        results.append(final)
    return results


CHANGE_INDEXES = ("flat", "ivf", "hnsw")


def check_changes(args, load: bool) -> bool:
    """Validate the Phase 5 flags (CONTRACT 13.2). Returns True if the run applies a change."""
    changed = args.delete is not None or args.update is not None
    if not changed and not args.compact and args.compact_mode == "rebuild":
        return False
    if args.index not in CHANGE_INDEXES:
        raise UsageError(f"--delete/--update/--compact: only {list(CHANGE_INDEXES)}, not {args.index}")
    if args.delete is not None and args.update is not None:
        raise UsageError("--delete and --update are not combined in one run")
    if not changed:
        raise UsageError("--compact/--compact-mode need --delete or --update")
    if args.delete is not None and args.delete not in changes.DELETES:
        raise UsageError(f"unknown delete set {args.delete!r}; known: {list(changes.DELETES)}")
    if args.update is not None and args.update not in changes.UPDATES:
        raise UsageError(f"unknown update set {args.update!r}; known: {list(changes.UPDATES)}")
    if args.compact_mode not in changes.COMPACT_MODES:
        raise UsageError(f"unknown compact mode {args.compact_mode!r}; known: {list(changes.COMPACT_MODES)}")
    if load:
        raise UsageError("--delete/--update are not combined with load runs")
    return True


def apply_changes(mod, index, args, n: int, extra: dict):
    """Delete or update, then optionally compact; each timed into extra. Returns the index
    (compaction may return a new one) and the search_params labels."""
    labels = {}
    if args.delete is not None:
        mask = changes.delete_mask(args.data, args.delete, n)
        t0 = time.perf_counter()
        index = mod.delete(index, mask)
        extra["delete_s"] = time.perf_counter() - t0
        extra["deleted_rows"] = int(mask.sum())
        labels["deleted"] = args.delete
    else:
        ids, vecs = changes.update_set(args.data, args.update, n)
        t0 = time.perf_counter()
        index = mod.update(index, ids, vecs)
        extra["update_s"] = time.perf_counter() - t0
        extra["updated_rows"] = int(len(ids))
        labels["updated"] = args.update
    extra.update(index.get("extra", {}))  # e.g. hnsw update_repair_added
    extra["index_bytes_before_compact"] = int(mod.index_bytes(index))  # after the change, before compaction
    labels["compacted"] = 0
    if args.compact:
        t0 = time.perf_counter()
        index = mod.compact(index, args.compact_mode)
        extra["compact_s"] = time.perf_counter() - t0
        extra["compact_mode"] = args.compact_mode
        extra["index_bytes_after"] = int(mod.index_bytes(index))
        for key, value in index.get("extra", {}).items():
            if key.startswith("compact_"):
                extra[key] = value
        labels["compacted"] = 1
    return index, labels


def run(args) -> dict:
    mod, build_params, searches = resolve_params(args)  # validate before the slow load
    load = args.clients != 1 or args.duration > 0 or args.insert_rate > 0
    if args.clients < 1 or args.duration < 0 or args.insert_rate < 0:
        raise UsageError("need --clients >= 1, --duration >= 0, --insert-rate >= 0")
    if load and not hasattr(mod, "insert"):
        raise UsageError(f"--clients/--duration/--insert-rate: only hnsw supports load runs, not {args.index}")
    changed = check_changes(args, load)
    vectors = read_npy(args.data / "vectors.npy", args.limit)
    queries = read_npy(args.data / "queries.npy")
    n, dim = vectors.shape
    if build_params.get("train_size", 0) is None:  # ivf: 6.2 default depends on nlist and N
        build_params["train_size"] = kmeans.default_train_size(n, build_params["nlist"])

    if hasattr(mod, "OUT_PATH"):  # diskann writes <out>.diskann next to the output JSON (CONTRACT 6.7)
        mod.OUT_PATH = args.out
    if hasattr(mod, "DATA_DIR"):  # flat, ivf, hnsw read filter_<name>.npy from here (CONTRACT 11)
        mod.DATA_DIR = args.data
        from . import filters
        for p in searches:  # a bad filter name is a usage error, found before the slow build
            if str(p.get("filter", "none")) not in filters.NAMES:
                raise UsageError(f"unknown filter {p['filter']!r}; known: {list(filters.NAMES)}")
    n_build = n - n // 10 if args.insert_rate > 0 else None  # CONTRACT 12.2: build on the first 90%
    if n_build is None:
        index = mod.build(vectors, build_params, args.threads, args.seed)
    else:
        index = mod.build(vectors, build_params, args.threads, args.seed, n_build=n_build)
    build = {
        "train_s": index["train_s"],
        "add_s": index["add_s"],
        "total_s": index["train_s"] + index["add_s"],
        "peak_rss_mb": peak_rss_mb(),
        "index_bytes": int(mod.index_bytes(index)),
    }
    extra = dict(index.get("extra", {}))
    if changed:
        index, labels = apply_changes(mod, index, args, n, extra)
        searches = [{**p, **labels} for p in searches]
    for i in range(min(args.warmup, len(queries))):
        mod.search(index, queries[i], args.k, searches[0])
    if not load:
        results = [run_search(mod, index, queries, args.k, p) for p in searches]
    else:
        results = run_load_all(mod, index, vectors, queries, args, searches, n_build)
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
        "extra": extra,
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
