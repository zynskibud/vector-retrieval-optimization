"""Benchmark one index type inside one database, with the language benches' CLI and JSON.

Run inside the dbbench container (make dbbench ARGS="..."):
  python -m tools.db.bench --db qdrant --index hnsw --data data/processed/dev --out results/raw/dev/qdrant-hnsw.json --search ef=64

CONTRACT sections 2 to 4 apply; tools/db/README.md says how a database maps to them.

Phase 5 (CONTRACT section 13): --delete del10|del30|del50 or --update upd10, then optionally
--compact, run after the build and before the warm-up. Each step is timed into extra
(delete_s, deleted_rows, update_s, updated_rows, compact_s). extra.disk_bytes / index_bytes are
the stats after the build; extra.disk_bytes_after / index_bytes_after are the stats when the
searches start (after the compaction with --compact, else right after the change). The full
stats() dicts are in extra.stats_changed (after the change) and extra.stats_after (after compact).
Every search run carries search_params.deleted / updated and compacted = 0|1.
"""

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.db import base


class UsageError(Exception):
    """Unknown database, index, or parameter: exit code 2."""


def parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_pairs(items: list[str], defaults: dict, what: str) -> dict:
    out = dict(defaults)
    for item in items:
        for pair in filter(None, item.split(",")):
            key, sep, value = pair.partition("=")
            if not sep or not value:
                raise UsageError(f"bad {what} parameter {pair!r}, want KEY=VALUE")
            if key not in defaults:
                raise UsageError(f"unknown {what} parameter {key!r}; known: {sorted(defaults)}")
            out[key] = parse_value(value)
    return out


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def machine() -> dict:
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "cpu part", "hardware")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"os": sys.platform, "arch": platform.machine(), "cpu": cpu or platform.processor() or "unknown", "cores": os.cpu_count()}


def run_search(client, queries: np.ndarray, k: int, params: dict) -> dict:
    """Timed loop: one query at a time through the client library (round trip included)."""
    q = len(queries)
    ids, scores, latency = [], [], []
    t_start = time.perf_counter()
    for i in range(q):
        t0 = time.perf_counter()
        row_ids, row_scores = client.search(queries[i], k, params)
        latency.append((time.perf_counter() - t0) * 1000.0)
        row_ids, row_scores = base.pad(row_ids, row_scores, k)
        ids.append([int(x) for x in row_ids])
        scores.append([None if s is None or math.isinf(s) else s for s in row_scores])
    total = time.perf_counter() - t_start
    return {
        "search_params": params, "ids": ids, "scores": scores, "latency_ms": latency,
        "total_s": total, "qps": q / total, "distance_computations": None, "extra": {},
    }


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="dbbench")
    ap.add_argument("--db", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build", action="append", default=[])
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--threads", type=int, default=None, help="accepted and ignored: the server chooses")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--delete", default=None, help="Phase 5: del10 | del30 | del50")
    ap.add_argument("--update", default=None, help="Phase 5: upd10")
    ap.add_argument("--compact", action="store_true", help="Phase 5: run the database's repair after the change")
    return ap.parse_args(argv)


def check_changes(args) -> None:
    if args.delete and args.update:
        raise UsageError("--delete and --update are not combined in one run (CONTRACT section 13.2)")
    if args.delete and args.delete not in base.DELETE_NAMES:
        raise UsageError(f"unknown delete set {args.delete!r}; known: {base.DELETE_NAMES}")
    if args.update and args.update not in base.UPDATE_NAMES:
        raise UsageError(f"unknown update set {args.update!r}; known: {base.UPDATE_NAMES}")
    if args.compact and not (args.delete or args.update):
        raise UsageError("--compact needs --delete or --update")


def apply_changes(client, args, n: int, meta) -> tuple[dict, dict]:
    """Run --delete / --update / --compact. Returns (extra keys, search_params keys)."""
    extra: dict = {}
    if args.delete:
        ids = base.delete_ids(args.data, args.delete, n)
        extra["delete_s"] = client.delete(ids)
        extra["deleted_rows"] = int(len(ids))
    if args.update:
        ids, vecs = base.update_rows(args.data, args.update, n)
        extra["update_s"] = client.update(ids, vecs, meta.take(ids))
        extra["updated_rows"] = int(len(ids))
    after = client.stats()
    extra["stats_changed"] = after
    if args.compact:
        detail = client.compact()
        extra["compact_s"] = float(detail.pop("compact_s"))
        extra["compact_detail"] = detail
        after = client.stats()
        extra["stats_after"] = after
    for key in ("disk_bytes", "index_bytes", "table_bytes", "segments_count", "rows"):
        if key in after:
            extra[f"{key}_after"] = after[key]
    marks = {"compacted": int(bool(args.compact))}
    if args.delete:
        marks["deleted"] = args.delete
    if args.update:
        marks["updated"] = args.update
    return extra, marks


def run(args) -> dict:
    if args.db not in base.DATABASES:
        raise UsageError(f"unknown database {args.db!r}; known: {base.DATABASES}")
    if args.index not in base.BUILD_DEFAULTS:
        raise UsageError(f"unknown index {args.index!r}; known: {sorted(base.BUILD_DEFAULTS)}")
    if args.index not in base.SUPPORTED[args.db]:
        raise UsageError(f"{args.db} has no {args.index}")
    build_params = parse_pairs(args.build, base.build_defaults(args.db, args.index), "build")
    searches = [parse_pairs([s], base.search_defaults(args.db, args.index), "search") for s in args.search]
    searches = searches or [base.search_defaults(args.db, args.index)]
    check_changes(args)
    for sp in searches:
        if str(sp.get("filter", "none")) not in base.FILTER_NAMES:
            raise UsageError(f"unknown filter {sp['filter']!r}; known: {base.FILTER_NAMES}")

    vectors = np.load(args.data / "vectors.npy", mmap_mode="r")
    if args.limit:
        vectors = vectors[: args.limit]
    vectors = np.ascontiguousarray(vectors)
    queries = np.load(args.data / "queries.npy")
    meta = pq.read_table(args.data / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, len(vectors))
    n, dim = vectors.shape

    # Rows that pass each filter among the loaded rows (CONTRACT section 11.2, extra.filter_rows).
    filter_rows = {f: int(np.load(args.data / f"filter_{f}.npy")[:n].sum())
                   for f in {str(sp.get("filter", "none")) for sp in searches} if f != "none"}

    client = base.get_client(args.db)
    client.data_dir = args.data
    client.connect()
    try:
        client.reset()
        load_s = client.load(vectors, meta, args.batch)
        build_s = client.build_index(args.index, build_params)
        stats = client.stats()
        changes, marks = ({}, {}) if not (args.delete or args.update) else apply_changes(client, args, n, meta)
        for i in range(min(args.warmup, len(queries))):
            client.search(queries[i], args.k, searches[0])
        results = [run_search(client, queries, args.k, p) for p in searches]
        for r in results:
            r["search_params"] = {**r["search_params"], **marks}
            if "filter" in r["search_params"]:
                r["extra"]["filter_rows"] = filter_rows.get(str(r["search_params"]["filter"]), n)
    finally:
        client.close()

    return {
        "contract_version": 1, "language": args.db, "index": args.index, "data_dir": str(args.data),
        "n": n, "dim": dim, "q": len(queries), "k": args.k, "threads": 0, "seed": args.seed,
        "build_params": build_params,
        "build": {"train_s": 0.0, "add_s": load_s + build_s, "total_s": load_s + build_s,
                  "peak_rss_mb": peak_rss_mb(), "index_bytes": int(stats.pop("index_bytes", 0))},
        "searches": results, "machine": machine(),
        "extra": {"load_s": load_s, "server_build_s": build_s, **stats, **changes},
    }


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        result = run(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result))
    except UsageError as e:
        print(f"dbbench: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - any failure is exit 1 with the message
        print(f"dbbench: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
