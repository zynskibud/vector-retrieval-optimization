"""Phase 4 load bench for the databases (indexes/CONTRACT.md section 12.3).

Same CLI and JSON as tools/db/bench.py, plus:
  --clients C       worker threads, one client connection each (default 1)
  --duration S      seconds of the closed loop (default 20)
  --insert-rate R   rows per second added during the loop by one inserter thread (default 0)

Run inside the dbbench container:
  python -m tools.load.bench --db qdrant --index hnsw --data data/processed/dev --out results/raw/dev/x.json \
      --search ef=64 --clients 8 --duration 20

Protocol per search setting:
  1. Warm-up on the main connection (--warmup queries), then 10 queries per worker connection.
  2. C threads start together. Worker w loops over the queries from offset w*Q/C, in a closed
     loop, until S seconds pass. Worker 0 starts at query 0; its first pass gives ids and
     scores (it finishes that pass even after S seconds, so recall is always checkable).
  3. qps = queries done / wall time of the loop; latency_ms = every query of every worker;
     cpu_pct = client process CPU time (getrusage) / wall time x 100.
With --insert-rate R the build uses the first 90% of the rows. One inserter thread (its own
connection) adds the rest in batches of 100 rows at R rows/s, until the rows run out or the
loop ends. Then client.finish_inserts() (Milvus: one flush) and a one-thread pass over the
queries: the last search run, search_params.phase = "after_inserts".
"""

import argparse
import json
import math
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.db import base
from tools.db.bench import UsageError, machine, parse_args as db_parse_args, parse_pairs, peak_rss_mb, run_search

INSERT_BATCH = 100
WORKER_WARMUP = 10


def parse_args(argv):
    argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="loadbench", add_help=False)
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--insert-rate", type=float, default=0.0)
    extra, rest = ap.parse_known_args(argv)
    args = db_parse_args(rest)
    args.clients, args.duration, args.insert_rate = extra.clients, extra.duration, extra.insert_rate
    return args


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def open_client(args):
    c = base.get_client(args.db)
    c.data_dir = args.data
    c.connect()
    c.attach(args.index)
    return c


class Inserter(threading.Thread):
    """Adds rows [start, stop) in batches of 100 at `rate` rows/s until done or `stop_event`."""

    def __init__(self, client, vectors, meta, start: int, stop: int, rate: float, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.client, self.vectors, self.meta = client, vectors, meta
        self.start_row, self.stop_row, self.rate, self.stop_event = start, stop, rate, stop_event
        self.inserted, self.errors, self.batch_ms = 0, 0, []
        self.loop_end = float("inf")  # set when the search loop ends
        self.inserted_during_loop = 0

    def run(self) -> None:
        t0 = time.perf_counter()
        for b, s in enumerate(range(self.start_row, self.stop_row, INSERT_BATCH)):
            due = t0 + b * INSERT_BATCH / self.rate
            while (wait := due - time.perf_counter()) > 0:
                if self.stop_event.wait(min(wait, 0.05)):
                    return
            if self.stop_event.is_set():
                return
            e = min(s + INSERT_BATCH, self.stop_row)
            try:
                dt = self.client.insert(self.vectors[s:e], self.meta.slice(s, e - s), list(range(s, e)))
                self.batch_ms.append(dt * 1000.0)
                self.inserted += e - s
                if time.perf_counter() <= self.loop_end:
                    self.inserted_during_loop = self.inserted
            except Exception as ex:  # noqa: BLE001 - counted, reported in extra.insert_errors
                self.errors += 1
                print(f"loadbench: insert error: {type(ex).__name__}: {ex}", file=sys.stderr)


def load_loop(clients, queries: np.ndarray, k: int, params: dict, duration: float, on_start=None) -> dict:
    """C workers, closed loop for `duration` seconds. Returns one search run (CONTRACT section 12.1)."""
    q, c = len(queries), len(clients)
    lat = [[] for _ in range(c)]
    errs = [0] * c
    first_ids: list = [None] * q
    first_scores: list = [None] * q
    barrier = threading.Barrier(c + 1)
    deadline = [0.0]

    def worker(w: int) -> None:
        client, my_lat = clients[w], lat[w]
        i, done = (w * q) // c, 0
        barrier.wait()
        while True:
            now = time.perf_counter()
            first_pass = w == 0 and done < q
            if now >= deadline[0] and not first_pass:
                break
            t0 = time.perf_counter()
            try:
                row_ids, row_scores = client.search(queries[i], k, params)
                my_lat.append((time.perf_counter() - t0) * 1000.0)
                ok = True
            except Exception:  # noqa: BLE001 - a failed query counts as an error
                errs[w] += 1
                ok = False
            if first_pass:
                if ok:
                    row_ids, row_scores = base.pad(row_ids, row_scores, k)
                    first_ids[i] = [int(x) for x in row_ids]
                    first_scores[i] = [None if s is None or math.isinf(s) else s for s in row_scores]
                else:
                    first_ids[i], first_scores[i] = [-1] * k, [None] * k
            done += 1
            i = (i + 1) % q

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(c)]
    for t in threads:
        t.start()
    cpu0 = cpu_seconds()
    t_start = time.perf_counter()
    deadline[0] = t_start + duration
    if on_start:
        on_start()
    barrier.wait()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    cpu = cpu_seconds() - cpu0
    latency = [x for part in lat for x in part]
    return {
        "search_params": params, "ids": first_ids, "scores": first_scores, "latency_ms": latency,
        "total_s": wall, "qps": len(latency) / wall, "distance_computations": None,
        "extra": {"errors": sum(errs), "cpu_pct": 100.0 * cpu / wall, "clients": c, "duration_s": duration,
                  "queries_done": len(latency)},
    }


def run(args) -> dict:
    if args.db not in base.DATABASES:
        raise UsageError(f"unknown database {args.db!r}; known: {base.DATABASES}")
    if args.index not in base.BUILD_DEFAULTS:
        raise UsageError(f"unknown index {args.index!r}; known: {sorted(base.BUILD_DEFAULTS)}")
    if args.index not in base.SUPPORTED[args.db]:
        raise UsageError(f"{args.db} has no {args.index}")
    if args.clients < 1 or args.duration <= 0 or args.insert_rate < 0:
        raise UsageError("want --clients >= 1, --duration > 0, --insert-rate >= 0")
    build_params = parse_pairs(args.build, base.build_defaults(args.db, args.index), "build")
    searches = [parse_pairs([s], base.search_defaults(args.db, args.index), "search") for s in args.search]
    searches = searches or [base.search_defaults(args.db, args.index)]
    if args.insert_rate and len(searches) > 1:
        raise UsageError("--insert-rate takes one --search setting (the inserted rows change the index)")

    vectors = np.load(args.data / "vectors.npy", mmap_mode="r")
    if args.limit:
        vectors = vectors[: args.limit]
    vectors = np.ascontiguousarray(vectors)
    queries = np.load(args.data / "queries.npy")
    meta = pq.read_table(args.data / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, len(vectors))
    n, dim = vectors.shape
    n_build = int(n * 0.9) if args.insert_rate else n

    client = base.get_client(args.db)
    client.data_dir = args.data
    client.connect()
    workers, inserter_client = [], None
    try:
        client.reset()
        load_s = client.load(vectors[:n_build], meta.slice(0, n_build), args.batch)
        build_s = client.build_index(args.index, build_params)
        stats = client.stats()
        for i in range(min(args.warmup, len(queries))):
            client.search(queries[i], args.k, searches[0])
        workers = [open_client(args) for _ in range(args.clients)]
        for wc in workers:
            for i in range(WORKER_WARMUP):
                wc.search(queries[i], args.k, searches[0])

        results, inserter = [], None
        for p in searches:
            start = None
            if args.insert_rate:
                inserter_client = open_client(args)
                stop_event = threading.Event()
                inserter = Inserter(inserter_client, vectors, meta, n_build, n, args.insert_rate, stop_event)
                start = inserter.start
            r = load_loop(workers, queries, args.k, p, args.duration, on_start=start)
            if inserter is not None:
                inserter.loop_end = time.perf_counter()
                t_wait = time.perf_counter()
                inserter.join()  # finish the remaining rows (docstring)
                ins = {"insert_rate": args.insert_rate, "inserted_rows": inserter.inserted,
                       "inserted_during_loop": inserter.inserted_during_loop,
                       "insert_tail_s": time.perf_counter() - t_wait,
                       "insert_p50_ms": statistics.median(inserter.batch_ms) if inserter.batch_ms else None,
                       "insert_errors": inserter.errors}
                r["extra"].update(ins)
            results.append(r)
        if inserter is not None:
            client.finish_inserts()
            after = run_search(client, queries, args.k, {**searches[0], "phase": "after_inserts"})
            after["extra"].update(ins)
            after["extra"]["rows"] = client.stats().get("rows")
            results.append(after)
    finally:
        for wc in workers + ([inserter_client] if inserter_client else []):
            wc.close()
        client.close()

    return {
        "contract_version": 1, "language": args.db, "index": args.index, "data_dir": str(args.data),
        "n": n, "dim": dim, "q": len(queries), "k": args.k, "threads": 0, "seed": args.seed,
        "build_params": build_params,
        "build": {"train_s": 0.0, "add_s": load_s + build_s, "total_s": load_s + build_s,
                  "peak_rss_mb": peak_rss_mb(), "index_bytes": int(stats.pop("index_bytes", 0))},
        "searches": results, "machine": machine(),
        "extra": {"load_s": load_s, "server_build_s": build_s, "build_rows": n_build, "clients": args.clients,
                  "duration_s": args.duration, "insert_rate": args.insert_rate, **stats},
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
        print(f"loadbench: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - any failure is exit 1 with the message
        print(f"loadbench: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
