"""Backup and restore of one index in one database (Phase 7, CONTRACT section 15.3).

Steps: load and build, search run 1 (the 1,000 queries), back up, drop, restore, search run 2
(search_params.phase = "after_restore"), compare. Output: CONTRACT section 3 JSON, the same shape
as tools/db/bench.py, with extra.backup_s, backup_bytes, restore_s, rebuild_needed, rows_before,
rows_after, restore_identical (IDs of run 1 == IDs of run 2, and rows_before == rows_after),
ids_equal_fraction (the fraction of the 1,000 queries whose k IDs are equal, in order), and
ids_overlap (the mean over queries of |IDs of run 1 & IDs of run 2| / k). A database that builds
the index again on restore (pgvector) with unseeded randomness cannot give identical IDs;
ids_overlap then measures how close the rebuilt index is.

Stages. This program runs inside the dbbench container, which has no Docker socket. Qdrant needs
nothing else: its snapshot API does the whole round trip over the internal network. pg_dump
lives only in the pgvector container, and the Milvus cold backup stops and starts a container,
so for those two databases the host script scripts/backup_db.sh runs the backup, drop, and
restore between two stages of this program:

  --stage all     (Qdrant) every step in one process. The snapshot file is <out>.snapshot
                  on the raw volume; it is deleted at the end (its size is in extra).
  --stage before  load, build, search run 1, stats; write the state file <out>.stage1.json on
                  the raw volume. Milvus: flush first, so every row is on disk before the stop.
  (host step)     scripts/backup_db.sh: pgvector = pg_dump -Fc -t items inside the pgvector
                  container, DROP TABLE, pg_restore; milvus = stop, tar the milvus-data volume
                  to <out stem>.tar on the raw volume, start, wait healthy; then the restore:
                  stop, wipe the volume (this is the drop), untar, start, wait healthy.
                  The .tar and the dump are deleted after the restore (their size is in extra).
                  The script times each step with the shell clock and passes the numbers to:
  --stage after --host-json '{"backup_s": ..., "backup_bytes": ..., "restore_s": ..., ...}'
                  reconnect (Milvus: load the collection; its seconds are added to restore_s),
                  search run 2, compare with the state file, write <out>, delete the state file.

Run (the Makefile target starts the stages): make backup-db DB=qdrant ARGS="--data data/processed/dev"
One case by hand, Qdrant only: docker compose --profile db run --rm dbbench uv run --frozen \\
  python -m tools.backup.bench --db qdrant --index hnsw --data data/processed/dev --out results/raw/dev/bak-qdrant-hnsw.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.bench.schema import validate
from tools.db import base
from tools.db.bench import UsageError, machine, parse_pairs, peak_rss_mb, run_search

INDEXES = ("flat", "ivf", "hnsw")
SINGLE_STAGE = {"qdrant"}  # databases whose whole round trip runs from dbbench


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="backup-bench")
    ap.add_argument("--db", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stage", choices=("all", "before", "after"), default="all")
    ap.add_argument("--host-json", default=None, help="stage after: the host step's timings as JSON")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build", action="append", default=[])
    ap.add_argument("--search", default=None, help="one search setting; default: the contract default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=None, help="accepted and ignored: the server chooses")
    return ap.parse_args(argv)


def state_path(out: Path) -> Path:
    return out.with_name(out.stem + ".stage1.json")


def check(args) -> tuple[dict, dict]:
    if args.db not in base.DATABASES:
        raise UsageError(f"unknown database {args.db!r}; known: {base.DATABASES}")
    if args.index not in INDEXES or args.index not in base.SUPPORTED[args.db]:
        raise UsageError(f"{args.db} backup: index must be one of {sorted(set(INDEXES) & base.SUPPORTED[args.db])}")
    if args.stage == "all" and args.db not in SINGLE_STAGE:
        raise UsageError(f"{args.db} needs a host step: run it with make backup-db DB={args.db} (scripts/backup_db.sh)")
    if args.stage == "after" and not args.host_json:
        raise UsageError("--stage after needs --host-json")
    build_params = parse_pairs(args.build, base.build_defaults(args.db, args.index), "build")
    search = parse_pairs([args.search] if args.search else [], base.search_defaults(args.db, args.index), "search")
    return build_params, search


def warm(client, queries, k, params, n):
    for i in range(min(n, len(queries))):
        client.search(queries[i], k, params)


def stage_before(args, client, build_params: dict, search: dict, queries) -> dict:
    """Load, build, search run 1. Returns the partial document (the state)."""
    vectors = np.load(args.data / "vectors.npy", mmap_mode="r")
    if args.limit:
        vectors = vectors[: args.limit]
    vectors = np.ascontiguousarray(vectors)
    meta = pq.read_table(args.data / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, len(vectors))
    n, dim = vectors.shape
    client.reset()
    load_s = client.load(vectors, meta, args.batch)
    build_s = client.build_index(args.index, build_params)
    stats = client.stats()
    warm(client, queries, args.k, search, args.warmup)
    run1 = run_search(client, queries, args.k, search)
    if hasattr(client, "flush_all"):
        client.flush_all()
    return {
        "contract_version": 1, "language": args.db, "index": args.index, "data_dir": str(args.data),
        "n": n, "dim": dim, "q": len(queries), "k": args.k, "threads": 0, "seed": args.seed,
        "build_params": build_params,
        "build": {"train_s": 0.0, "add_s": load_s + build_s, "total_s": load_s + build_s,
                  "peak_rss_mb": peak_rss_mb(), "index_bytes": int(stats.pop("index_bytes", 0))},
        "searches": [run1], "machine": machine(),
        "extra": {"load_s": load_s, "server_build_s": build_s, **stats, "rows_before": int(stats.get("rows", 0))},
    }


def stage_after(args, client, doc: dict, search: dict, queries, backup: dict, restore: dict) -> dict:
    """Search run 2 on the restored data; merge the timings; compare."""
    stats = client.stats()
    stats.pop("index_bytes", None)
    warm(client, queries, args.k, search, args.warmup)
    run2 = run_search(client, queries, args.k, search)
    run2["search_params"] = {**run2["search_params"], "phase": "after_restore"}
    run1 = doc["searches"][0]
    same = [a == b for a, b in zip(run1["ids"], run2["ids"])]
    rows_before, rows_after = doc["extra"]["rows_before"], int(stats.get("rows", 0))
    doc["searches"].append(run2)
    doc["build"]["peak_rss_mb"] = max(doc["build"]["peak_rss_mb"], peak_rss_mb())
    doc["extra"].update({
        "backup_s": float(backup["backup_s"]), "backup_bytes": int(backup["backup_bytes"]),
        "restore_s": float(restore["restore_s"]), "rebuild_needed": bool(restore["rebuild_needed"]),
        "cold": bool(restore.get("cold", False)),
        "rows_before": rows_before, "rows_after": rows_after,
        "restore_identical": bool(all(same) and rows_before == rows_after),
        "ids_equal_fraction": sum(same) / len(same),
        "ids_overlap": sum(len(set(a) & set(b)) / len(a) for a, b in zip(run1["ids"], run2["ids"])) / len(same),
        "backup_detail": {k: v for k, v in backup.items() if k not in ("backup_s", "backup_bytes")},
        "restore_detail": {k: v for k, v in restore.items() if k not in ("restore_s", "rebuild_needed", "cold")},
        "stats_after_restore": stats,
    })
    return doc


def run(args) -> dict | None:
    build_params, search = check(args)
    queries = np.load(args.data / "queries.npy")
    client = base.get_client(args.db)
    client.data_dir = args.data
    client.connect()
    try:
        if args.stage == "before":
            doc = stage_before(args, client, build_params, search, queries)
            state_path(args.out).parent.mkdir(parents=True, exist_ok=True)
            state_path(args.out).write_text(json.dumps(doc))
            return None
        if args.stage == "after":
            doc = json.loads(state_path(args.out).read_text())
            host = json.loads(args.host_json)
            reopen_s = client.reopen(args.index)
            backup = {k: host[k] for k in host if k.startswith("backup")} | {"method": host.get("method", "")}
            restore = {k: host[k] for k in host if not k.startswith("backup") and k != "method"}
            restore["restore_s"] = float(host["restore_s"]) + reopen_s
            restore["reopen_s"] = reopen_s
            doc = stage_after(args, client, doc, search, queries, backup, restore)
            state_path(args.out).unlink()
            return doc
        # stage all (Qdrant): the client does every step.
        doc = stage_before(args, client, build_params, search, queries)
        snap = args.out.with_name(args.out.stem + ".snapshot")
        backup = client.backup(snap)
        client.drop()
        restore = client.restore(snap)
        restore["reopen_s"] = client.reopen(args.index)
        doc = stage_after(args, client, doc, search, queries, backup, restore)
        snap.unlink()
        return doc
    finally:
        client.close()


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    t0 = time.perf_counter()
    try:
        doc = run(args)
        if doc is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(doc))
            ex = doc["extra"]
            print(f"{args.out.name}: restore_identical={ex['restore_identical']} rows {ex['rows_before']} -> {ex['rows_after']} "
                  f"ids_equal={ex['ids_equal_fraction']:.4f} overlap={ex['ids_overlap']:.4f} backup_s={ex['backup_s']:.2f} backup_bytes={ex['backup_bytes']} "
                  f"restore_s={ex['restore_s']:.2f} rebuild_needed={ex['rebuild_needed']} ({time.perf_counter() - t0:.0f}s)")
            for e in validate(doc):
                print(f"  schema: {e}")
    except UsageError as e:
        print(f"backup-bench: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - any failure is exit 1 with the message
        print(f"backup-bench: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
