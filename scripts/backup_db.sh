#!/usr/bin/env bash
# Phase 7 (CONTRACT section 15.3): backup and restore of one database, every case of the
# runner's backup mode (flat, ivf, hnsw where the database has them). Runs on the host
# because two steps need Docker: pg_dump / pg_restore live only in the pgvector container, and
# the Milvus cold backup stops and starts the milvus container. No toolchain runs here: every
# program runs in a container; the host only runs docker compose, docker run, and a clock.
#
# Usage (through the Makefile, with the database up: make db-up DB=<db>):
#   scripts/backup_db.sh <qdrant|pgvector|milvus> --data DIR [--indexes flat,ivf,hnsw] [--limit N] [--force]
#
# Per case (tools/backup/bench.py documents the stages):
#   qdrant    one dbbench run, --stage all (the snapshot API does the round trip).
#   pgvector  dbbench --stage before; host: pg_dump -Fc -t items to /tmp in the pgvector
#             container, DROP TABLE items, pg_restore (builds the vector index again), delete the
#             dump; dbbench --stage after --host-json '{...}'.
#   milvus    dbbench --stage before (flushes); host: stop milvus, tar the milvus-data volume to
#             <out stem>.tar on the raw volume, start, wait healthy (backup); stop, wipe the
#             volume, untar, start, wait healthy (restore); delete the .tar; dbbench --stage after.
# The host times each step with Perl's Time::HiRes clock (macOS date has no sub-second format).
# --limit N writes to results/raw/<data>/bak-test/ (the tests; the report skips that folder).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$REPO_ROOT" | tr '[:upper:]' '[:lower:]')}"
VOL_RAW="${PROJECT}_raw"
VOL_MILVUS="${PROJECT}_milvus-data"

DB="${1:?usage: scripts/backup_db.sh <db> --data DIR [--indexes LIST] [--limit N] [--force]}"
shift
DATA="data/processed/dev"; INDEXES="flat,ivf,hnsw"; LIMIT=""; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --indexes) INDEXES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "backup_db.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

DBRUN=(docker compose --profile db run --rm -T dbbench uv run --frozen python -m)
LIMIT_ARGS=(); [ -n "$LIMIT" ] && LIMIT_ARGS=(--limit "$LIMIT")
now() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }
sub() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", a - b }'; }
# A throwaway container on the two volumes, no network, inside the day caps.
voltool() { docker run --rm --network none --cpus 3 --memory 4g -v "$VOL_MILVUS:/v" -v "$VOL_RAW:/raw" vro-bench:latest sh -c "$1"; }
milvus_up() { docker compose --profile milvus up -d --wait milvus >/dev/null; }
milvus_stop() { docker compose --profile milvus stop -t 120 milvus >/dev/null; }
pgx() { docker compose --profile pgvector exec -T pgvector "$@"; }

CASES="$("${DBRUN[@]}" tools.bench.runner --backup --list --languages "$DB" --indexes "$INDEXES" --data "$DATA" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"})"
[ -n "$CASES" ] || { echo "backup_db.sh: no cases for $DB ($INDEXES)"; exit 0; }

status=0
while read -r INDEX OUT EXISTS; do
  if [ "$EXISTS" = 1 ] && [ "$FORCE" = 0 ]; then echo "$(basename "$OUT"): exists, skipped (use --force)"; continue; fi
  echo "== $DB $INDEX -> $OUT"
  BENCH=(tools.backup.bench --db "$DB" --index "$INDEX" --data "$DATA" --out "$OUT" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"})
  case "$DB" in
    qdrant)
      "${DBRUN[@]}" "${BENCH[@]}" --stage all </dev/null || status=1 ;;
    pgvector)
      "${DBRUN[@]}" "${BENCH[@]}" --stage before </dev/null || { status=1; continue; }
      DUMP=/tmp/vro-backup.dump
      t0=$(now); pgx pg_dump -Fc -U vro -d vro -t items -f "$DUMP" </dev/null; t1=$(now)
      BYTES=$(pgx stat -c %s "$DUMP" </dev/null | tr -d '\r')
      pgx psql -q -U vro -d vro -c 'DROP TABLE items' </dev/null; t2=$(now)
      pgx pg_restore -U vro -d vro "$DUMP" </dev/null; t3=$(now)
      pgx rm -f "$DUMP" </dev/null
      REBUILD=true; [ "$INDEX" = flat ] && REBUILD=false   # flat has no vector index to build
      HOST="{\"backup_s\": $(sub "$t1" "$t0"), \"backup_bytes\": $BYTES, \"drop_s\": $(sub "$t2" "$t1"), \"restore_s\": $(sub "$t3" "$t2"), \"rebuild_needed\": $REBUILD, \"cold\": false, \"method\": \"pg_dump -Fc -t items / DROP TABLE / pg_restore, run in the pgvector container by docker compose exec\"}"
      echo "host step: $HOST"
      "${DBRUN[@]}" "${BENCH[@]}" --stage after --host-json "$HOST" </dev/null || status=1 ;;
    milvus)
      "${DBRUN[@]}" "${BENCH[@]}" --stage before </dev/null || { status=1; continue; }
      REL="${OUT#results/raw/}"; TAR="/raw/${REL%.json}.tar"
      t0=$(now); milvus_stop; t1=$(now)
      voltool "tar -C /v -cf '$TAR' ." </dev/null; t2=$(now)
      milvus_up; t3=$(now)
      BYTES=$(voltool "stat -c %s '$TAR'" </dev/null)
      t4=$(now); milvus_stop; t5=$(now)
      voltool "find /v -mindepth 1 -delete && tar -C /v -xf '$TAR'" </dev/null; t6=$(now)
      milvus_up; t7=$(now)
      voltool "rm -f '$TAR'" </dev/null
      HOST="{\"backup_s\": $(sub "$t3" "$t0"), \"backup_bytes\": $BYTES, \"backup_stop_s\": $(sub "$t1" "$t0"), \"backup_tar_s\": $(sub "$t2" "$t1"), \"backup_start_s\": $(sub "$t3" "$t2"), \"restore_s\": $(sub "$t7" "$t4"), \"stop_s\": $(sub "$t5" "$t4"), \"wipe_untar_s\": $(sub "$t6" "$t5"), \"start_s\": $(sub "$t7" "$t6"), \"rebuild_needed\": false, \"cold\": true, \"method\": \"cold: stop milvus, tar the milvus-data volume, start (docker compose and docker run on the host)\"}"
      echo "host step: $HOST"
      "${DBRUN[@]}" "${BENCH[@]}" --stage after --host-json "$HOST" </dev/null || status=1 ;;
    *) echo "backup_db.sh: unknown database $DB" >&2; exit 2 ;;
  esac
done <<< "$CASES"
exit $status
