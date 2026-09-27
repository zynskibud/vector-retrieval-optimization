#!/usr/bin/env bash
# Run one TIMING job under the coordinator's timing lock, detached, with the lid-sleep
# guard. Every benchmark sweep on this machine starts through this script.
#
# Usage: scripts/run.sh [--dry-run] <job>
#        scripts/run.sh --stop
#
# Jobs (each is one or more `make` targets, which run inside the container):
#   dev-sweep     all six indexes, all languages and FAISS, dev set, 3 repeats
#                 (Python DiskANN 1 repeat: its build takes 30 minutes), then the report
#   full-sweep    same on the full corpus (Python HNSW and DiskANN on the dev set, see
#                 results/summary/OPEN-QUESTIONS.md); needs Docker at 16 GB and mem_limit 12g
#   db-sweep      Phase 2: each database in turn (up, its supported indexes on the dev set, 3 repeats, down)
#   load-sweep    Phase 4 (TIMING): hnsw in python, go, cpp, rust under 1..64 clients and one
#                 insert run (make load), then each database in turn (up, make load-db, down), then the report
#   changes-sweep Phase 5 (TIMING): flat, ivf, hnsw in python, go, cpp, rust with del10/30/50 (with and
#                 without compaction) and upd10 (make changes), then each database in turn (up,
#                 make changes-db, down), then the report
#   cache-sweep   Phase 6 (TIMING): embedding cache, backends none/lru/redis x capacity 500/2000/5000
#                 on zipf, one uniform run per backend, one lru invalidation run (make cache-up,
#                 make cache, make cache-down), then the report
#   backup-sweep  Phase 7 (TIMING): each database in turn (up, make backup-db: flat, ivf, hnsw where it
#                 has them, backup, drop, restore, searches before and after; down), then the report
#   test          make test (light, but it still takes the lock so it never overlaps a sweep)
#
# Steps: preflight (stop on FAIL), take the lock with owner
# "vector-retrieval <job> <ISO time>", start the job detached under
# `caffeinate -i`, release the lock when the job exits or is killed
# (SIGTERM/SIGINT; a SIGKILL leaves a stale lock that preflight reports and
# the next run.sh removes). Output goes to results/logs/<job>-<time>.log.
#
# Only start a TIMING job after the coordinator sends "GO <job>".

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_ENGINEERING_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LOCK_DIR="$AI_ENGINEERING_ROOT/.coord/timing.lock"
LOG_DIR="$REPO_ROOT/results/logs"
DRY_RUN=0
JOB=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --stop)    JOB="--stop" ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *)         JOB="$arg" ;;
  esac
done

# --stop: kill this project's running job; its trap releases the lock.
if [ "$JOB" = "--stop" ]; then
  pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "no running job of this project"; exit 0
  fi
  echo "stopping job pid $pid ($(cat "$LOCK_DIR/owner" 2>/dev/null))"
  kill -TERM "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  docker ps -q --filter ancestor=vro-bench:latest | xargs -r docker stop >/dev/null 2>&1
  [ -d "$LOCK_DIR" ] && rm -rf "$LOCK_DIR" && echo "lock released"
  exit 0
fi

case "$JOB" in
  dev-sweep)
    CMD="make bench ARGS='--data data/processed/dev --languages rust,cpp,go,faiss --indexes flat,ivf,pq,hnsw,ivf_pq,diskann --repeat 3'
&& make bench ARGS='--data data/processed/dev --languages python --indexes flat,ivf,pq,hnsw,ivf_pq --repeat 3'
&& make bench ARGS='--data data/processed/dev --languages python --indexes diskann --repeat 1'
&& make report ARGS='--data data/processed/dev'" ;;
  full-sweep)
    CMD="make bench ARGS='--data data/processed --languages rust,cpp,go,faiss --indexes flat,ivf,pq,hnsw,ivf_pq,diskann --repeat 3'
&& make bench ARGS='--data data/processed --languages python --indexes flat,ivf,pq,ivf_pq --repeat 3'
&& make report ARGS='--data data/processed'" ;;
  db-sweep)
    CMD="for db in qdrant pgvector milvus; do make db-up DB=\$db && make bench-db ARGS=\"--data data/processed/dev --languages \$db --indexes flat,ivf,pq,ivf_pq,hnsw,diskann --repeat 3\"; make db-down DB=\$db; done
&& make report ARGS='--data data/processed/dev'" ;;
  load-sweep)
    CMD="make load ARGS='--data data/processed/dev --languages rust,cpp,go,python'
&& for db in qdrant pgvector milvus; do make db-up DB=\$db && make load-db ARGS=\"--data data/processed/dev --languages \$db\"; make db-down DB=\$db; done
&& make report ARGS='--data data/processed/dev'" ;;
  changes-sweep)
    CMD="make changes ARGS='--data data/processed/dev --languages rust,cpp,go,python'
&& for db in qdrant pgvector milvus; do make db-up DB=\$db && make changes-db ARGS=\"--data data/processed/dev --languages \$db\"; make db-down DB=\$db; done
&& make report ARGS='--data data/processed/dev'" ;;
  cache-sweep)
    CMD="make cache-up && make cache ARGS='--data data/processed/dev'; make cache-down
&& make report ARGS='--data data/processed/dev'" ;;
  backup-sweep)
    CMD="for db in qdrant pgvector milvus; do make db-up DB=\$db && make backup-db DB=\$db ARGS='--data data/processed/dev'; make db-down DB=\$db; done
&& make report ARGS='--data data/processed/dev'" ;;
  test) CMD="make test" ;;
  "")   echo "usage: scripts/run.sh [--dry-run] <dev-sweep|full-sweep|db-sweep|load-sweep|changes-sweep|cache-sweep|backup-sweep|test> | --stop" >&2; exit 2 ;;
  *)    echo "unknown job: $JOB" >&2; exit 2 ;;
esac
CMD="$(printf '%s' "$CMD" | tr '\n' ' ')"

echo "job: $JOB"
echo "command: $CMD"
echo

# Preflight. Stop on a hard failure; do not touch the lock.
if ! "$REPO_ROOT/scripts/preflight.sh"; then
  echo "run.sh: preflight failed, not starting" >&2
  exit 1
fi
echo

# Remove a stale lock of our own (owner written by this script, pid gone).
if [ -d "$LOCK_DIR" ] && grep -q '^vector-retrieval ' "$LOCK_DIR/owner" 2>/dev/null; then
  pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "removing stale lock: $(cat "$LOCK_DIR/owner")"
    rm -rf "$LOCK_DIR"
  fi
fi

# Take the lock. mkdir is atomic: if it fails, someone else holds it.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "run.sh: lock held by: $(cat "$LOCK_DIR/owner" 2>/dev/null || echo unknown). Not starting." >&2
  exit 1
fi
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "vector-retrieval $JOB $START" > "$LOCK_DIR/owner"

if [ "$DRY_RUN" -eq 1 ]; then
  caps="day caps (3 CPUs, 6 GB)"; [ "$JOB" != "test" ] && caps="TIMING caps (VRO_CPUS=6 VRO_MEM=12g VRO_DB_CPUS=4)"
  echo "dry run: took and released the lock; would run with $caps: $CMD"
  rm -rf "$LOCK_DIR"
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$JOB-$(date +%Y%m%d-%H%M%S).log"

# The TIMING caps (PROTOCOL.md): 6 CPUs and 12 GB for the bench container, 4 CPUs for a
# database. docker-compose.yml defaults to the day caps (3 CPUs, 6 GB, 3 CPUs) when these
# are unset, and only this script sets them. The test job keeps the day caps.
if [ "$JOB" != "test" ]; then
  export VRO_CPUS=6 VRO_MEM=12g VRO_DB_CPUS=4
fi

# The detached wrapper owns the lock: it releases it on exit, TERM, or INT,
# and stops the container it started so a killed job does not keep running.
nohup caffeinate -i bash -c '
  trap "docker ps -q --filter ancestor=vro-bench:latest | xargs -r docker stop >/dev/null 2>&1; rm -rf \"$0\"; echo \"[run.sh] lock released $(date -u +%FT%TZ)\"" EXIT
  trap "exit 143" TERM INT
  cd "$1" && echo "[run.sh] start $(date -u +%FT%TZ) job=$2" && eval "$3"
  echo "[run.sh] end $(date -u +%FT%TZ) exit=$?"
' "$LOCK_DIR" "$REPO_ROOT" "$JOB" "$CMD" >"$LOG" 2>&1 &
echo $! > "$LOCK_DIR/pid"

echo "started $JOB, pid $(cat "$LOCK_DIR/pid"), lock $LOCK_DIR"
echo "log: $LOG"
echo "status: scripts/status.sh    stop: scripts/run.sh --stop"
