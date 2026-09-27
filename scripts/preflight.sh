#!/usr/bin/env bash
# Preflight checks before a heavy job (a benchmark sweep) on this machine.
#
# Prints one line per check. Exits 1 on a hard failure: disk under 15 GB,
# the timing lock held by someone else, Docker down, the image missing,
# or the dev data missing. Exits 3 (WAIT) when the only problem is a GPU
# lock (gpu.lock or heavy.lock): run.sh then starts and waits for it to
# clear. Load, power, and lid only warn.
#
# This script only looks. It never starts or stops a container or a job.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_ENGINEERING_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LOCK_DIR="$AI_ENGINEERING_ROOT/.coord/timing.lock"
GPU_LOCKS="$AI_ENGINEERING_ROOT/.coord/gpu.lock $AI_ENGINEERING_ROOT/.coord/heavy.lock"
DISK_FAIL_GB=15
LOAD_WARN=2
HARD_FAIL=0
WAIT=0

ok()   { printf 'OK    %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; HARD_FAIL=1; }
wait_() { printf 'WAIT  %s\n' "$1"; WAIT=1; }

echo "Preflight for Vector Retrieval Optimization, $(date)"
echo "Repo: $REPO_ROOT"
echo

# Load average: a benchmark's latency numbers need an idle machine.
load1="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
if [ -n "$load1" ] && awk -v l="$load1" -v w="$LOAD_WARN" 'BEGIN{exit !(l < w)}'; then
  ok "load average $load1 (under $LOAD_WARN)"
else
  warn "load average ${load1:-unknown} is not under $LOAD_WARN; the runner waits up to 3 min per case, then runs anyway"
fi

# Disk: DiskANN writes 4 KB per row (4.9 GB for the full corpus) into the raw volume.
free_gb="$(df -g "$REPO_ROOT" | awk 'NR==2 {print $4}')"
if [ "${free_gb:-0}" -ge "$DISK_FAIL_GB" ]; then
  ok "disk free ${free_gb} GB"
else
  fail "disk free ${free_gb:-?} GB, under $DISK_FAIL_GB GB"
fi

# Coordinator locks: TIMING runs alone (timing.lock) and needs the GPU lock free (gpu.lock, old name heavy.lock).
if [ -d "$LOCK_DIR" ]; then
  owner="$(cat "$LOCK_DIR/owner" 2>/dev/null || echo unknown)"
  case "$owner" in
    "vector-retrieval "*)
      pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        fail "timing lock held by this project's running job: $owner (pid $pid)"
      else
        warn "stale timing lock from this project: $owner; scripts/run.sh removes it"
      fi
      ;;
    *) fail "timing lock held by another project: $owner" ;;
  esac
else
  ok "no timing lock"
fi
for g in $GPU_LOCKS; do
  if [ -d "$g" ]; then
    wait_ "GPU lock held ($(basename "$g")): $(cat "$g/owner" 2>/dev/null || echo unknown); a TIMING job waits for it (run.sh polls every 5 min)"
  fi
done

# Docker daemon, VM size, image, and no benchmark container already running.
if docker info >/dev/null 2>&1; then
  ok "Docker daemon up"
  mem_gb="$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')"
  if [ "${mem_gb:-0}" -ge 15 ]; then
    ok "Docker VM memory ${mem_gb} GB"
  else
    warn "Docker VM memory ${mem_gb:-?} GB; the full-corpus runs need 16 GB (Docker Desktop > Settings > Resources)"
  fi
  if docker image inspect vro-bench:latest >/dev/null 2>&1; then
    ok "vro-bench:latest image present"
  else
    fail "vro-bench:latest image missing (run: make setup)"
  fi
  running="$(docker ps --filter ancestor=vro-bench:latest --format '{{.Names}}' | tr '\n' ' ')"
  if [ -n "$running" ]; then
    warn "vro-bench container already running: $running"
  else
    ok "no vro-bench container running"
  fi
else
  fail "Docker daemon not reachable"
fi

# Data.
if [ -f "$REPO_ROOT/data/processed/dev/ground_truth.npy" ]; then
  ok "dev data present"
else
  fail "data/processed/dev missing (run tools.data.prepare, ground_truth, subset on the host)"
fi
if [ -f "$REPO_ROOT/data/processed/ground_truth.npy" ]; then
  ok "full data present"
else
  warn "full data missing (needed only for the full-corpus runs)"
fi

# Power and lid: a sweep runs for hours.
if pmset -g batt 2>/dev/null | head -1 | grep -q "AC Power"; then
  ok "on AC power"
else
  warn "not on AC power"
fi
clamshell="$(ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -o '"AppleClamshellState" = [A-Za-z]*' | awk '{print $NF}')"
case "$clamshell" in
  No)  ok "lid open" ;;
  Yes) warn "lid closed; caffeinate -i keeps the job alive only while the lid is open" ;;
  *)   ;;
esac

echo
if [ "$HARD_FAIL" -ne 0 ]; then
  echo "preflight: FAIL"
  exit 1
fi
if [ "$WAIT" -ne 0 ]; then
  echo "preflight: WAIT (GPU lock held)"
  exit 3
fi
echo "preflight: OK"
