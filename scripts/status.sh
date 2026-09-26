#!/usr/bin/env bash
# Print the coordinator lock, this project's running job and containers, the
# newest log's progress, and how many result files each data set has.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_ENGINEERING_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LOCK_DIR="$AI_ENGINEERING_ROOT/.coord/heavy.lock"
LOG_DIR="$REPO_ROOT/results/logs"

echo "Vector Retrieval status, $(date)"
echo

echo "Lock:"
if [ -d "$LOCK_DIR" ]; then
  owner="$(cat "$LOCK_DIR/owner" 2>/dev/null || echo unknown)"
  pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "  held: $owner (pid $pid, running)"
  elif [ -n "$pid" ]; then
    echo "  held: $owner (pid $pid, NOT running: stale)"
  else
    echo "  held: $owner"
  fi
else
  echo "  free"
fi

echo
echo "Containers (vro-bench):"
docker ps --filter ancestor=vro-bench:latest --format '  {{.Names}}  {{.Status}}  {{.Command}}' 2>/dev/null | sed 's/$//' | grep . || echo "  none"

echo
echo "Load: $(sysctl -n vm.loadavg 2>/dev/null)   Disk free: $(df -g "$REPO_ROOT" | awk 'NR==2 {print $4}') GB"

echo
newest="$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)"
if [ -n "$newest" ]; then
  echo "Newest log: $newest"
  echo "  cases done: $(grep -c ': exit 0' "$newest")   failed: $(grep -c ': exit [^0]' "$newest")"
  echo "  last line:  $(tail -1 "$newest" | cut -c1-150)"
else
  echo "No job logs yet (results/logs/)"
fi

echo
echo "Result files in the raw volume:"
if docker info >/dev/null 2>&1 && docker image inspect vro-bench:latest >/dev/null 2>&1; then
  (cd "$REPO_ROOT" && docker compose run --rm --no-deps bench sh -c \
    'n=0; for d in results/raw/*/; do [ -d "$d" ] || continue; n=1; printf "  %s %s json\n" "$d" "$(ls "$d"*.json 2>/dev/null | wc -l | tr -d " ")"; done; [ $n = 1 ] || echo "  (no result files yet)"' 2>/dev/null | grep -v '^WARN')
else
  echo "  Docker or the image is not available"
fi
