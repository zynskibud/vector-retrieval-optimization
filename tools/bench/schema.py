"""Validate one bench output JSON against indexes/CONTRACT.md section 3.

Run: uv run python -m tools.bench.schema FILE [FILE ...]
"""

import argparse
import json
import sys
from pathlib import Path

LANGUAGES = {"python", "go", "cpp", "rust", "faiss", "qdrant", "pgvector", "milvus", "cache"}  # faiss = the reference (tools/bench/faiss_ref.py); cache = Phase 6 (tools/cache/bench.py)
INDEXES = {"flat", "ivf", "pq", "ivf_pq", "hnsw", "diskann"}
TOP_KEYS = {
    "contract_version": int, "language": str, "index": str, "data_dir": str, "n": int, "dim": int,
    "q": int, "k": int, "threads": int, "seed": int, "build_params": dict, "build": dict,
    "searches": list, "machine": dict, "extra": dict,
}
BUILD_KEYS = {"train_s": float, "add_s": float, "total_s": float, "peak_rss_mb": float, "index_bytes": int}
SEARCH_KEYS = {"search_params": dict, "ids": list, "scores": list, "latency_ms": list, "total_s": float, "qps": float, "extra": dict}
MACHINE_KEYS = {"os": str, "arch": str, "cpu": str, "cores": int}
# A load run (CONTRACT section 12.1) has extra.clients; it then needs these keys in extra.
LOAD_KEYS = {"errors": int, "cpu_pct": float, "clients": int, "duration_s": float, "queries_done": int}
PHASES = {"after_inserts", "after_restore"}  # search_params.phase (CONTRACT sections 12.2 and 15.3)
# A cache run (language "cache", CONTRACT section 14.3) needs these keys in search_params and extra.
# Its latency_ms has one value per request (extra.requests), not q.
CACHE_PARAMS = {"backend": str, "capacity": int, "workload": str, "model_version": str}
CACHE_KEYS = {"requests": int, "hit_rate": float, "embed_p50_ms": float, "search_p50_ms": float, "e2e_p50_ms": float,
              "e2e_p99_ms": float, "entries": int, "cache_bytes": int, "evictions": int, "request_pool_ids": list}


def _is(value, typ) -> bool:
    """Type check where bool is never an int and an int is a valid float."""
    if isinstance(value, bool):
        return typ is bool
    if typ is float:
        return isinstance(value, (int, float))
    return isinstance(value, typ)


def _check_keys(obj: dict, spec: dict, where: str) -> list[str]:
    errors = []
    for key, typ in spec.items():
        if key not in obj:
            errors.append(f"{where}: missing key '{key}'")
        elif not _is(obj[key], typ):
            errors.append(f"{where}.{key}: expected {typ.__name__}, got {type(obj[key]).__name__}")
    return errors


def _check_matrix(rows, q: int, k: int, where: str, cell_ok, cell_name: str) -> list[str]:
    if not isinstance(rows, list) or len(rows) != q:
        got = len(rows) if isinstance(rows, list) else type(rows).__name__
        return [f"{where}: expected {q} rows, got {got}"]
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != k:
            got = len(row) if isinstance(row, list) else type(row).__name__
            return [f"{where}[{i}]: expected {k} values, got {got}"]
        for j, v in enumerate(row):
            if not cell_ok(v):
                return [f"{where}[{i}][{j}]: expected {cell_name}, got {v!r}"]
    return []


def validate(doc) -> list[str]:
    """Return a list of error strings. An empty list means the document is valid."""
    if not isinstance(doc, dict):
        return [f"top level: expected object, got {type(doc).__name__}"]
    errors = _check_keys(doc, TOP_KEYS, "top level")
    extra_top = set(doc) - set(TOP_KEYS)
    if extra_top:
        errors.append(f"top level: unknown keys {sorted(extra_top)} (extra keys go under 'extra')")
    if doc.get("contract_version") != 1:
        errors.append(f"contract_version: expected 1, got {doc.get('contract_version')!r}")
    if "language" in doc and doc["language"] not in LANGUAGES:
        errors.append(f"language: expected one of {sorted(LANGUAGES)}, got {doc['language']!r}")
    if "index" in doc and doc["index"] not in INDEXES:
        errors.append(f"index: expected one of {sorted(INDEXES)}, got {doc['index']!r}")
    for key in ("n", "dim", "q", "k"):
        if _is(doc.get(key), int) and doc[key] <= 0:
            errors.append(f"{key}: expected > 0, got {doc[key]}")
    if isinstance(doc.get("build"), dict):
        errors += _check_keys(doc["build"], BUILD_KEYS, "build")
    if isinstance(doc.get("machine"), dict):
        errors += _check_keys(doc["machine"], MACHINE_KEYS, "machine")
    q, k, n = doc.get("q"), doc.get("k"), doc.get("n")
    if not (_is(q, int) and _is(k, int) and _is(n, int)):
        return errors + ["searches: not checked, because n, q, or k is missing or invalid"]
    searches = doc.get("searches")
    if isinstance(searches, list) and not searches:
        errors.append("searches: expected at least one search run, got an empty list")
    for s, run in enumerate(searches if isinstance(searches, list) else []):
        where = f"searches[{s}]"
        if not isinstance(run, dict):
            errors.append(f"{where}: expected object, got {type(run).__name__}")
            continue
        errors += _check_keys(run, SEARCH_KEYS, where)
        if "distance_computations" not in run:
            errors.append(f"{where}: missing key 'distance_computations' (use null if not counted)")
        elif run["distance_computations"] is not None and not _is(run["distance_computations"], float):
            errors.append(f"{where}.distance_computations: expected number or null, got {run['distance_computations']!r}")
        errors += _check_matrix(run.get("ids"), q, k, f"{where}.ids",
                                lambda v: _is(v, int) and -1 <= v < n, f"int in [-1, {n})")
        errors += _check_matrix(run.get("scores"), q, k, f"{where}.scores",
                                lambda v: v is None or _is(v, float), "float or null")
        extra = run.get("extra") if isinstance(run.get("extra"), dict) else {}
        is_load = "clients" in extra
        is_cache = doc.get("language") == "cache"
        if is_cache:
            errors += _check_keys(extra, CACHE_KEYS, f"{where}.extra")
            if isinstance(run.get("search_params"), dict):
                errors += _check_keys(run["search_params"], CACHE_PARAMS, f"{where}.search_params")
            if isinstance(extra.get("request_pool_ids"), list) and len(extra["request_pool_ids"]) != q:
                errors.append(f"{where}.extra.request_pool_ids: expected {q} values, got {len(extra['request_pool_ids'])}")
            if "invalidate_at" in (run.get("search_params") or {}) and "hit_rate_after_invalidate" not in extra:
                errors.append(f"{where}.extra: missing key 'hit_rate_after_invalidate' (search_params has invalidate_at)")
        if is_load:
            errors += _check_keys(extra, LOAD_KEYS, f"{where}.extra")
            if _is(extra.get("clients"), int) and extra["clients"] < 1:
                errors.append(f"{where}.extra.clients: expected >= 1, got {extra['clients']}")
        sp = run.get("search_params")
        if isinstance(sp, dict) and "phase" in sp and sp["phase"] not in PHASES:
            errors.append(f"{where}.search_params.phase: expected one of {sorted(PHASES)}, got {sp['phase']!r}")
        lat = run.get("latency_ms")
        if isinstance(lat, list):
            if is_load:
                # All latencies of all workers: any count; it must match queries_done.
                if not lat:
                    errors.append(f"{where}.latency_ms: a load run needs at least one value")
                elif _is(extra.get("queries_done"), int) and len(lat) != extra["queries_done"]:
                    errors.append(f"{where}.latency_ms: {len(lat)} values, but extra.queries_done = {extra['queries_done']}")
                elif not all(_is(v, float) and v >= 0 for v in lat):
                    errors.append(f"{where}.latency_ms: every value must be a number >= 0")
            elif is_cache:
                if _is(extra.get("requests"), int) and len(lat) != extra["requests"]:
                    errors.append(f"{where}.latency_ms: {len(lat)} values, but extra.requests = {extra['requests']}")
                elif not all(_is(v, float) and v >= 0 for v in lat):
                    errors.append(f"{where}.latency_ms: every value must be a number >= 0")
            elif len(lat) != q:
                errors.append(f"{where}.latency_ms: expected {q} values, got {len(lat)}")
            elif not all(_is(v, float) and v >= 0 for v in lat):
                errors.append(f"{where}.latency_ms: every value must be a number >= 0")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    failed = False
    for path in ap.parse_args().files:
        try:
            errors = validate(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            errors = [f"cannot read: {e}"]
        failed |= bool(errors)
        print(f"{path}: {'OK' if not errors else 'INVALID'}")
        for e in errors:
            print(f"  {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
