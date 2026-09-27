"""Run benchmark cases one at a time, each as a subprocess of a language's bench program.

Cases never run in parallel: latency numbers depend on an idle CPU.
Outputs: results/raw/<data-name>/<language>-<index>-<hash of build params>.json

Run: uv run python -m tools.bench.runner --data data/processed/dev --languages faiss --indexes flat,ivf,hnsw [--repeat 3] [--dry-run]

Load mode (Phase 4, CONTRACT section 12): --load sweeps --clients over LOAD_CLIENTS at the default
search setting, plus one run with --clients 8 --insert-rate 1000, --duration 20, repeat 1.
Languages run hnsw only; databases run all their supported indexes through tools.load.bench.
Outputs: results/raw/<data-name>/load-<system>-<index>-c<C>[-ins<R>].json
  uv run python -m tools.bench.runner --load --data data/processed/dev --languages rust,cpp,go,python
  uv run python -m tools.bench.runner --load --data data/processed/dev --languages qdrant   (in dbbench)

Changes mode (Phase 5, CONTRACT section 13): --changes runs, per (system, index), --delete del10,
del30, del50 (each without and with --compact) and --update upd10, at the default build and
search setting, repeat 1. Languages run flat, ivf, hnsw; databases every supported index.
Outputs: results/raw/<data-name>/chg-<system>-<index>-<change>[-compact].json
  make changes    ARGS="--data data/processed/dev --languages rust,cpp,go,python"
  make changes-db ARGS="--data data/processed/dev --languages qdrant"   (after make db-up DB=qdrant)

Cache mode (Phase 6, CONTRACT section 14.3): --cache runs tools.cache.bench for backend none
(once), lru and redis at capacity 500, 2000, 5000 on the zipf workload, one uniform run per
backend (capacity 2000), and one lru run with --invalidate-at 25000. Repeat 1. Runs inside
dbbench so Redis is reachable; --backends none,lru skips Redis.
Outputs: results/raw/<data-name>/cache-<backend>-c<capacity>-<workload>[-inv].json
  make cache ARGS="--data data/processed/dev"   (after make cache-up)

Backup mode (Phase 7, CONTRACT section 15.3): --backup, per database, flat, ivf, hnsw where the
database has them, one run each of tools.backup.bench (repeat 1). Qdrant runs here (inside
dbbench). pgvector and Milvus need a host step between two stages, so their cases run through
scripts/backup_db.sh (make backup-db); this mode only lists them for that script (--list prints
one "<index> <out> <exists 0|1>" line per case). With --limit N the outputs go to
results/raw/<data-name>/bak-test/ so that the report does not read them.
Outputs: results/raw/<data-name>/bak-<db>-<index>.json
  make backup-db DB=qdrant ARGS="--data data/processed/dev"   (after make db-up DB=qdrant)
"""

import argparse
import hashlib
import itertools
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.bench.schema import validate

PROGRAMS = {
    "cache": ["uv", "run", "python", "-m", "tools.cache.bench"],  # Phase 6: only through --cache
    "python": ["uv", "run", "python", "-m", "indexes.python.bench"],
    "go": ["indexes/go/bin/bench"],
    "cpp": ["indexes/cpp/build/bench"],
    "rust": ["indexes/rust/target/release/bench"],
    "faiss": ["uv", "run", "python", "-m", "tools.bench.faiss_ref"],
    # Phase 2 databases: run inside the dbbench container (make dbbench / make bench-db).
    "qdrant": ["uv", "run", "python", "-m", "tools.db.bench", "--db", "qdrant"],
    "pgvector": ["uv", "run", "python", "-m", "tools.db.bench", "--db", "pgvector"],
    "milvus": ["uv", "run", "python", "-m", "tools.db.bench", "--db", "milvus"],
}
PYTHON_MODULES = {"python": Path("indexes/python/bench.py"), "faiss": Path("tools/bench/faiss_ref.py"),
                  "qdrant": Path("tools/db/qdrant.py"), "pgvector": Path("tools/db/pgvector.py"),
                  "milvus": Path("tools/db/milvus.py"), "cache": Path("tools/cache/bench.py")}
# (language, index) pairs that do not exist; the runner skips them.
UNSUPPORTED = {("faiss", "diskann"), ("qdrant", "ivf"), ("qdrant", "ivf_pq"), ("qdrant", "diskann"),
               ("pgvector", "pq"), ("pgvector", "ivf_pq"), ("pgvector", "diskann"), ("milvus", "pq")}

# Per index: build variants (each value list is swept, one build per combination)
# and the search sweep (every combination is one --search inside the same build).
# A "search" value can also be a list of such dicts: each is gridded on its own and the
# grids are concatenated. Phase 3 (CONTRACT section 11) uses that for flat, ivf, hnsw:
# the unfiltered sweep as before (no filter key, which means filter=none), then the four
# filters at the default search setting only. So a filter adds 4 searches per build, not
# 4 x the sweep.
FILTERS = ["top50", "top10", "top1", "top01"]


def with_filters(unfiltered: dict, default: dict) -> list[dict]:
    return [unfiltered, {**{k: [v] for k, v in default.items()}, "filter": FILTERS}]


SWEEPS = {
    # flat has no search params: an empty search dict makes no --search flag, so its sweep is the filter itself.
    "flat": {"build": {}, "search": {"filter": ["none"] + FILTERS}},
    "ivf": {"build": {"nlist": [1024]}, "search": with_filters({"nprobe": [1, 4, 8, 16, 32, 64]}, {"nprobe": 8})},
    "pq": {"build": {"m": [48], "metric": ["ip", "l2"]}, "search": {"rerank": [0, 100]}},
    "ivf_pq": {"build": {"nlist": [1024], "m": [48], "metric": ["ip", "l2"]},
               "search": {"nprobe": [4, 8, 16, 32], "rerank": [0, 100]}},
    "hnsw": {"build": {"m": [16], "ef_construct": [100]}, "search": with_filters({"ef": [16, 32, 64, 128, 256]}, {"ef": 64})},
    "diskann": {"build": {"r": [64], "l_build": [100], "metric": ["ip", "l2"]},
                "search": {"l": [50, 100, 200], "io": ["mmap", "nocache"]}},
}


# Per-language overrides of SWEEPS (Phase 2): Qdrant's quantization is a dimension of its HNSW.
# pgvector (Phase 3): a B-tree on views (build views_index) and iterative scans (search
# iterative) are dimensions of its filtered search (tools/db/README.md).
SWEEPS_BY_LANGUAGE = {
    "qdrant": {
        "hnsw": {"build": {"m": [16], "ef_construct": [100], "quant": ["none", "scalar", "product", "binary"]},
                 "search": with_filters({"ef": [16, 32, 64, 128, 256], "rescore": [0, 1]}, {"ef": 64, "rescore": 0})},
    },
    "pgvector": {
        "ivf": {"build": {"nlist": [1024], "views_index": [0, 1]},
                "search": [{"nprobe": [1, 4, 8, 16, 32, 64]}, {"nprobe": [8], "filter": FILTERS, "iterative": [0, 1]}]},
        "hnsw": {"build": {"m": [16], "ef_construct": [100], "views_index": [0, 1]},
                 "search": [{"ef": [16, 32, 64, 128, 256]}, {"ef": [64], "filter": FILTERS, "iterative": [0, 1]}]},
    },
}


# Phase 4 load mode.
LOAD_CLIENTS = [1, 2, 4, 8, 16, 32, 64]
LOAD_INSERT = {"clients": 8, "insert_rate": 1000}
LOAD_LANGUAGES = ("python", "go", "cpp", "rust")  # their bench takes --clients, --duration, --insert-rate
LOAD_PROGRAMS = {db: ["uv", "run", "python", "-m", "tools.load.bench", "--db", db] for db in ("qdrant", "pgvector", "milvus")}
# Databases: every supported index (tools/db/base.py SUPPORTED); languages: hnsw only.
LOAD_INDEXES = {"qdrant": ["flat", "pq", "hnsw"], "pgvector": ["flat", "ivf", "hnsw"],
                "milvus": ["flat", "ivf", "ivf_pq", "hnsw", "diskann"]}


def load_cases(languages: list[str], indexes: list[str], data: Path, duration: float) -> list[dict]:
    """One case per (system, index, clients), plus one insert run per (system, index)."""
    out = []
    for lang in languages:
        for name in [i for i in LOAD_INDEXES.get(lang, ["hnsw"]) if i in indexes]:
            prog = LOAD_PROGRAMS.get(lang, PROGRAMS[lang])
            runs = [(c, 0) for c in LOAD_CLIENTS] + [(LOAD_INSERT["clients"], LOAD_INSERT["insert_rate"])]
            for c, rate in runs:
                tag = f"c{c}" + (f"-ins{rate}" if rate else "")
                path = Path("results/raw") / data.name / f"load-{lang}-{name}-{tag}.json"
                cmd = prog + ["--index", name, "--data", str(data), "--out", str(path),
                              "--clients", str(c), "--duration", str(duration)]
                if rate:
                    cmd += ["--insert-rate", str(rate)]
                if os.environ.get("VRO_THREADS") and lang not in LOAD_PROGRAMS:
                    cmd += ["--threads", os.environ["VRO_THREADS"]]
                out.append({"language": lang, "index": name, "out": path, "cmd": cmd})
    return out


# Phase 5 changes mode.
CHANGE_LANGUAGES = ("python", "go", "cpp", "rust")
CHANGE_INDEXES = ["flat", "ivf", "hnsw"]
CHANGE_RUNS = [(["--delete", d] + (["--compact"] if c else []), f"{d}{'-compact' if c else ''}")
               for d in ("del10", "del30", "del50") for c in (False, True)] + [(["--update", "upd10"], "upd10")]


def change_cases(languages: list[str], indexes: list[str], data: Path) -> list[dict]:
    """One case per (system, index, change); default build and search setting (no --build / --search)."""
    out = []
    for lang in languages:
        for name in [i for i in LOAD_INDEXES.get(lang, CHANGE_INDEXES) if i in indexes]:
            for flags, tag in CHANGE_RUNS:
                path = Path("results/raw") / data.name / f"chg-{lang}-{name}-{tag}.json"
                cmd = PROGRAMS[lang] + ["--index", name, "--data", str(data), "--out", str(path), *flags]
                if os.environ.get("VRO_THREADS") and lang in CHANGE_LANGUAGES:
                    cmd += ["--threads", os.environ["VRO_THREADS"]]
                out.append({"language": lang, "index": name, "out": path, "cmd": cmd})
    return out


# Phase 6 cache mode.
CACHE_PROGRAM = PROGRAMS["cache"]
CACHE_BACKENDS = ("none", "lru", "redis")
CACHE_CAPACITIES = [500, 2000, 5000]
CACHE_UNIFORM_CAPACITY = 2000
CACHE_INVALIDATE = {"backend": "lru", "capacity": 2000, "at": 25000}


def cache_cases(backends: list[str], data: Path, requests: int) -> list[dict]:
    """(backend, capacity, workload, invalidate_at) runs of CONTRACT section 14.3."""
    runs = []
    for b in backends:
        caps = [0] if b == "none" else CACHE_CAPACITIES
        runs += [(b, c, "zipf", None) for c in caps]
        runs.append((b, 0 if b == "none" else CACHE_UNIFORM_CAPACITY, "uniform", None))
    if CACHE_INVALIDATE["backend"] in backends and CACHE_INVALIDATE["at"] < requests:
        runs.append((CACHE_INVALIDATE["backend"], CACHE_INVALIDATE["capacity"], "zipf", CACHE_INVALIDATE["at"]))
    out = []
    for b, c, w, inv in runs:
        path = Path("results/raw") / data.name / f"cache-{b}-c{c}-{w}{'-inv' if inv is not None else ''}.json"
        cmd = CACHE_PROGRAM + ["--data", str(data), "--out", str(path), "--workload", w, "--backend", b,
                               "--capacity", str(c), "--requests", str(requests)]
        if inv is not None:
            cmd += ["--invalidate-at", str(inv)]
        out.append({"language": "cache", "index": "hnsw", "out": path, "cmd": cmd})
    return out


# Phase 7 backup mode.
BACKUP_PROGRAM = ["uv", "run", "python", "-m", "tools.backup.bench"]
BACKUP_INDEXES = {"qdrant": ["flat", "hnsw"], "pgvector": ["flat", "ivf", "hnsw"], "milvus": ["flat", "ivf", "hnsw"]}
BACKUP_IN_CONTAINER = {"qdrant"}  # the others need scripts/backup_db.sh


def backup_cases(dbs: list[str], indexes: list[str], data: Path, limit: int | None) -> list[dict]:
    """One case per (database, index) at the default build and search setting."""
    out = []
    folder = Path("results/raw") / data.name / ("bak-test" if limit else "")
    for db in dbs:
        for name in [i for i in BACKUP_INDEXES[db] if i in indexes]:
            path = folder / f"bak-{db}-{name}.json"
            cmd = BACKUP_PROGRAM + ["--db", db, "--index", name, "--data", str(data), "--out", str(path)]
            if limit:
                cmd += ["--limit", str(limit)]
            out.append({"language": db, "index": name, "out": path, "cmd": cmd})
    return out


def sweep_for(lang: str, name: str) -> dict:
    return SWEEPS_BY_LANGUAGE.get(lang, {}).get(name, SWEEPS[name])


def grid(spec: dict | list[dict]) -> list[dict]:
    """All combinations of a {key: values} spec; a list of specs gives their grids concatenated."""
    if isinstance(spec, list):
        return [g for part in spec for g in grid(part)]
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*spec.values())]


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def cases(languages: list[str], indexes: list[str], data: Path) -> list[dict]:
    """One case per (language, index, build variant)."""
    out = []
    for lang, name in itertools.product(languages, indexes):
        spec = sweep_for(lang, name)
        for bp in grid(spec["build"]):
            searches = grid(spec["search"])
            path = Path("results/raw") / data.name / f"{lang}-{name}-{params_hash(bp)}.json"
            cmd = PROGRAMS[lang] + ["--index", name, "--data", str(data), "--out", str(path)]
            if os.environ.get("VRO_THREADS"):  # the container is capped at fewer CPUs than it reports
                cmd += ["--threads", os.environ["VRO_THREADS"]]
            cmd += [a for k, v in bp.items() for a in ("--build", f"{k}={v}")]
            cmd += [a for sp in searches if sp for a in ("--search", ",".join(f"{k}={v}" for k, v in sp.items()))]
            out.append({"language": lang, "index": name, "out": path, "cmd": cmd})
    return out


def program_exists(lang: str) -> bool:
    return PYTHON_MODULES[lang].exists() if lang in PYTHON_MODULES else Path(PROGRAMS[lang][0]).exists()


def run_once(cmd: list[str], timeout: float | None) -> tuple[int | str, str]:
    """Run one bench process. Returns (exit code or "timeout", stderr)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as e:
        err = e.stderr or ""
        return "timeout", err.decode(errors="replace") if isinstance(err, bytes) else err


def mean_p50(doc: dict) -> float:
    """One number per run for picking the median run: mean over searches of the p50 latency."""
    return statistics.mean(statistics.median(s["latency_ms"]) for s in doc["searches"])


def run_case(case: dict, timeout: float | None, repeat: int) -> int | str:
    """Run a case `repeat` times as separate processes and keep the run with the median p50.

    macOS moves a process between performance and efficiency cores, so one run can be
    2x off (docs/measurement-notes.md, item 1). The spread of all runs is kept in extra.
    """
    out: Path = case["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[float, dict]] = []
    t0 = time.perf_counter()
    for i in range(repeat):
        tmp = out.with_suffix(f".run{i}.json")
        cmd = list(case["cmd"])
        cmd[cmd.index("--out") + 1] = str(tmp)
        code, stderr = run_once(cmd, timeout)
        if code != 0:
            err = out.with_suffix(".stderr.txt")
            err.write_text(stderr)
            print(f"{out.name}: exit {code} on run {i} in {time.perf_counter() - t0:.1f}s, stderr saved to {err}", flush=True)
            case["stderr"] = stderr
            return code
        doc = json.loads(tmp.read_text())
        runs.append((mean_p50(doc), doc))
        tmp.unlink()
    runs.sort(key=lambda r: r[0])
    p50s = [round(r[0], 4) for r in runs]
    doc = runs[len(runs) // 2][1]
    doc["extra"]["runner"] = {"repeat": repeat, "p50_ms_runs": p50s, "load1_at_start": case["load1"]}
    out.write_text(json.dumps(doc))
    spread = f", p50 runs {p50s}" if repeat > 1 else ""
    print(f"{out.name}: exit 0 in {time.perf_counter() - t0:.1f}s{spread}", flush=True)
    for e in validate(doc):
        print(f"  schema: {e}")
    return 0


def wait_for_idle(max_load: float, patience_s: float = 180.0) -> float:
    """Wait until the 1-minute load average is below max_load. Our own multi-threaded
    builds raise it too, so after `patience_s` the run continues and the load is recorded."""
    t0 = time.perf_counter()
    while (load1 := os.getloadavg()[0]) > max_load:
        if time.perf_counter() - t0 > patience_s:
            print(f"  warning: load average still {load1:.1f} > {max_load} after {patience_s:.0f}s; running anyway", flush=True)
            break
        time.sleep(10)
    return round(load1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    ap.add_argument("--languages", default="python,go,cpp,rust,faiss")
    ap.add_argument("--indexes", default=None, help="default: all (load mode: hnsw for languages, all supported for databases)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--repeat", type=int, default=3, help="runs per case; the median-p50 run is kept")
    ap.add_argument("--load", action="store_true", help="Phase 4: clients sweep plus one insert run (repeat 1)")
    ap.add_argument("--changes", action="store_true", help="Phase 5: deletes (with and without compaction) and updates (repeat 1)")
    ap.add_argument("--cache", action="store_true", help="Phase 6: embedding cache sweep (repeat 1; run in dbbench for redis)")
    ap.add_argument("--backup", action="store_true", help="Phase 7: database backup and restore (repeat 1; see make backup-db)")
    ap.add_argument("--list", action="store_true", help="backup mode: print '<index> <out> <exists>' per case and exit")
    ap.add_argument("--limit", type=int, default=None, help="backup mode: rows (test runs; outputs under bak-test/)")
    ap.add_argument("--backends", default=",".join(CACHE_BACKENDS), help="cache mode: backends to run")
    ap.add_argument("--requests", type=int, default=50000, help="cache mode: requests per workload")
    ap.add_argument("--duration", type=float, default=20.0, help="load mode: seconds per run")
    ap.add_argument("--max-load", type=float, default=2.0, help="wait (up to 3 min) while the 1-minute load average is above this")
    args = ap.parse_args()
    languages, indexes = args.languages.split(","), (args.indexes or ",".join(SWEEPS)).split(",")
    if args.cache:
        languages, indexes = ["cache"], ["hnsw"]
    elif "cache" in languages:
        sys.exit("language cache runs only with --cache")
    for bad in [l for l in languages if l not in PROGRAMS] + [i for i in indexes if i not in SWEEPS]:
        sys.exit(f"unknown language or index: {bad}")
    if args.cache:
        backends = args.backends.split(",")
        bad = [b for b in backends if b not in CACHE_BACKENDS]
        if bad:
            sys.exit(f"unknown cache backend {bad}; known: {list(CACHE_BACKENDS)}")
        todo, repeat = cache_cases(backends, args.data, args.requests), 1
    elif args.load:
        bad = [l for l in languages if l not in LOAD_LANGUAGES and l not in LOAD_PROGRAMS]
        if bad:
            sys.exit(f"load mode has no {bad}; known: {list(LOAD_LANGUAGES) + list(LOAD_PROGRAMS)}")
        todo, repeat = load_cases(languages, indexes, args.data, args.duration), 1
    elif args.changes:
        bad = [l for l in languages if l not in CHANGE_LANGUAGES and l not in LOAD_PROGRAMS]
        if bad:
            sys.exit(f"changes mode has no {bad}; known: {list(CHANGE_LANGUAGES) + list(LOAD_PROGRAMS)}")
        todo, repeat = change_cases(languages, indexes, args.data), 1
    elif args.backup:
        bad = [l for l in languages if l not in BACKUP_INDEXES]
        if bad:
            sys.exit(f"backup mode has no {bad}; known: {list(BACKUP_INDEXES)}")
        todo, repeat = backup_cases(languages, indexes, args.data, args.limit), 1
        if args.list:
            for case in todo:
                print(case["index"], case["out"], int(case["out"].exists()))
            return
        for case in [c for c in todo if c["language"] not in BACKUP_IN_CONTAINER]:
            print(f"{case['out'].name}: {case['language']} needs a host step; run make backup-db DB={case['language']}")
        todo = [c for c in todo if c["language"] in BACKUP_IN_CONTAINER]
    else:
        todo, repeat = cases(languages, indexes, args.data), args.repeat

    no_load: set[str] = set()  # languages whose bench rejected --clients (exit 2)
    for case in todo:
        if case["language"] in no_load:
            print(f"{case['out'].name}: skipped, {case['language']} bench has no --clients")
        elif args.dry_run:
            print(shlex.join(case["cmd"]))
        elif not args.backup and not program_exists(case["language"]):
            print(f"{case['out'].name}: skipped, bench program for {case['language']} not found ({shlex.join(PROGRAMS[case['language']])})")
        elif (case["language"], case["index"]) in UNSUPPORTED:
            print(f"{case['out'].name}: skipped, {case['language']} has no {case['index']}")
        elif case["out"].exists() and not args.force:
            print(f"{case['out'].name}: exists, skipped (use --force)")
        else:
            case["load1"] = wait_for_idle(args.max_load)
            code = run_case(case, args.timeout, repeat)
            if args.load and code == 2 and case["language"] in LOAD_LANGUAGES:
                last = (case.get("stderr") or "").strip().splitlines()[-1:] or ["(no stderr)"]
                print(f"  {case['language']}: bench exits 2 on --clients ({last[0]}); skipping its other load runs")
                no_load.add(case["language"])


if __name__ == "__main__":
    main()
