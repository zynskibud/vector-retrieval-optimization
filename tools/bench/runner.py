"""Run benchmark cases one at a time, each as a subprocess of a language's bench program.

Cases never run in parallel: latency numbers depend on an idle CPU.
Outputs: results/raw/<data-name>/<language>-<index>-<hash of build params>.json

Run: uv run python -m tools.bench.runner --data data/processed/dev --languages faiss --indexes flat,ivf,hnsw [--repeat 3] [--dry-run]
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
                  "milvus": Path("tools/db/milvus.py")}
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


def run_case(case: dict, timeout: float | None, repeat: int) -> None:
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
            return
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
    ap.add_argument("--indexes", default=",".join(SWEEPS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--repeat", type=int, default=3, help="runs per case; the median-p50 run is kept")
    ap.add_argument("--max-load", type=float, default=2.0, help="wait (up to 3 min) while the 1-minute load average is above this")
    args = ap.parse_args()
    languages, indexes = args.languages.split(","), args.indexes.split(",")
    for bad in [l for l in languages if l not in PROGRAMS] + [i for i in indexes if i not in SWEEPS]:
        sys.exit(f"unknown language or index: {bad}")

    for case in cases(languages, indexes, args.data):
        if args.dry_run:
            print(shlex.join(case["cmd"]))
        elif not program_exists(case["language"]):
            print(f"{case['out'].name}: skipped, bench program for {case['language']} not found ({shlex.join(PROGRAMS[case['language']])})")
        elif (case["language"], case["index"]) in UNSUPPORTED:
            print(f"{case['out'].name}: skipped, {case['language']} has no {case['index']}")
        elif case["out"].exists() and not args.force:
            print(f"{case['out'].name}: exists, skipped (use --force)")
        else:
            case["load1"] = wait_for_idle(args.max_load)
            run_case(case, args.timeout, args.repeat)


if __name__ == "__main__":
    main()
