"""Run benchmark cases one at a time, each as a subprocess of a language's bench program.

Cases never run in parallel: latency numbers depend on an idle CPU.
Outputs: results/raw/<data-name>/<language>-<index>-<hash of build params>.json

Run: uv run python -m tools.bench.runner --data data/processed/dev --languages faiss --indexes flat,ivf,hnsw [--dry-run]
"""

import argparse
import hashlib
import itertools
import json
import shlex
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
}
PYTHON_MODULES = {"python": Path("indexes/python/bench.py"), "faiss": Path("tools/bench/faiss_ref.py")}

# Per index: build variants (each value list is swept, one build per combination)
# and the search sweep (every combination is one --search inside the same build).
SWEEPS = {
    "flat": {"build": {}, "search": {}},
    "ivf": {"build": {"nlist": [1024]}, "search": {"nprobe": [1, 4, 8, 16, 32, 64]}},
    "pq": {"build": {"m": [48], "metric": ["ip", "l2"]}, "search": {"rerank": [0, 100]}},
    "ivf_pq": {"build": {"nlist": [1024], "m": [48], "metric": ["ip", "l2"]},
               "search": {"nprobe": [4, 8, 16, 32], "rerank": [0, 100]}},
    "hnsw": {"build": {"m": [16], "ef_construct": [100]}, "search": {"ef": [16, 32, 64, 128, 256]}},
    "diskann": {"build": {"r": [64], "l_build": [100]},
                "search": {"l": [50, 100, 200], "io": ["mmap", "nocache"]}},
}


def grid(spec: dict) -> list[dict]:
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*spec.values())]


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def cases(languages: list[str], indexes: list[str], data: Path) -> list[dict]:
    """One case per (language, index, build variant)."""
    out = []
    for lang, name in itertools.product(languages, indexes):
        for bp in grid(SWEEPS[name]["build"]):
            searches = grid(SWEEPS[name]["search"])
            path = Path("results/raw") / data.name / f"{lang}-{name}-{params_hash(bp)}.json"
            cmd = PROGRAMS[lang] + ["--index", name, "--data", str(data), "--out", str(path)]
            cmd += [a for k, v in bp.items() for a in ("--build", f"{k}={v}")]
            cmd += [a for sp in searches if sp for a in ("--search", ",".join(f"{k}={v}" for k, v in sp.items()))]
            out.append({"language": lang, "index": name, "out": path, "cmd": cmd})
    return out


def program_exists(lang: str) -> bool:
    return PYTHON_MODULES[lang].exists() if lang in PYTHON_MODULES else Path(PROGRAMS[lang][0]).exists()


def run_case(case: dict, timeout: float | None) -> None:
    out: Path = case["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(case["cmd"], capture_output=True, text=True, timeout=timeout)
        code, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as e:
        code, stderr = "timeout", (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    print(f"{out.name}: exit {code} in {time.perf_counter() - t0:.1f}s", flush=True)
    if code != 0:
        err = out.with_suffix(".stderr.txt")
        err.write_text(stderr)
        print(f"  stderr saved to {err}")
        return
    errors = validate(json.loads(out.read_text()))
    for e in errors:
        print(f"  schema: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    ap.add_argument("--languages", default="python,go,cpp,rust,faiss")
    ap.add_argument("--indexes", default=",".join(SWEEPS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()
    languages, indexes = args.languages.split(","), args.indexes.split(",")
    for bad in [l for l in languages if l not in PROGRAMS] + [i for i in indexes if i not in SWEEPS]:
        sys.exit(f"unknown language or index: {bad}")

    for case in cases(languages, indexes, args.data):
        if args.dry_run:
            print(shlex.join(case["cmd"]))
        elif not program_exists(case["language"]):
            print(f"{case['out'].name}: skipped, bench program for {case['language']} not found ({shlex.join(PROGRAMS[case['language']])})")
        elif case["language"] == "faiss" and case["index"] == "diskann":
            print(f"{case['out'].name}: skipped, faiss has no diskann")
        elif case["out"].exists() and not args.force:
            print(f"{case['out'].name}: exists, skipped (use --force)")
        else:
            run_case(case, args.timeout)


if __name__ == "__main__":
    main()
