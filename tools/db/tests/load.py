"""Shared Phase 4 load test for the database tests (CONTRACT section 12.5).

Runs tools.load.bench and tools.db.bench as subprocesses on the first 20,000 rows of the dev set.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.bench.metrics import recall_at_k
from tools.bench.schema import validate

DATA = Path("data/processed/dev")
N = 20000
K = 10


def _run(module: str, db: str, out: Path, *args: str) -> dict:
    cmd = [sys.executable, "-m", module, "--db", db, "--index", "hnsw", "--data", str(DATA), "--out", str(out),
           "--limit", str(N), "--search", "ef=64", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert validate(doc) == []
    return doc


def truth() -> np.ndarray:
    vectors = np.load(DATA / "vectors.npy", mmap_mode="r")[:N]
    queries = np.load(DATA / "queries.npy")
    return np.argsort(-(queries @ np.asarray(vectors).T), axis=1, kind="stable")[:, :K]


def check(db: str) -> None:
    gt = truth()
    tmp = Path("/tmp")
    # 8 clients, 5 s, static index.
    doc = _run("tools.load.bench", db, tmp / f"{db}-load8.json", "--clients", "8", "--duration", "5")
    run = doc["searches"][0]
    first = recall_at_k(run["ids"], gt, K)[1]
    print(f"{db} clients=8: qps={run['qps']:.0f} errors={run['extra']['errors']} cpu={run['extra']['cpu_pct']:.0f}% "
          f"first-pass recall={first:.4f}")
    assert run["extra"]["errors"] == 0 and run["extra"]["clients"] == 8
    assert run["qps"] > 0 and first >= 0.95, first

    # 4 clients, 10 s, 2,000 rows inserted at 2,000 rows/s into a build on 18,000.
    doc = _run("tools.load.bench", db, tmp / f"{db}-load-ins.json", "--clients", "4", "--duration", "10",
               "--insert-rate", "2000")
    assert doc["extra"]["build_rows"] == 18000
    loop, after = doc["searches"]
    assert after["search_params"]["phase"] == "after_inserts"
    assert loop["extra"]["errors"] == 0 and loop["extra"]["insert_errors"] == 0
    assert loop["extra"]["inserted_rows"] == 2000 and after["extra"]["inserted_rows"] == 2000
    after_recall = recall_at_k(after["ids"], gt, K)[1]

    static = _run("tools.db.bench", db, tmp / f"{db}-static.json")
    static_recall = recall_at_k(static["searches"][0]["ids"], gt, K)[1]
    print(f"{db} inserts: loop qps={loop['qps']:.0f} insert_p50_ms={loop['extra']['insert_p50_ms']:.1f} "
          f"after-inserts recall={after_recall:.4f} static recall={static_recall:.4f}")
    assert abs(after_recall - static_recall) <= 0.01, (after_recall, static_recall)
