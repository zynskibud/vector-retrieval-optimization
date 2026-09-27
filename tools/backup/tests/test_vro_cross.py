"""Cross-language .vro test (CONTRACT section 15.4): the Rust bench builds flat, ivf, hnsw on
20,000 rows with --threads 1 and writes each with --save; the Go, C++ and Python benches load
each file with --load and must return the same IDs as Rust for all 1,000 queries.

Runs inside the bench container with the built binaries (make build first); a language whose
bench program is missing is skipped. Run: make vro-test
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DATA = "data/processed/dev"
LIMIT = "20000"
RUST = Path("indexes/rust/target/release/bench")
LOADERS = {
    "go": [str(Path("indexes/go/bin/bench"))],
    "cpp": [str(Path("indexes/cpp/build/bench"))],
    "python": [sys.executable, "-m", "indexes.python.bench"],
}
EXISTS = {"go": Path("indexes/go/bin/bench").exists(), "cpp": Path("indexes/cpp/build/bench").exists(), "python": True}
INDEXES = ("flat", "ivf", "hnsw")
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **ONE_THREAD}, timeout=3600)
    assert proc.returncode == 0, f"{' '.join(cmd)}\nexit {proc.returncode}\n{proc.stderr[-2000:]}"


@pytest.fixture(scope="module")
def rust_files(tmp_path_factory):
    if not RUST.exists():
        pytest.skip(f"{RUST} not built")
    folder = tmp_path_factory.mktemp("vro")
    out = {}
    for index in INDEXES:
        vro, js = folder / f"rust-{index}.vro", folder / f"rust-{index}.json"
        run([str(RUST), "--index", index, "--data", DATA, "--limit", LIMIT, "--threads", "1",
             "--save", str(vro), "--out", str(js)])
        out[index] = (vro, json.loads(js.read_text()))
    return folder, out


@pytest.mark.parametrize("lang", sorted(LOADERS))
@pytest.mark.parametrize("index", INDEXES)
def test_load_rust_file(rust_files, lang, index):
    if not EXISTS[lang]:
        pytest.skip(f"{lang} bench not built")
    folder, files = rust_files
    vro, rust_doc = files[index]
    js = folder / f"{lang}-{index}.json"
    run(LOADERS[lang] + ["--index", index, "--data", DATA, "--limit", LIMIT, "--threads", "1",
                         "--load", str(vro), "--out", str(js)])
    doc = json.loads(js.read_text())
    assert doc["build"]["train_s"] == 0 and doc["build"]["add_s"] == 0
    assert doc["extra"]["loaded_from"]
    assert doc["build_params"] == rust_doc["build_params"]
    rust_ids, ids = rust_doc["searches"][0]["ids"], doc["searches"][0]["ids"]
    assert len(ids) == len(rust_ids) == 1000
    differ = [q for q in range(1000) if ids[q] != rust_ids[q]]
    assert not differ, f"{len(differ)} queries differ, first {differ[:5]}"
