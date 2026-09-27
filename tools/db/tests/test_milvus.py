"""Milvus client tests (tools/db/README.md, Tests). Needs `make db-up DB=milvus`.

Run: make db-test DB=milvus
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tools.bench import schema
from tools.db import base
from tools.db.milvus import make_client
from tools.db.tests import filtering

DATA = Path("data/processed/dev")
N = 20000
K = 10


@pytest.fixture(scope="module")
def setup():
    vectors = np.ascontiguousarray(np.load(DATA / "vectors.npy", mmap_mode="r")[:N])
    queries = np.load(DATA / "queries.npy")
    meta = pq.read_table(DATA / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, N)
    scores = queries @ vectors.T
    truth = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    c = make_client()
    c.connect()
    c.reset()
    c.load(vectors, meta, 2000)
    c.vectors = vectors  # for the filter test
    yield c, queries, truth
    c.reset()
    c.close()


def recall(c, queries, truth, params):
    hits = 0
    for i, q in enumerate(queries):
        ids, _ = c.search(q, K, params)
        hits += len(set(ids[:K]) & set(truth[i].tolist()))
    return hits / (len(queries) * K)


def test_rows(setup):
    c, _, _ = setup
    assert c.stats()["rows"] == N


@pytest.mark.parametrize("index,build,search,floor", [
    ("flat", {}, {}, 1.0),
    ("hnsw", base.BUILD_DEFAULTS["hnsw"], {"ef": 64}, 0.95),
    ("ivf", {"nlist": 256}, {"nprobe": 8}, 0.75),
    ("ivf_pq", {"nlist": 256, "m": 48, "nbits": 8}, {"nprobe": 8, "rerank": 0}, 0.45),
    ("diskann", base.BUILD_DEFAULTS["diskann"], {"l": 100}, 0.90),
])
def test_recall(setup, index, build, search, floor):
    c, queries, truth = setup
    if c.client.has_collection("vro") and c.index is not None:
        c.client.release_collection("vro")
        c.client.drop_index("vro", "embedding")
    try:
        c.build_index(index, build)
    except Exception as e:  # noqa: BLE001
        if index == "diskann":
            pytest.skip(f"DISKANN did not build in standalone mode: {e}")
        raise
    r = recall(c, queries, truth, search)
    print(f"{index} recall@10 = {r:.4f}")
    assert r >= floor


def test_hnsw_filter(setup):
    c, queries, _ = setup
    if c.index is not None:
        c.client.release_collection("vro")
        c.client.drop_index("vro", "embedding")
    c.build_index("hnsw", base.BUILD_DEFAULTS["hnsw"])
    r = filtering.check(c, DATA, c.vectors, queries, {"ef": 64, "filter": "top10"})
    assert r >= 0.85, r
    filtering.check(c, DATA, c.vectors, queries, {"ef": 64, "filter": "top01"})


def test_bench_subprocess(setup):
    c, _, _ = setup
    out = Path("/tmp/m.json")
    cmd = [sys.executable, "-m", "tools.db.bench", "--db", "milvus", "--index", "hnsw", "--data", str(DATA),
           "--out", str(out), "--limit", str(N), "--search", "ef=16", "--search", "ef=64"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    doc = json.loads(out.read_text())
    # tools/bench/schema.py lists only the languages; "milvus" is the database name (README, Output JSON).
    errors = schema.validate(doc)
    assert errors == [], errors
    assert "load_s" in doc["extra"] and "server_build_s" in doc["extra"]
    c.index = None  # the subprocess replaced the collection


def test_load():
    """Phase 4 (CONTRACT section 12.5): 8 clients; then 4 clients with inserts. Leaves the data dropped."""
    from tools.db.tests import load

    try:
        load.check("milvus")
    finally:
        c = make_client()
        c.connect()
        c.reset()
        c.close()


def test_changes(setup):
    """Phase 5 (CONTRACT section 13.5): del30, compact, upd10 on hnsw. Leaves the collection dropped."""
    from tools.db.tests import changes

    c, _, _ = setup
    changes.check(c, "hnsw", search={"ef": 64})
    c.index = None
