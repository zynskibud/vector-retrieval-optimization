"""Qdrant client tests (tools/db/README.md, Tests). Needs `make db-up DB=qdrant`."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tools.bench.schema import validate
from tools.db import base
from tools.db.qdrant import make_client
from tools.db.tests import filtering

DATA = Path("data/processed/dev")
N = 20000
K = 10


@pytest.fixture(scope="module")
def data():
    vectors = np.ascontiguousarray(np.load(DATA / "vectors.npy", mmap_mode="r")[:N])
    queries = np.load(DATA / "queries.npy")
    meta = pq.read_table(DATA / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, N)
    truth = np.argsort(-(queries @ vectors.T), axis=1, kind="stable")[:, :K]
    return vectors, queries, meta, truth


@pytest.fixture(scope="module")
def client():
    c = make_client()
    c.connect()
    yield c
    c.reset()
    c.close()


def recall(c, queries, truth, params) -> float:
    hits = 0
    for q, t in zip(queries, truth):
        ids, _ = c.search(q, K, params)
        hits += len(set(ids) & set(t.tolist()))
    return hits / (len(queries) * K)


def setup_index(c, data, index, build):
    vectors, _, meta, _ = data
    c.reset()
    c.load(vectors, meta, 2000)
    c.build_index(index, {**base.BUILD_DEFAULTS[index], **build})


def test_load_rows(client, data):
    setup_index(client, data, "flat", {})
    assert client.stats()["rows"] == N


def test_flat_recall(client, data):
    setup_index(client, data, "flat", {})
    assert recall(client, data[1], data[3], {}) == 1.0


def test_hnsw_recall(client, data):
    setup_index(client, data, "hnsw", {"m": 16, "ef_construct": 100})
    r = recall(client, data[1], data[3], {"ef": 64, "rescore": 0})
    assert r >= 0.95, r


def test_hnsw_scalar_recall(client, data):
    setup_index(client, data, "hnsw", {"quant": "scalar"})
    r = recall(client, data[1], data[3], {"ef": 64, "rescore": 1})
    assert r >= 0.90, r


def test_hnsw_filter(client, data):
    setup_index(client, data, "hnsw", {})
    vectors, queries = data[0], data[1]
    r = filtering.check(client, DATA, vectors, queries, {"ef": 64, "rescore": 0, "filter": "top10"})
    assert r >= 0.85, r
    filtering.check(client, DATA, vectors, queries, {"ef": 64, "rescore": 0, "filter": "top01"})


def test_bench_subprocess(client, tmp_path):
    out = Path("/tmp/q.json")
    cmd = [sys.executable, "-m", "tools.db.bench", "--db", "qdrant", "--index", "hnsw", "--data", str(DATA),
           "--out", str(out), "--limit", str(N), "--search", "ef=16", "--search", "ef=64"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    errors = validate(doc)
    assert errors == []
    assert "load_s" in doc["extra"] and "server_build_s" in doc["extra"]
    client.reset()
    assert not client.client.collection_exists("vro")


def test_load(client):
    """Phase 4 (CONTRACT section 12.5): 8 clients; then 4 clients with inserts. Leaves the collection dropped."""
    from tools.db.tests import load

    load.check("qdrant")
    client.reset()


def test_changes(client):
    """Phase 5 (CONTRACT section 13.5): del30, compact, upd10 on hnsw. Leaves the collection dropped."""
    from tools.db.tests import changes

    changes.check(client, "hnsw", search={"ef": 64, "rescore": 0})
