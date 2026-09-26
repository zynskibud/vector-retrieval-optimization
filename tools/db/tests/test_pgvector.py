"""pgvector client tests. Run with the database up: make db-up DB=pgvector; make db-test DB=pgvector.

20,000 rows of data/processed/dev. IVF uses lists=256 (not the contract 1024) on 20k rows,
so each list holds about 78 rows, the same order as 1024 lists on the 100k dev set;
nprobe=8 as the contract default.
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
from tools.db.pgvector import make_client
from tools.db.tests import filtering

DATA = Path("data/processed/dev")
N = 20000
K = 10


@pytest.fixture(scope="module")
def setup():
    vectors = np.ascontiguousarray(np.load(DATA / "vectors.npy", mmap_mode="r")[:N])
    queries = np.load(DATA / "queries.npy")
    meta = pq.read_table(DATA / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, N)
    sims = queries @ vectors.T
    truth = np.argsort(-sims, axis=1, kind="stable")[:, :K]
    client = make_client()
    client.connect()
    client.reset()
    client.load(vectors, meta, 2000)
    client.vectors = vectors  # for the filter tests
    yield client, queries, truth
    client.reset()
    client.close()


def recall(client, queries, truth, params):
    hits = 0
    for i, q in enumerate(queries):
        ids, _ = client.search(q, K, params)
        hits += len(set(ids[:K]) & set(truth[i].tolist()))
    return hits / (len(queries) * K)


def test_rows(setup):
    client, _, _ = setup
    assert client.stats()["rows"] == N


def test_flat(setup):
    client, queries, truth = setup
    client.build_index("flat", {})
    assert recall(client, queries, truth, {}) == 1.0


def test_ivf(setup):
    client, queries, truth = setup
    client.conn.execute("DROP INDEX IF EXISTS items_ivf, items_hnsw")
    client.build_index("ivf", {"nlist": 256})
    r = recall(client, queries, truth, {"nprobe": 8})
    print(f"ivf lists=256 nprobe=8 recall@10={r:.4f}")
    assert r >= 0.75
    assert client.stats()["index_bytes"] > 0
    client.conn.execute("DROP INDEX items_ivf")


def test_hnsw(setup):
    client, queries, truth = setup
    client.conn.execute("DROP INDEX IF EXISTS items_ivf, items_hnsw")
    client.build_index("hnsw", base.BUILD_DEFAULTS["hnsw"])
    r = recall(client, queries, truth, {"ef": 64})
    print(f"hnsw m=16 ef_construct=100 ef=64 recall@10={r:.4f}")
    assert r >= 0.95
    client.conn.execute("DROP INDEX items_hnsw")


@pytest.mark.parametrize("views_index,iterative", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_hnsw_filter(setup, views_index, iterative):
    client, queries, _ = setup
    client.conn.execute("DROP INDEX IF EXISTS items_ivf, items_hnsw")
    client.build_index("hnsw", {**base.BUILD_DEFAULTS["hnsw"], "views_index": views_index})
    params = {"ef": 64, "filter": "top10", "iterative": iterative}
    r = filtering.check(client, DATA, client.vectors, queries, params)
    assert r >= 0.85, r
    filtering.check(client, DATA, client.vectors, queries, {**params, "filter": "top01"})
    client.conn.execute("DROP INDEX IF EXISTS items_hnsw, items_views")


def test_bench_subprocess(setup, tmp_path):
    out = Path("/tmp/p.json")
    cmd = [sys.executable, "-m", "tools.db.bench", "--db", "pgvector", "--index", "hnsw",
           "--data", str(DATA), "--out", str(out), "--limit", str(N),
           "--search", "ef=16", "--search", "ef=64"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    # documented `language` value (tools/db/README.md), so that one error is expected.
    errors = schema.validate(doc)
    assert errors == []
    assert doc["extra"]["load_s"] > 0
    assert doc["extra"]["server_build_s"] > 0
    assert doc["build"]["index_bytes"] > 0
