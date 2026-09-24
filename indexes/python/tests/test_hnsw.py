"""HNSW tests (CONTRACT 6.6, 9). Uses the first 20,000 dev rows, with exact top-10 truth
computed by brute force inside the test, because a full 100k Python build is too slow for a test."""

import json
import subprocess
import sys
from collections import deque

import numpy as np
import pytest

from indexes.python import hnsw
from indexes.python.npy import read_npy

DEV = "data/processed/dev"
LIMIT = 20000
M = 16


@pytest.fixture(scope="module")
def data():
    vectors = read_npy(f"{DEV}/vectors.npy", LIMIT)
    queries = read_npy(f"{DEV}/queries.npy")
    s = queries @ vectors.T
    ids = np.arange(LIMIT)
    truth = np.stack([ids[np.lexsort((ids, -row))[:10]] for row in s])
    index = hnsw.build(vectors, {"m": M, "ef_construct": 100}, 1, 42)
    return vectors, queries, truth, index


def recall(index, queries, truth, ef):
    ids = np.stack([hnsw.search(index, q, 10, {"ef": ef})[0] for q in queries])
    return sum(len(set(a) & set(b)) for a, b in zip(ids.tolist(), truth.tolist())) / truth.size


def test_recall_and_monotonic(data):
    _, queries, truth, index = data
    r16, r64, r128 = (recall(index, queries, truth, ef) for ef in (16, 64, 128))
    print(f"recall@10 ef=16 {r16:.4f} ef=64 {r64:.4f} ef=128 {r128:.4f}")
    assert r64 >= 0.95
    assert r128 >= r64 >= r16


def test_search_output(data):
    _, queries, _, index = data
    ids, scores = hnsw.search(index, queries[0], 10, {"ef": 64})
    assert ids.dtype == np.int64 and scores.dtype == np.float32 and len(ids) == 10
    assert (np.diff(scores) <= 0).all()
    assert index["distance_computations"] > 0


def test_levels_deterministic():
    a = hnsw.draw_levels(LIMIT, M, 42)
    b = hnsw.draw_levels(LIMIT, M, 42)
    assert (a == b).all()
    frac = float((a >= 1).mean())
    assert 0.04 <= frac <= 0.09, frac


def test_degree_limits(data):
    _, _, _, index = data
    g = index["graph"]
    assert (g.cnt0 <= 2 * M).all()
    assert (g.up_cnt <= M).all()
    assert g.nbr0.dtype == np.int32 and g.up.dtype == np.int32


def test_layer0_connected(data):
    _, _, _, index = data
    g = index["graph"]
    seen = np.zeros(LIMIT, dtype=bool)
    seen[index["entry"]] = True
    todo = deque([index["entry"]])
    while todo:
        c = todo.popleft()
        for e in g.nbr0[c, : g.cnt0[c]]:
            if not seen[e]:
                seen[e] = True
                todo.append(e)
    assert seen.all(), int((~seen).sum())
    print("extra", index["extra"])


def test_threads_ignored():
    """The Python build is sequential for any thread count (CONTRACT 6.6 allows this), so an
    all-cores build equals a one-thread build. Checked on 3,000 rows to keep the test short."""
    import os

    vectors = read_npy(f"{DEV}/vectors.npy", 3000)
    a = hnsw.build(vectors, {"m": M, "ef_construct": 100}, 1, 42)["graph"]
    b = hnsw.build(vectors, {"m": M, "ef_construct": 100}, os.cpu_count(), 42)["graph"]
    assert (a.nbr0 == b.nbr0).all() and (a.up == b.up).all()


def test_bench_json(tmp_path):
    from tools.bench import schema

    out = tmp_path / "r.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "hnsw", "--data", DEV,
           "--out", str(out), "--limit", str(LIMIT), "--threads", "1", "--warmup", "10",
           "--search", "ef=16", "--search", "ef=64"]
    subprocess.run(cmd, check=True)
    result = json.loads(out.read_text())
    assert schema.validate(result) == []
    for key in ("top_layer", "entry_point", "nodes_per_layer", "unreachable_before_repair", "repair_added",
                "repair_added_unreachable", "build_threads"):
        assert key in result["extra"], key
    assert [s["search_params"]["ef"] for s in result["searches"]] == [16, 64]
    assert result["build"]["index_bytes"] > 0
