"""Load-run tests (CONTRACT 12.5) for the Python HNSW, on the first 20,000 dev rows.

Truth is exact top-10 by brute force inside the test (as in test_hnsw.py).
Under the GIL, 8 client threads do not beat 1 thread by much, so (a) asserts qps > 0 only and
prints both numbers.
"""

import json

import numpy as np
import pytest

from indexes.python import bench
from indexes.python.npy import read_npy

DEV = "data/processed/dev"
LIMIT = 20000
LOAD_KEYS = {"errors", "cpu_pct", "clients", "duration_s", "queries_done"}


@pytest.fixture(scope="module")
def truth():
    vectors = read_npy(f"{DEV}/vectors.npy", LIMIT)
    queries = read_npy(f"{DEV}/queries.npy")
    s = queries @ vectors.T
    ids = np.arange(LIMIT)
    return np.stack([ids[np.lexsort((ids, -row))[:10]] for row in s])


def recall(ids, truth):
    return sum(len(set(a) & set(b)) for a, b in zip(ids, truth.tolist())) / truth.size


def bench_json(tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    rc = bench.main(["--index", "hnsw", "--data", DEV, "--out", str(out), "--limit", str(LIMIT),
                     "--warmup", "10", "--search", "ef=64", *extra])
    assert rc == 0
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def static(tmp_path_factory):
    return bench_json(tmp_path_factory.mktemp("static"), "static")


def test_clients_8(tmp_path, truth, static):
    r = bench_json(tmp_path, "c8", "--clients", "8", "--duration", "5")
    (s,) = r["searches"]
    e = s["extra"]
    rec = recall(s["ids"], truth)
    print(f"clients=8 qps {s['qps']:.1f} (1 client one pass {static['searches'][0]['qps']:.1f}) "
          f"cpu_pct {e['cpu_pct']:.0f} recall {rec:.4f}")
    assert e["errors"] == 0
    assert s["qps"] > 0 and e["queries_done"] == len(s["latency_ms"]) > 0
    assert rec >= 0.95


def test_inserts(tmp_path, truth, static):
    r = bench_json(tmp_path, "ins", "--clients", "4", "--duration", "10", "--insert-rate", "2000")
    assert r["n"] == LIMIT
    loop, after = r["searches"]
    assert loop["extra"]["errors"] == 0
    assert after["search_params"]["phase"] == "after_inserts"
    assert after["extra"]["inserted_rows"] == 2000
    assert after["extra"]["insert_p50_ms"] > 0
    assert 0 <= after["extra"]["inserted_during_loop"] <= 2000 and after["extra"]["insert_tail_s"] >= 0
    rec_after = recall(after["ids"], truth)
    rec_static = recall(static["searches"][0]["ids"], truth)
    print(f"after-inserts recall {rec_after:.4f}, static recall {rec_static:.4f}, "
          f"loop qps {loop['qps']:.1f}, insert p50 {after['extra']['insert_p50_ms']:.1f} ms/100 rows, "
          f"during loop {after['extra']['inserted_during_loop']}, tail {after['extra']['insert_tail_s']:.1f} s")
    assert abs(rec_after - rec_static) <= 0.01


def test_json_keys(tmp_path):
    r = bench_json(tmp_path, "keys", "--clients", "2", "--duration", "1")
    (s,) = r["searches"]
    assert LOAD_KEYS <= set(s["extra"])
    assert (s["extra"]["clients"], s["extra"]["duration_s"]) == (2, 1.0)
    assert len(s["ids"]) == len(s["scores"]) == 1000


def test_other_index_rejected(tmp_path):
    out = str(tmp_path / "r.json")
    assert bench.main(["--index", "flat", "--data", DEV, "--out", out, "--clients", "4"]) == 2
