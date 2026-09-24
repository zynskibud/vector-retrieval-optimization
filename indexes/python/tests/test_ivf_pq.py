import json
import os
import subprocess
import sys

import numpy as np
import pytest

from indexes.python import ivf_pq
from indexes.python.npy import read_npy
from tools.bench import schema

DEV = "data/processed/dev"


def recall(ids, gt):
    return sum(len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt.tolist())) / gt.size


@pytest.fixture(scope="module")
def data():
    return read_npy(f"{DEV}/vectors.npy"), read_npy(f"{DEV}/queries.npy"), read_npy(f"{DEV}/ground_truth.npy")[:, :10]


@pytest.fixture(scope="module", params=["ip", "l2"])
def built(request, data):
    params = dict(ivf_pq.BUILD_PARAMS, metric=request.param)
    return request.param, ivf_pq.build(data[0], params, os.cpu_count(), 42)


def run(index, queries, nprobe, rerank):
    ids, scores, dists = [], [], []
    for q in queries:
        row, sc = ivf_pq.search(index, q, 10, {"nprobe": nprobe, "rerank": rerank})
        assert row.dtype == np.int64 and sc.dtype == np.float32 and len(row) == 10
        ids.append(row)
        scores.append(sc)
        dists.append(index["distance_computations"])
    return np.stack(ids), np.stack(scores), dists


def test_ivf_pq_full_dev(built, data):
    """Full dev set (100k rows), dev ground truth, both metrics."""
    metric, index = built
    vectors, queries, gt = data
    n, nlist = len(vectors), index["nlist"]
    off = index["offsets"]
    assert len(off) == nlist + 1 and off[0] == 0 and off[-1] == n
    assert np.array_equal(np.sort(index["list_ids"]), np.arange(n))
    assert index["codes"].shape == (n, 48) and index["codes"].dtype == np.uint8
    assert ivf_pq.index_bytes(index) == nlist * 384 * 4 + n * 8 + (nlist + 1) * 8 + 48 * 256 * 8 * 4 + n * 48

    ids8, sc8, d8 = run(index, queries, 8, 0)
    ids32, _, _ = run(index, queries, 32, 0)
    idsr, _, dr = run(index, queries, 8, 100)
    r8, r32, rr = recall(ids8, gt), recall(ids32, gt), recall(idsr, gt)
    print(f"metric={metric} nprobe=8 {r8:.4f} nprobe=32 {r32:.4f} nprobe=8,rerank=100 {rr:.4f}")
    assert r8 >= 0.45
    assert r32 >= r8
    assert rr >= r8
    assert all(nlist <= d <= nlist + n for d in d8)
    assert all(a + min(100, a - nlist) == b for a, b in zip(d8, dr))
    if metric == "l2":
        assert (sc8 <= 0).all()


def test_ivf_pq_bench_json(tmp_path):
    out = tmp_path / "ivf_pq.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "ivf_pq", "--data", DEV, "--out", str(out),
           "--limit", "20000", "--build", "metric=l2",
           "--search", "nprobe=8,rerank=0", "--search", "nprobe=8,rerank=100"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert schema.validate(doc) == []
    assert doc["extra"]["id_type"] == "int64"
    assert [s["search_params"]["rerank"] for s in doc["searches"]] == [0, 100]
