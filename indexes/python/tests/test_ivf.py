import json
import subprocess
import sys

import numpy as np
import pytest

from indexes.python import ivf, kmeans
from indexes.python.npy import read_npy
from tools.bench import schema

DEV = "data/processed/dev"


def recall(ids: np.ndarray, gt: np.ndarray) -> float:
    return sum(len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt.tolist())) / gt.size


@pytest.fixture(scope="module")
def dev():
    vectors = read_npy(f"{DEV}/vectors.npy")
    queries = read_npy(f"{DEV}/queries.npy")
    gt = read_npy(f"{DEV}/ground_truth.npy")[:, :10]
    params = dict(ivf.BUILD_PARAMS)
    params["train_size"] = kmeans.default_train_size(len(vectors), params["nlist"])
    index = ivf.build(vectors, params, 1, 42)
    return vectors, queries, gt, index


def run(index, queries, nprobe):
    ids, dists = [], []
    for q in queries:
        row, scores = ivf.search(index, q, 10, {"nprobe": nprobe})
        assert row.dtype == np.int64 and scores.dtype == np.float32 and len(row) == 10
        ids.append(row)
        dists.append(index["distance_computations"])
    return np.stack(ids), dists


def test_ivf_recall_and_invariants(dev):
    vectors, queries, gt, index = dev
    n, nlist = len(vectors), index["nlist"]
    assert index["offsets"][-1] == n and len(index["offsets"]) == nlist + 1
    assert np.array_equal(np.sort(index["list_ids"]), np.arange(n))
    assert ivf.index_bytes(index) == nlist * vectors.shape[1] * 4 + n * 8

    ids8, d8 = run(index, queries, 8)
    ids64, _ = run(index, queries, 64)
    r8, r64 = recall(ids8, gt), recall(ids64, gt)
    print(f"recall@10 nprobe=8: {r8:.4f}, nprobe=64: {r64:.4f}")
    assert r8 >= 0.75
    assert r64 >= r8
    for ids in (ids8, ids64):
        assert ((ids == -1) | ((ids >= 0) & (ids < n))).all()
    assert all(nlist <= d <= nlist + n for d in d8)


def test_ivf_bench_json(tmp_path):
    out = tmp_path / "ivf.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "ivf", "--data", DEV,
           "--out", str(out), "--limit", "20000", "--search", "nprobe=4", "--search", "nprobe=16"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert schema.validate(doc) == []
    assert doc["build_params"]["train_size"] == min(20000, 256 * 1024)
    assert [s["search_params"]["nprobe"] for s in doc["searches"]] == [4, 16]
    assert doc["extra"]["id_type"] == "int64"
