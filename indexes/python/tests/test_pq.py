import json
import os
import subprocess
import sys

import numpy as np
import pytest

from indexes.python import pq
from indexes.python.npy import read_npy
from tools.bench import schema

DEV = "data/processed/dev"


@pytest.fixture(scope="module")
def data():
    return read_npy(f"{DEV}/vectors.npy"), read_npy(f"{DEV}/queries.npy"), read_npy(f"{DEV}/ground_truth.npy")[:, :10]


@pytest.fixture(scope="module", params=["ip", "l2"])
def built(request, data):
    vectors = data[0]
    params = dict(pq.BUILD_PARAMS, metric=request.param)
    return request.param, pq.build(vectors, params, os.cpu_count(), 42)


def run(index, queries, rerank):
    res = [pq.search(index, q, 10, {"rerank": rerank}) for q in queries]
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def recall(ids, gt):
    return sum(len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt.tolist())) / gt.size


def test_pq_recall_rerank_scores(built, data):
    metric, index = built
    vectors, queries, gt = data
    assert index["codes"].shape == (len(vectors), 48)
    assert index["codes"].dtype == np.uint8 and index["codes"].flags["C_CONTIGUOUS"]
    assert pq.index_bytes(index) == 48 * 256 * 8 * 4 + len(vectors) * 48

    ids0, scores0 = run(index, queries, 0)
    assert index["distance_computations"] == len(vectors)
    r0 = recall(ids0, gt)
    ids1, _ = run(index, queries, 100)
    assert index["distance_computations"] == len(vectors) + 100
    r1 = recall(ids1, gt)
    print(f"metric={metric} recall rerank=0 {r0:.4f} rerank=100 {r1:.4f}")
    assert r0 >= 0.50
    assert r1 >= r0
    if metric == "ip":
        # The table score must equal q . decode(code) exactly (up to float error).
        top = ids0[:, 0]
        cb = index["codebooks"]
        decoded = cb[np.arange(48), index["codes"][top]].reshape(len(top), -1)
        assert np.allclose(scores0[:, 0], np.einsum("ij,ij->i", decoded, queries), atol=1e-4)
        # PQ under-estimates the dot product (centroids are cluster means, which are shorter
        # than the points); the mean error for the top result stays below 0.2.
        true_top = np.einsum("ij,ij->i", vectors[top], queries)
        err = np.abs(scores0[:, 0] - true_top)
        print(f"ip top-1 |score - q.x|: mean {err.mean():.4f} max {err.max():.4f}")
        assert err.mean() < 0.2
    else:
        assert (scores0 <= 0).all()


def test_pq_rejects_bad_params(data):
    v = data[0][:1000]
    with pytest.raises(ValueError):
        pq.build(v, dict(pq.BUILD_PARAMS, nbits=4), 1, 42)
    with pytest.raises(ValueError):
        pq.build(v, dict(pq.BUILD_PARAMS, metric="cos"), 1, 42)


def test_pq_bench_json(tmp_path):
    out = tmp_path / "pq.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "pq", "--data", DEV, "--out", str(out),
           "--limit", "20000", "--build", "metric=l2", "--search", "rerank=0", "--search", "rerank=100"]
    subprocess.run(cmd, check=True)
    doc = json.loads(out.read_text())
    schema.validate(doc)
    assert [s["search_params"]["rerank"] for s in doc["searches"]] == [0, 100]
