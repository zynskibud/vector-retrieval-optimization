"""Phase 3 filtered search (CONTRACT 11) for flat, ivf, hnsw.

Runs on the first 20,000 dev rows (--limit 20000). The filtered truth is computed here by brute
force over the passing rows of the cut mask, because ground_truth_<name>.npy covers the full dev set.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from indexes.python import flat, hnsw, ivf
from indexes.python.npy import read_npy
from tools.bench import schema

DEV = "data/processed/dev"
N = 20000
K = 10


def recall(ids, gt):
    return sum(len(set(a) & set(b) - {-1}) for a, b in zip(ids.tolist(), gt.tolist())) / gt.size


@pytest.fixture(scope="module")
def data():
    vectors = read_npy(f"{DEV}/vectors.npy", N)
    queries = read_npy(f"{DEV}/queries.npy")
    for mod in (flat, ivf, hnsw):
        mod.DATA_DIR = DEV
    masks = {name: np.load(f"{DEV}/filter_{name}.npy")[:N] for name in ("top10", "top01")}
    truth = {}
    for name, m in masks.items():
        ids = np.flatnonzero(m)
        s = queries @ vectors[ids].T
        order = np.lexsort((np.broadcast_to(ids, s.shape), -s), axis=1)[:, :K]
        truth[name] = ids[order]
    return vectors, queries, masks, truth


@pytest.fixture(scope="module")
def indexes(data):
    vectors = data[0]
    return {
        "flat": flat.build(vectors, {}, 1, 42),
        "ivf": ivf.build(vectors, {"nlist": 256, "train_size": N, "iters": 20}, 1, 42),
        "hnsw": hnsw.build(vectors, dict(hnsw.BUILD_PARAMS), 1, 42),
    }


MODS = {"flat": flat, "ivf": ivf, "hnsw": hnsw}
FLOOR = {"flat": 1.0, "ivf": 0.70, "hnsw": 0.85}


def run(mod, index, queries, params):
    ids = np.stack([mod.search(index, q, K, params)[0] for q in queries])
    return ids, dict(index["search_extra"])


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_filter(name, data, indexes):
    vectors, queries, masks, truth = data
    mod, index = MODS[name], indexes[name]
    base = dict(mod.SEARCH_PARAMS)
    assert base["filter"] == "none"
    for f in ("top10", "top01"):
        ids, extra = run(mod, index, queries, {**base, "filter": f})
        assert extra["filter_rows"] == int(masks[f].sum())
        flat_ids = ids.ravel()
        assert np.all((flat_ids == -1) | masks[f][np.maximum(flat_ids, 0)]), f"{name} {f}: failing id returned"
        r = recall(ids, truth[f])
        print(f"{name} filter={f} recall@10={r:.4f} extra={extra}")
        if f == "top10":
            assert r >= FLOOR[name], f"{name} top10 recall {r}"
    # filter=none gives the same results as a search without the key (the pre-Phase-3 path).
    ids_none, _ = run(mod, index, queries[:100], {**base, "filter": "none"})
    old = {k: v for k, v in base.items() if k != "filter"}
    ids_old, _ = run(mod, index, queries[:100], old)
    assert np.array_equal(ids_none, ids_old)


def test_bad_filter_name(indexes, data):
    with pytest.raises(ValueError):
        flat.search(indexes["flat"], data[1][0], K, {"filter": "top7"})


def test_bench_hnsw_filters(tmp_path):
    out = tmp_path / "r.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "hnsw", "--data", DEV, "--out", str(out),
           "--limit", "5000", "--warmup", "5",
           "--search", "filter=none", "--search", "filter=top10", "--search", "filter=top01"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert schema.validate(doc) == []
    assert [s["search_params"]["filter"] for s in doc["searches"]] == ["none", "top10", "top01"]
    for s in doc["searches"]:
        assert "filter_rows" in s["extra"] and "visited" in s["extra"]


@pytest.mark.parametrize("args", [["--index", "hnsw", "--search", "filter=top7"],
                                  ["--index", "pq", "--search", "filter=top10"],
                                  ["--index", "diskann", "--search", "filter=top10"]])
def test_bench_rejects(args, tmp_path):
    cmd = [sys.executable, "-m", "indexes.python.bench", "--data", DEV, "--out", str(tmp_path / "x.json"), *args]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 2
