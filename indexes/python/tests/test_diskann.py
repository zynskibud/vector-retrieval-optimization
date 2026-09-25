"""DiskANN tests (CONTRACT 6.7, 9). Uses the first 20,000 dev rows, with exact top-10 truth
computed by brute force inside the test, because a full 100k Python build is too slow for a test."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from indexes.python import diskann
from indexes.python.npy import read_npy

DEV = "data/processed/dev"
LIMIT = 20000
R = 64


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    vectors = read_npy(f"{DEV}/vectors.npy", LIMIT)
    queries = read_npy(f"{DEV}/queries.npy")
    s = queries @ vectors.T
    ids = np.arange(LIMIT)
    truth = np.stack([ids[np.lexsort((ids, -row))[:10]] for row in s])
    diskann.OUT_PATH = tmp_path_factory.mktemp("diskann") / "r.json"
    try:
        index = diskann.build(vectors, dict(diskann.BUILD_PARAMS), os.cpu_count(), 42)
    finally:
        diskann.OUT_PATH = None
    return queries, truth, index


def run(index, queries, **params):
    p = dict(diskann.SEARCH_PARAMS, **params)
    ids, lat, reads = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        row, _ = diskann.search(index, q, 10, p)
        lat.append(time.perf_counter() - t0)
        ids.append(row)
        reads.append(index["search_extra"]["disk_reads"])
    return np.stack(ids), float(np.median(lat)) * 1000, float(np.mean(reads))


def recall(ids, truth):
    return sum(len(set(a) & set(b)) for a, b in zip(ids.tolist(), truth.tolist())) / truth.size


def test_recall_and_monotonic(data):
    queries, truth, index = data
    r50, r100, r200 = (recall(run(index, queries, l=l)[0], truth) for l in (50, 100, 200))
    print(f"recall@10 l=50 {r50:.4f} l=100 {r100:.4f} l=200 {r200:.4f}")
    assert r100 >= 0.90
    assert r200 >= r100 >= r50


def test_io_modes(data):
    queries, _, index = data
    ids_m, p50_m, _ = run(index, queries, l=100, io="mmap")
    ids_n, p50_n, reads_n = run(index, queries, l=100, io="nocache")
    print(f"p50 mmap {p50_m:.3f} ms, nocache {p50_n:.3f} ms, disk_reads {reads_n:.1f}")
    assert (ids_m == ids_n).all()
    assert reads_n > 0
    assert p50_n > 2 * p50_m  # the mmap run warmed the pages; the switch must evict them


def test_graph_and_file(data):
    _, _, index = data
    assert os.path.getsize(index["path"]) == LIMIT * 4096 == index["extra"]["disk_bytes"]
    mm = np.memmap(index["path"], dtype="<i4", mode="r", shape=(LIMIT, 1024))
    edges = np.array(mm[:, 384 : 384 + R])
    del mm
    deg = (edges >= 0).sum(axis=1)
    assert deg.min() >= 1 and deg.max() <= R
    # Empty slots are only at the end, edges point inside the corpus and never to self.
    assert ((edges >= 0) == (np.arange(R)[None, :] < deg[:, None])).all()
    assert edges.max() < LIMIT
    assert not (edges == np.arange(LIMIT)[:, None]).any()
    print("extra", index["extra"])


def test_corpus_released(data):
    _, _, index = data
    assert "vectors" not in index
    for value in index.values():
        if isinstance(value, np.ndarray):
            assert value.dtype != np.float32 or value.size < LIMIT * 384
    assert diskann.index_bytes(index) == LIMIT * 48 + 48 * 256 * 8 * 4 + 4


def test_search_output(data):
    queries, _, index = data
    ids, scores = diskann.search(index, queries[0], 10, dict(diskann.SEARCH_PARAMS))
    assert ids.dtype == np.int64 and scores.dtype == np.float32 and len(ids) == 10
    assert (np.diff(scores) <= 0).all()
    assert index["distance_computations"] > 0
    assert index["search_extra"]["disk_bytes_read"] == index["search_extra"]["disk_reads"] * 4096


def test_bench_json(tmp_path):
    from tools.bench import schema

    out = tmp_path / "r.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "diskann", "--data", DEV,
           "--out", str(out), "--limit", str(LIMIT), "--warmup", "10",
           "--search", "l=50,io=mmap", "--search", "l=50,io=nocache"]
    subprocess.run(cmd, check=True)
    result = json.loads(out.read_text())
    assert schema.validate(result) == []
    assert result["extra"]["disk_bytes"] == LIMIT * 4096
    assert [s["search_params"]["io"] for s in result["searches"]] == ["mmap", "nocache"]
    for s in result["searches"]:
        assert s["extra"]["disk_reads"] > 0
    assert result["searches"][0]["ids"] == result["searches"][1]["ids"]
