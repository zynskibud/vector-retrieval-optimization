import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.bench.metrics import latency_stats, qps, recall_at_k, summarize
from tools.bench.schema import validate
from tools.bench.tests.test_schema import make_doc

DEV = Path("data/processed/dev")
GT = np.array([[0, 1, 9], [2, 5, 9], [7, 8, 9]])  # 3 queries; the known answers for make_doc()


def test_recall_known():
    per_query, mean = recall_at_k(make_doc()["searches"][0]["ids"], GT, 2)
    assert per_query.tolist() == [1.0, 0.5, 0.0]  # -1 never matches
    assert mean == pytest.approx(0.5)


def test_recall_minus_one_not_counted():
    per_query, _ = recall_at_k([[-1, -1]], [[-1, 3]], 2)
    assert per_query.tolist() == [0.0]


def test_latency_and_qps():
    s = latency_stats([1.0, 2.0, 3.0])
    assert s["p50_ms"] == 2.0 and s["mean_ms"] == 2.0
    assert qps([1.0, 1.0]) == pytest.approx(1000.0)


def test_summarize():
    rows = summarize(make_doc(), GT, k=2)
    assert len(rows) == 1
    assert rows[0]["recall@2"] == pytest.approx(0.5)
    assert rows[0]["p50_ms"] == pytest.approx(0.2)


@pytest.mark.skipif(not (DEV / "vectors.npy").exists(), reason="dev data missing")
def test_faiss_flat_recall_is_one(tmp_path):
    out = tmp_path / "flat.json"
    subprocess.run([sys.executable, "-m", "tools.bench.faiss_ref", "--index", "flat", "--data", str(DEV),
                    "--out", str(out), "--limit", "20000", "--warmup", "10"], check=True)
    doc = json.loads(out.read_text())
    assert validate(doc) == []
    # Ground truth for the 20k-row prefix, by brute force on the same rows.
    x = np.load(DEV / "vectors.npy", mmap_mode="r")[:20000]
    gt = np.argsort(-(np.load(DEV / "queries.npy") @ np.asarray(x).T), axis=1)[:, :10]
    assert summarize(doc, gt)[0]["recall@10"] == 1.0
