import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.bench.metrics import summarize
from tools.bench.schema import validate

DEV = Path("data/processed/dev")


@pytest.mark.skipif(not (DEV / "vectors.npy").exists(), reason="dev data missing")
def test_bench_small_run(tmp_path):
    """300 requests of the zipf workload on the dev set, lru capacity 50, invalidation at 200."""
    out = tmp_path / "c.json"
    subprocess.run([sys.executable, "-m", "tools.cache.bench", "--data", str(DEV), "--out", str(out),
                    "--workload", "zipf", "--backend", "lru", "--capacity", "50", "--requests", "300",
                    "--invalidate-at", "200", "--workload-dir", str(tmp_path), "--warmup", "2"], check=True)
    doc = json.loads(out.read_text())
    assert validate(doc) == []
    run = doc["searches"][0]
    ex = run["extra"]
    assert doc["q"] == 300 and len(run["latency_ms"]) == 300
    assert 0 < ex["hit_rate"] < 1 and ex["entries"] <= 50 and "hit_rate_after_invalidate" in ex
    row = summarize(doc, np.load(DEV / "ground_truth.npy"))[0]
    if ex["query_requests"]:
        assert ex["recall_on_queries"] >= 0.9
        assert abs(row["recall@10"] - ex["recall_on_queries"]) < 1e-9  # all 300 rows are in ids
