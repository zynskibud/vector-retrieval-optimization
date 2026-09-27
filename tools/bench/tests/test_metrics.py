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


def test_summarize_picks_truth_by_filter():
    doc = make_doc()
    unfiltered = doc["searches"][0]
    filtered = {**unfiltered, "search_params": {**unfiltered["search_params"], "filter": "top10"}}
    doc["searches"] = [unfiltered, filtered]
    # The top10 truth equals the returned IDs' first two columns, so that run scores 1.0.
    top10 = np.array([r[:2] for r in unfiltered["ids"]])
    top10[top10 < 0] = 99
    rows = summarize(doc, {"none": GT, "top10": top10}, k=2)
    assert [r["filter"] for r in rows] == ["none", "top10"]
    assert rows[0]["recall@2"] == pytest.approx(0.5)
    assert rows[1]["recall@2"] == pytest.approx(5 / 6)  # one -1 in the doc never matches
    with pytest.raises(ValueError):
        summarize(doc, GT, k=2)  # a filtered run needs its own truth


@pytest.mark.skipif(not (DEV / "filter_top10.npy").exists(), reason="dev filters missing")
def test_faiss_flat_filter_recall_is_one(tmp_path):
    n = 20000
    out = tmp_path / "flat.json"
    subprocess.run([sys.executable, "-m", "tools.bench.faiss_ref", "--index", "flat", "--data", str(DEV),
                    "--out", str(out), "--limit", str(n), "--warmup", "10", "--search", "filter=top10"], check=True)
    doc = json.loads(out.read_text())
    assert validate(doc) == []
    mask = np.load(DEV / "filter_top10.npy")[:n]
    passing = np.flatnonzero(mask)
    run = doc["searches"][0]
    assert run["search_params"]["filter"] == "top10"
    assert run["extra"]["filter_rows"] == len(passing)
    ids = np.array(run["ids"])
    assert mask[ids[ids >= 0]].all()
    x = np.asarray(np.load(DEV / "vectors.npy", mmap_mode="r")[:n])[passing]
    gt = passing[np.argsort(-(np.load(DEV / "queries.npy") @ x.T), axis=1)[:, :10]]
    assert summarize(doc, {"top10": gt})[0]["recall@10"] == 1.0


def test_summarize_load_fields(tmp_path):
    from tools.bench.report import plot_load
    import pandas as pd
    from tools.bench.tests.test_schema import make_load_doc

    doc = make_load_doc()
    doc["searches"][0]["extra"].update({"insert_rate": 1000.0, "inserted_rows": 5})
    gt = np.array([[0, 1], [2, 3], [4, 5]])
    row = summarize(doc, gt, k=2)[0]
    assert row["recall@2"] == 5 / 6  # from worker 0's first-pass ids
    assert row["clients"] == 2 and row["cpu_pct"] == 150.0 and row["errors"] == 0 and row["insert_rate"] == 1000.0
    assert row["p99_ms"] > 0 and row["phase"] == ""
    plain = summarize(make_doc(), gt, k=2)[0]
    assert np.isnan(plain["clients"]) and plain["insert_rate"] == 0
    df = pd.DataFrame([
        {"is_load": True, "filter": "none", "load_line": line, "clients": c, "qps": 100.0 * c, "p99_ms": 1.0 + c}
        for line in ("rust", "rust +inserts") for c in (1, 2, 4)])
    assert plot_load(df, "hnsw", "t", tmp_path / "hnsw-load.png")
    assert (tmp_path / "hnsw-load.png").stat().st_size > 0
    assert not plot_load(df.assign(is_load=False), "hnsw", "t", tmp_path / "x.png")


def _change_doc(**marks):
    doc = make_doc()
    run = doc["searches"][0]
    doc["searches"] = [{**run, "search_params": {**run["search_params"], **marks}}]
    return doc


def test_summarize_picks_truth_by_change():
    """Phase 5 (CONTRACT section 13.2): deleted / updated select ground_truth_<name>."""
    ids = np.array(make_doc()["searches"][0]["ids"])[:, :2]
    exact = np.where(ids < 0, 99, ids)  # the run's own answers: recall 5/6 (one -1)
    truths = {"none": GT, "del30": exact, "upd10": exact}
    row = summarize(_change_doc(deleted="del30", compacted=0), truths, k=2)[0]
    assert row["recall@2"] == pytest.approx(5 / 6) and row["deleted"] == "del30" and row["compacted"] == 0
    row = summarize(_change_doc(updated="upd10", compacted=1), truths, k=2)[0]
    assert row["recall@2"] == pytest.approx(5 / 6) and row["updated"] == "upd10" and row["compacted"] == 1
    assert summarize(make_doc(), truths, k=2)[0]["recall@2"] == pytest.approx(0.5)  # unchanged run: ground_truth.npy
    with pytest.raises(ValueError):
        summarize(_change_doc(deleted="del50"), truths, k=2)  # no truth for del50
    with pytest.raises(ValueError):
        summarize(_change_doc(deleted="del30", filter="top10"), truths, k=2)  # no truth for the combination


def test_deleted_returned():
    from tools.bench.metrics import deleted_returned

    mask = np.zeros(10, dtype=bool)
    mask[[0, 7]] = True
    assert deleted_returned([[0, 1, -1], [7, 7, 3]], mask) == 3  # -1 never counts
    doc = _change_doc(deleted="del30", compacted=0)
    doc["extra"] = {"delete_s": 1.5, "compact_s": 2.0, "disk_bytes": 100, "disk_bytes_after": 60}
    ids = np.array(doc["searches"][0]["ids"])
    row = summarize(doc, {"del30": GT}, k=2, delete_masks={"del30": mask})[0]
    assert row["deleted_returned"] == deleted_returned(ids, mask)
    assert row["delete_s"] == 1.5 and row["compact_s"] == 2.0 and row["disk_bytes_after"] == 60
    assert np.isnan(summarize(doc, {"del30": GT}, k=2)[0]["deleted_returned"])  # no mask given
    assert np.isnan(summarize(make_doc(), GT, k=2)[0]["compacted"])


def test_plot_delete_and_change_cases(tmp_path):
    import pandas as pd

    from tools.bench.report import change_table, plot_delete
    from tools.bench.runner import change_cases

    rows = []
    for frac, name in ((0.0, ""), (0.1, "del10"), (0.3, "del30"), (0.5, "del50")):
        for comp in ((0, 1) if name else (float("nan"),)):
            rows.append({"is_default": True, "filter": "none", "is_load": False, "phase": "", "updated": "",
                         "deleted": name, "compacted": comp, "filter_line": "rust", "deleted_fraction": frac,
                         "recall@10": 0.95 - frac / 10, "p50_ms": 1.0 + frac, "is_change": bool(name)})
    df = pd.DataFrame(rows)
    assert plot_delete(df, "hnsw", "t", tmp_path / "hnsw-delete.png")
    assert (tmp_path / "hnsw-delete.png").stat().st_size > 0
    assert not plot_delete(df[df["deleted"] == ""], "hnsw", "t", tmp_path / "x.png")
    cases = change_cases(["rust", "qdrant"], ["flat", "ivf", "pq", "hnsw"], Path("data/processed/dev"))
    names = sorted(c["out"].name for c in cases)
    assert len(cases) == 3 * 7 + 3 * 7  # rust: flat, ivf, hnsw; qdrant: flat, pq, hnsw; 7 changes each
    assert "chg-rust-hnsw-del30-compact.json" in names and "chg-qdrant-pq-upd10.json" in names
    assert "chg-rust-pq-del10.json" not in names
    cmd = next(c["cmd"] for c in cases if c["out"].name == "chg-rust-ivf-del50-compact.json")
    assert cmd[-3:] == ["--delete", "del50", "--compact"] or cmd[-5:-2] == ["--delete", "del50", "--compact"]
    for col in ("index_bytes", "index_bytes_after", "disk_bytes", "disk_bytes_after", "delete_s", "update_s",
                "compact_s", "deleted_returned", "language", "file"):
        df[col] = 0
    df["index"] = "hnsw"
    assert "del30" in change_table(df)


def test_cache_recall_uses_pool_ids():
    from tools.bench.tests.test_schema import make_cache_doc

    doc = make_cache_doc()  # ids rows [0, 1], [2, 3], [4, -1]; pool ids 0, 7, 1
    gt = np.array([[0, 1], [4, 9]])  # 2 queries: pool 0 and 1; pool 7 is a corpus text
    row = summarize(doc, gt, k=2)[0]
    assert row["recall@2"] == pytest.approx((1.0 + 0.5) / 2)
    assert row["backend"] == "lru" and row["hit_rate"] == 0.25
