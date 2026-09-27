"""Phase 5 updates, deletes, and compaction (CONTRACT 13) for flat, ivf, hnsw.

Runs on the first 20,000 dev rows (--limit 20000). The truth over the remaining rows is computed
here by brute force, because ground_truth_del*.npy covers the full dev set. One hnsw build is
shared: each check works on a clone (hnsw.clone), so no check changes the shared build.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from indexes.python import changes, flat, hnsw, ivf
from indexes.python.npy import read_npy
from tools.bench import schema

DEV = "data/processed/dev"
N = 20000
K = 10
IVF = {"nlist": 256, "train_size": N, "iters": 20}
HNSW = {"m": 16, "ef_construct": 100}
PARAMS = {"flat": {"filter": "none"}, "ivf": {"nprobe": 8, "filter": "none"}, "hnsw": {"ef": 64, "filter": "none"}}
MODS = {"flat": flat, "ivf": ivf, "hnsw": hnsw}


def truth(vectors, queries, ids):
    s = queries @ vectors[ids].T
    order = np.lexsort((np.broadcast_to(ids, s.shape), -s), axis=1)[:, :K]
    return ids[order]


def run(mod, index, queries, params):
    return np.stack([mod.search(index, q, K, params)[0] for q in queries])


def recall(ids, gt):
    return sum(len(set(a) & set(b) - {-1}) for a, b in zip(ids.tolist(), gt.tolist())) / gt.size


@pytest.fixture(scope="module")
def data():
    vectors = read_npy(f"{DEV}/vectors.npy", N)
    queries = read_npy(f"{DEV}/queries.npy")
    dead = changes.delete_mask(DEV, "del30", N)
    live = np.flatnonzero(~dead)
    return {
        "vectors": vectors,
        "queries": queries,
        "dead": dead,
        "live": live,
        "gt_all": truth(vectors, queries, np.arange(N)),
        "gt_del": truth(vectors, queries, live),
    }


@pytest.fixture(scope="module")
def base(data):
    """One build per index on all N rows, never changed by a test, plus its plain recall."""
    v = data["vectors"]
    out = {"flat": flat.build(v, {}, 1, 42), "ivf": ivf.build(v, dict(IVF), 1, 42), "hnsw": hnsw.build(v, dict(HNSW), 1, 42)}
    rec = {name: recall(run(MODS[name], idx, data["queries"], PARAMS[name]), data["gt_all"]) for name, idx in out.items()}
    return out, rec


def fresh(name, base_index):
    """An independent copy of the base build (cheap: no rebuild)."""
    if name == "hnsw":
        return hnsw.clone(base_index)
    return {k: v for k, v in base_index.items() if k not in ("filtered", "filtered_rows")}


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_delete(data, base, name):
    mod = MODS[name]
    index = mod.delete(fresh(name, base[0][name]), data["dead"])
    ids = run(mod, index, data["queries"], PARAMS[name])
    returned = ids[ids >= 0]
    assert not data["dead"][returned].any(), "a deleted ID was returned"
    r = recall(ids, data["gt_del"])
    if name == "flat":
        assert r == 1.0
    assert r >= base[1][name] - 0.03, (name, r, base[1][name])
    assert mod.index_bytes(index) >= changes.tombstone_bytes(N)


@pytest.mark.parametrize("name", ["flat", "ivf", "hnsw"])
def test_update(data, base, name):
    mod = MODS[name]
    ids, vecs = changes.update_set(DEV, "upd10", N)
    assert len(ids) > 1000
    index = mod.update(fresh(name, base[0][name]), ids, vecs)
    assert np.array_equal(data["vectors"][:5], read_npy(f"{DEV}/vectors.npy", 5)), "caller's array was written"
    pick = np.random.default_rng(0).choice(len(ids), 100, replace=False)
    top1 = [int(mod.search(index, vecs[j], K, PARAMS[name])[0][0]) for j in pick]
    assert top1 == ids[pick].tolist()


@pytest.mark.parametrize("name", ["flat", "ivf"])
def test_compact_flat_ivf(data, name):
    mod = MODS[name]
    v = data["vectors"]
    build = (lambda x: flat.build(x, {}, 1, 42)) if name == "flat" else (lambda x: ivf.build(x, {**IVF, "train_size": len(x)}, 1, 42))
    index = mod.delete(build(v), data["dead"])
    before = mod.index_bytes(index)
    index = mod.compact(index, "rebuild")
    after = mod.index_bytes(index)
    assert after < before, (before, after)
    ids = run(mod, index, data["queries"], PARAMS[name])
    assert not data["dead"][ids[ids >= 0]].any()
    r = recall(ids, data["gt_del"])
    # A fresh build on the remaining rows only (IDs mapped back to row IDs).
    live = data["live"]
    f = build(np.ascontiguousarray(v[live]))
    fr = recall(live[np.maximum(run(mod, f, data["queries"], PARAMS[name]), 0)], data["gt_del"])
    assert abs(r - fr) <= 0.01, (name, r, fr)


def test_compact_hnsw(data, base):
    """Rebuild: the rebuild is a fresh build on the remaining rows by construction (hnsw.build on
    the live rows in row order, same params and seed; levels redrawn), so a second fresh build
    would give identical output; the test checks the construction instead of paying for it.
    Repair mode: no deleted ID, and recall within 0.03 of the rebuild."""
    live = data["live"]
    rebuilt = hnsw.compact(hnsw.delete(hnsw.clone(base[0]["hnsw"]), data["dead"]), "rebuild")
    ids = run(hnsw, rebuilt, data["queries"], PARAMS["hnsw"])
    assert not data["dead"][ids[ids >= 0]].any()
    fr = recall(ids, data["gt_del"])
    g = rebuilt["graph"]
    assert g.n == len(live) and np.array_equal(rebuilt["id_map"], live)
    assert np.array_equal(g.levels, hnsw.draw_levels(len(live), HNSW["m"], 42))
    assert np.array_equal(g.v, data["vectors"][live])
    assert fr >= base[1]["hnsw"] - 0.03, (fr, base[1]["hnsw"])

    repaired = hnsw.delete(hnsw.clone(base[0]["hnsw"]), data["dead"])
    repaired = hnsw.compact(repaired, "repair")
    assert "deleted" not in repaired
    ids = run(hnsw, repaired, data["queries"], PARAMS["hnsw"])
    assert not data["dead"][ids[ids >= 0]].any()
    rr = recall(ids, data["gt_del"])
    assert rr >= fr - 0.03, (rr, fr)


def test_bench_delete_json(tmp_path):
    out = tmp_path / "r.json"
    cmd = [sys.executable, "-m", "indexes.python.bench", "--index", "hnsw", "--data", DEV, "--out", str(out),
           "--limit", str(N), "--threads", "1", "--warmup", "10", "--delete", "del30", "--search", "ef=64"]
    subprocess.run(cmd, check=True)
    doc = json.loads(out.read_text())
    assert schema.validate(doc) == []
    (s,) = doc["searches"]
    assert s["search_params"]["deleted"] == "del30" and s["search_params"]["compacted"] == 0
    assert doc["extra"]["deleted_rows"] == int(changes.delete_mask(DEV, "del30", N).sum())
    assert doc["extra"]["delete_s"] >= 0


def test_bench_change_usage(tmp_path):
    from indexes.python import bench
    out = str(tmp_path / "r.json")
    base = ["--data", DEV, "--out", out, "--limit", "1000"]
    assert bench.main(["--index", "pq", *base, "--delete", "del10"]) == 2
    assert bench.main(["--index", "flat", *base, "--delete", "del10", "--update", "upd10"]) == 2
    assert bench.main(["--index", "flat", *base, "--delete", "del99"]) == 2
    assert bench.main(["--index", "flat", *base, "--compact"]) == 2
    assert bench.main(["--index", "flat", *base, "--delete", "del10", "--compact", "--compact-mode", "x"]) == 2
