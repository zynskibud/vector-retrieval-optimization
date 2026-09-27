"""Shared Phase 5 check for the database tests (CONTRACT section 13.5).

On the first 20,000 rows of the dev set, with the truth computed here by brute force:
  del30: no deleted ID is returned, and recall@10 against the remaining-rows truth is at most
         0.03 below the recall of the same index before the delete (a higher recall passes);
  compact: disk_bytes is reported before and after, no deleted ID is returned, and recall is
         within 0.01 of a fresh build on the remaining rows (CONTRACT 13.5; the fresh build loads
         only the remaining rows, with IDs 0..M-1 that the test maps back to row IDs);
  upd10: for 100 sampled updated rows, a query equal to the new vector returns that row as top-1.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.db import base

DATA = Path("data/processed/dev")
N = 20000
K = 10


def _truth(queries, vectors, rows):
    return rows[np.argsort(-(queries @ vectors[rows].T), axis=1, kind="stable")[:, :K]]


def _search(client, queries, params, mask=None):
    """(recall inputs: list of ID lists, number of returned IDs in mask)."""
    out, bad = [], 0
    for q in queries:
        ids, _ = client.search(q, K, params)
        ids = [i for i in ids if i >= 0]
        if mask is not None:
            bad += int(mask[ids].sum()) if ids else 0
        out.append(ids)
    return out, bad


def _recall(got, truth) -> float:
    return sum(len(set(g[:K]) & set(t.tolist())) for g, t in zip(got, truth)) / (len(truth) * K)


def check(client, index: str = "hnsw", build: dict | None = None, search: dict | None = None) -> None:
    build = {**base.BUILD_DEFAULTS[index], **(build or {})}
    search = search if search is not None else {"ef": 64}
    vectors = np.ascontiguousarray(np.load(DATA / "vectors.npy", mmap_mode="r")[:N])
    queries = np.load(DATA / "queries.npy")
    meta = pq.read_table(DATA / "metadata.parquet", columns=list(base.META_COLUMNS)).slice(0, N)
    client.data_dir = DATA

    def fresh():
        client.reset()
        client.load(vectors, meta, 2000)
        client.build_index(index, build)

    # del30, then compact.
    fresh()
    r0 = _recall(_search(client, queries, search)[0], _truth(queries, vectors, np.arange(N)))
    ids = base.delete_ids(DATA, "del30", N)
    mask = np.zeros(N, dtype=bool)
    mask[ids] = True
    before = client.stats()
    delete_s = client.delete(ids)
    truth_del = _truth(queries, vectors, np.flatnonzero(~mask))
    got, bad = _search(client, queries, search, mask)
    r_del = _recall(got, truth_del)
    changed = client.stats()
    detail = client.compact()
    after = client.stats()
    got, bad_c = _search(client, queries, search, mask)
    r_c = _recall(got, truth_del)
    keep = np.flatnonzero(~mask)
    client.reset()
    client.load(np.ascontiguousarray(vectors[keep]), meta.take(keep), 2000)
    client.build_index(index, build)
    got_f = [[int(keep[i]) for i in g] for g in _search(client, queries, search)[0]]
    r_f = _recall(got_f, truth_del)
    keys = ("rows", "disk_bytes", "index_bytes", "segments_count", "deleted_vectors", "dead_tuples")
    show = lambda st: {k: st[k] for k in keys if k in st}
    print(f"\n{client.name} {index} {search}: recall before={r0:.4f} del30={r_del:.4f} (deleted returned {bad}, "
          f"delete_s={delete_s:.2f}) compacted={r_c:.4f} (deleted returned {bad_c}, compact_s={detail['compact_s']:.2f}) "
          f"fresh build on remaining rows={r_f:.4f}")
    print(f"  stats before: {show(before)}\n  after delete: {show(changed)}\n  after compact: {show(after)}")
    assert bad == 0 and bad_c == 0
    assert r0 - r_del <= 0.03, (r0, r_del)
    assert abs(r_c - r_f) <= 0.01, (r_c, r_f)
    assert before.get("disk_bytes", 0) > 0 and after.get("disk_bytes", 0) > 0, (before, after)

    # upd10: the new vector finds its row as top-1.
    fresh()
    uids, uvecs = base.update_rows(DATA, "upd10", N)
    update_s = client.update(uids, uvecs, meta.take(uids))
    pick = np.random.default_rng(0).choice(len(uids), size=100, replace=False)
    top1 = sum(int(client.search(uvecs[j], K, search)[0][0] == uids[j]) for j in pick)
    print(f"  upd10: {len(uids)} rows updated in {update_s:.2f}s; top-1 for {top1} of 100 sampled rows")
    assert top1 == 100, top1
    client.reset()
