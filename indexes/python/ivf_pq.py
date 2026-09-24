"""IVF with PQ-coded residuals (CONTRACT 6.5).

Recall floor (CONTRACT 9, dev set): defaults, recall@10 >= 0.45 (metric=ip and metric=l2).

Build: coarse centers with the shared k-means (dot product, normalized). Residual r = x - c(x).
Codebooks per sub-vector on the training residuals (pq.train_codebook: seed + j, no
normalization, assignment by `metric`). Encode every corpus row's residual.
Layout (CSR): list_ids int64 and codes (N, m) uint8, both in list order; the rows of list c
are [offsets[c], offsets[c + 1]). IDs ascend inside each list.

Search: the nprobe best centers.
  ip: one table T from q against the residual codebooks; score = q.c + sum_j T[j, code].
  l2: per probed list q' = q - c, T_c[j, k] = -||q'_j - codebook[j][k]||^2; score = sum_j T_c[j, code].
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import distance, kmeans, pq

BUILD_PARAMS: dict = {"nlist": 1024, "iters": 20, "train_size": 100000, "m": 48, "nbits": 8, "metric": "ip"}
SEARCH_PARAMS: dict = {"nprobe": 8, "rerank": 0}

BLOCK = 65536  # rows per step in assignment and encoding, to bound temporary arrays


def _assign_all(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    labels = np.empty(len(vectors), dtype=np.int64)
    for start in range(0, len(vectors), BLOCK):
        labels[start : start + BLOCK] = np.argmax(vectors[start : start + BLOCK] @ centers.T, axis=1)
    return labels


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    n, dim = vectors.shape
    m, metric = pq._check(params, dim)
    sub = dim // m
    nlist = int(params["nlist"])
    iters = int(params["iters"])
    train_n = max(nlist, min(n, int(params["train_size"])))

    # Train: coarse centers, then residual codebooks.
    t0 = time.perf_counter()
    centers = kmeans.kmeans(vectors, nlist, iters=iters, seed=seed, train_size=train_n, metric="ip", normalize=True)
    train = vectors[:train_n]
    train_labels = _assign_all(train, centers)
    resid = train - centers[train_labels]
    codebooks = np.empty((m, 256, sub), dtype=np.float32)

    def train_one(j: int) -> None:
        sub_train = np.ascontiguousarray(resid[:, j * sub : (j + 1) * sub], dtype=np.float32)
        codebooks[j] = pq.train_codebook(sub_train, iters, seed + j, metric)

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        list(pool.map(train_one, range(m)))
    del resid
    t1 = time.perf_counter()

    # Add: assign, encode residuals, fill CSR.
    labels = _assign_all(vectors, centers)
    codes_by_row = np.empty((n, m), dtype=np.uint8)
    for start in range(0, n, BLOCK):
        block = vectors[start : start + BLOCK]
        r = block - centers[labels[start : start + BLOCK]]
        codes_by_row[start : start + len(block)] = pq.encode(r, codebooks, metric)
    counts = np.bincount(labels, minlength=nlist)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    list_ids = np.argsort(labels, kind="stable").astype(np.int64)
    codes = np.ascontiguousarray(codes_by_row[list_ids])
    del codes_by_row
    t2 = time.perf_counter()

    return {
        "vectors": vectors,  # full vectors, only for rerank
        "centers": centers,
        "codebooks": codebooks,
        "codebook_sq": np.einsum("jcs,jcs->jc", codebooks, codebooks),
        "list_ids": list_ids,
        "codes": codes,
        "offsets": offsets,
        "table_offsets": np.arange(m, dtype=np.intp) * 256,
        "nlist": nlist,
        "m": m,
        "metric": metric,
        "train_s": t1 - t0,
        "add_s": t2 - t1,
        "extra": {"id_type": "int64"},
    }


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    nlist, m = index["nlist"], index["m"]
    nprobe = max(1, min(int(params["nprobe"]), nlist))
    offsets, list_ids, codes = index["offsets"], index["list_ids"], index["codes"]
    cb = index["codebooks"]
    sub = cb.shape[2]

    center_scores = distance.scores(query, index["centers"])
    probe, _ = distance.top_k(center_scores, nprobe)

    ranges = [np.arange(offsets[c], offsets[c + 1]) for c in probe]
    rows = np.concatenate(ranges)
    ids = list_ids[rows]
    row_codes = codes[rows] + index["table_offsets"]  # (R, m) flat positions in one (m, 256) table

    if index["metric"] == "ip":
        t = np.einsum("jcs,js->jc", cb, query.reshape(m, sub))
        base = np.repeat(center_scores[probe], [len(r) for r in ranges])
        s = base + np.take(t.ravel(), row_codes).sum(axis=1, dtype=np.float32)
    else:
        qp = (query[None, :] - index["centers"][probe]).reshape(nprobe, m, sub)  # q' per probed list
        dots = np.einsum("pjs,jcs->pjc", qp, cb)
        q_sq = np.einsum("pjs,pjs->pj", qp, qp)[:, :, None]
        tables = -(q_sq - 2 * dots + index["codebook_sq"][None])  # (nprobe, m, 256)
        which = np.repeat(np.arange(nprobe, dtype=np.intp), [len(r) for r in ranges])
        s = np.take(tables.ravel(), row_codes + (which * (m * 256))[:, None]).sum(axis=1, dtype=np.float32)
    s = s.astype(np.float32)

    scanned = len(ids)
    rerank = int(params.get("rerank", 0))
    if rerank <= 0:
        index["distance_computations"] = nlist + scanned
        return distance.top_k(s, k, ids)
    cand, _ = distance.top_k(s, rerank, ids)
    cand = cand[cand >= 0]
    x = index["vectors"][cand]
    if index["metric"] == "ip":
        exact = x @ query
    else:
        diff = x - query
        exact = -np.einsum("ij,ij->i", diff, diff)
    index["distance_computations"] = nlist + scanned + len(cand)
    return distance.top_k(exact.astype(np.float32), k, cand)


def index_bytes(index: dict) -> int:
    return int(index["centers"].nbytes + index["list_ids"].nbytes + index["offsets"].nbytes
               + index["codebooks"].size * 4 + index["codes"].size)
