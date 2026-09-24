"""Product quantization index (CONTRACT 6.4).

Recall floor (CONTRACT 9, dev set): defaults, recall@10 >= 0.50 (metric=ip and metric=l2).

Build: split each vector into m sub-vectors of dim/m values. Train one codebook of 256
centroids per sub-vector with the shared k-means (seed + j, no normalization, assignment
by `metric`). Encode every row as m uint8 codes (the index of the best centroid).
Search: build a table T (m, 256) from the query, score row i as sum_j T[j, code[i, j]].
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import distance, kmeans

BUILD_PARAMS: dict = {"m": 48, "nbits": 8, "metric": "ip", "train_size": 100000, "iters": 20}
SEARCH_PARAMS: dict = {"rerank": 0}

ENCODE_CHUNK = 65536  # rows encoded per step, to bound the (rows, 256) score matrix


def _check(params: dict, dim: int) -> tuple[int, str]:
    m, nbits, metric = int(params["m"]), params["nbits"], params["metric"]
    if nbits != 8:
        raise ValueError(f"pq: nbits={nbits} is not supported, only 8")
    if metric not in ("ip", "l2"):
        raise ValueError(f"pq: metric={metric!r} is not supported, want ip or l2")
    if m <= 0 or dim % m != 0:
        raise ValueError(f"pq: dim {dim} is not divisible by m={m}")
    return m, metric


def train_codebook(train: np.ndarray, iters: int, seed: int, metric: str) -> np.ndarray:
    """CONTRACT 6.4: the shared k-means with k=256, assignment by `metric`, no normalization."""
    return kmeans.kmeans(train, 256, iters=iters, seed=seed, train_size=len(train), metric=metric, normalize=False)


def encode(vectors: np.ndarray, codebooks: np.ndarray, metric: str) -> np.ndarray:
    """Return codes (N, m) uint8: best centroid per sub-vector under the metric."""
    n, dim = vectors.shape
    m, _, sub = codebooks.shape
    codes = np.empty((n, m), dtype=np.uint8)
    for start in range(0, n, ENCODE_CHUNK):
        block = vectors[start : start + ENCODE_CHUNK]
        for j in range(m):
            labels, _ = kmeans.assign(block[:, j * sub : (j + 1) * sub], codebooks[j], metric)
            codes[start : start + len(block), j] = labels
    return codes


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    n, dim = vectors.shape
    m, metric = _check(params, dim)
    sub = dim // m
    train_n = min(n, int(params["train_size"]))

    t0 = time.perf_counter()
    train = vectors[:train_n]
    codebooks = np.empty((m, 256, sub), dtype=np.float32)

    def train_one(j: int) -> None:
        sub_train = np.ascontiguousarray(train[:, j * sub : (j + 1) * sub], dtype=np.float32)
        codebooks[j] = train_codebook(sub_train, int(params["iters"]), seed + j, metric)

    # The m codebooks are independent (own seed, own slice), so the result does not depend
    # on the thread count. NumPy releases the GIL in its inner loops.
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        list(pool.map(train_one, range(m)))
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    codes = encode(vectors, codebooks, metric)
    add_s = time.perf_counter() - t0

    return {
        "vectors": vectors,  # full vectors, only for rerank
        "codebooks": codebooks,
        "codebook_sq": np.einsum("jcs,jcs->jc", codebooks, codebooks),  # ||c||^2 for l2 tables
        "codes": codes,
        "offsets": np.arange(m, dtype=np.intp) * 256,
        "m": m,
        "metric": metric,
        "train_s": train_s,
        "add_s": add_s,
    }


def table(index: dict, query: np.ndarray) -> np.ndarray:
    """T (m, 256): ip -> q_j . c; l2 -> -||q_j - c||^2 (CONTRACT 6.4.1)."""
    cb = index["codebooks"]
    m, _, sub = cb.shape
    qs = query.reshape(m, sub)
    dots = np.einsum("jcs,js->jc", cb, qs)
    if index["metric"] == "ip":
        return dots
    q_sq = np.einsum("js,js->j", qs, qs)[:, None]
    return -(q_sq - 2 * dots + index["codebook_sq"])


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    codes = index["codes"]
    n, m = codes.shape
    t = table(index, query)
    # s[i] = sum_j T[j, codes[i, j]]: flat index j*256 + code into the raveled table.
    s = np.take(t.ravel(), codes + index["offsets"]).sum(axis=1, dtype=np.float32)  # (N,)
    rerank = int(params.get("rerank", 0))
    if rerank <= 0:
        index["distance_computations"] = n
        return distance.top_k(s, k)
    cand, _ = distance.top_k(s, rerank)
    cand = cand[cand >= 0]
    x = index["vectors"][cand]
    if index["metric"] == "ip":
        exact = x @ query
    else:
        diff = x - query
        exact = -np.einsum("ij,ij->i", diff, diff)
    index["distance_computations"] = n + len(cand)
    return distance.top_k(exact.astype(np.float32), k, cand)


def index_bytes(index: dict) -> int:
    return int(index["codebooks"].size * 4 + index["codes"].size)
