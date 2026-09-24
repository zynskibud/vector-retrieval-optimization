"""Inverted file index (CONTRACT 6.3).

Recall floor (CONTRACT 9, dev set): nprobe=8, recall@10 >= 0.75.

In build_params, train_size None means the k-means default min(N, 256 * nlist).
bench fills it in before build (6.2).

Layout (CSR): list_ids is one contiguous int64 array of vector IDs, grouped by list.
The IDs of list c are list_ids[offsets[c] : offsets[c + 1]], in ascending ID order.
"""

import time

import numpy as np

from . import distance, kmeans

BUILD_PARAMS: dict = {"nlist": 1024, "train_size": None, "iters": 20}
SEARCH_PARAMS: dict = {"nprobe": 8}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    nlist = int(params["nlist"])
    iters = int(params["iters"])
    train_size = params.get("train_size")
    if train_size is None:
        train_size = kmeans.default_train_size(len(vectors), nlist)

    t0 = time.perf_counter()
    centers = kmeans.kmeans(vectors, nlist, iters=iters, seed=seed, train_size=int(train_size),
                            metric="ip", normalize=True)
    t1 = time.perf_counter()

    # Add: assign every corpus row to its best center, in blocks to bound the (block, nlist) matrix.
    n = len(vectors)
    labels = np.empty(n, dtype=np.int64)
    block = 65536
    for start in range(0, n, block):
        labels[start : start + block] = np.argmax(vectors[start : start + block] @ centers.T, axis=1)
    counts = np.bincount(labels, minlength=nlist)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    list_ids = np.argsort(labels, kind="stable").astype(np.int64)  # stable: IDs ascend in each list
    t2 = time.perf_counter()

    return {
        "vectors": vectors,
        "centers": centers,
        "list_ids": list_ids,
        "offsets": offsets,
        "nlist": nlist,
        "train_s": t1 - t0,
        "add_s": t2 - t1,
        "extra": {"id_type": "int64"},
    }


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    nlist = index["nlist"]
    nprobe = max(1, min(int(params["nprobe"]), nlist))
    offsets = index["offsets"]
    list_ids = index["list_ids"]

    center_scores = distance.scores(query, index["centers"])
    # The nprobe best centers; ties to the lower center index (same rule as for IDs).
    probe, _ = distance.top_k(center_scores, nprobe)

    ids = np.concatenate([list_ids[offsets[c] : offsets[c + 1]] for c in probe])
    index["distance_computations"] = nlist + len(ids)
    return distance.top_k(distance.scores(query, index["vectors"][ids]), k, ids)


def index_bytes(index: dict) -> int:
    return int(index["centers"].nbytes + index["list_ids"].nbytes)
