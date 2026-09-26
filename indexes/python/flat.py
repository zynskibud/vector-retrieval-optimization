"""Flat index: exact search by scanning every row (CONTRACT 6.1). Recall@10 = 1.0."""

import time

import numpy as np

from . import distance, filters

BUILD_PARAMS: dict = {}
SEARCH_PARAMS: dict = {"filter": "none"}
DATA_DIR = None  # set by bench; filter_<name>.npy lives here (CONTRACT 11)


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    t0 = time.perf_counter()
    index = {"vectors": vectors, "train_s": 0.0}
    index["add_s"] = time.perf_counter() - t0
    return index


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    vectors = index["vectors"]
    name = params.get("filter", "none")
    if filters.mask(DATA_DIR, name, len(vectors)) is None:
        index["distance_computations"] = len(vectors)
        index["search_extra"] = {"filter_rows": len(vectors)}
        return distance.top_k(distance.scores(query, vectors), k)
    # Filtered: gather the passing rows once per filter (a copy, cached in the index), then scan them.
    cache = index.setdefault("filtered", {})
    if name not in cache:
        ids = np.flatnonzero(filters.mask(DATA_DIR, name, len(vectors))).astype(np.int64)
        cache[name] = (ids, np.ascontiguousarray(vectors[ids]))
    ids, rows = cache[name]
    index["distance_computations"] = len(ids)
    index["search_extra"] = {"filter_rows": len(ids)}
    return distance.top_k(distance.scores(query, rows), k, ids)


def index_bytes(index: dict) -> int:
    return 0
