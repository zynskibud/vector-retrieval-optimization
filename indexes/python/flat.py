"""Flat index: exact search by scanning every row (CONTRACT 6.1). Recall@10 = 1.0."""

import time

import numpy as np

from . import distance

BUILD_PARAMS: dict = {}
SEARCH_PARAMS: dict = {}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    t0 = time.perf_counter()
    index = {"vectors": vectors, "train_s": 0.0}
    index["add_s"] = time.perf_counter() - t0
    return index


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    vectors = index["vectors"]
    index["distance_computations"] = len(vectors)
    return distance.top_k(distance.scores(query, vectors), k)


def index_bytes(index: dict) -> int:
    return 0
