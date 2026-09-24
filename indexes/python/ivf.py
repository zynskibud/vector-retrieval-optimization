"""Inverted file index (CONTRACT 6.3).

Recall floor (CONTRACT 9, dev set): nprobe=8, recall@10 >= 0.80.

In build_params, train_size None means the k-means default min(N, 256 * nlist).
"""

import numpy as np

BUILD_PARAMS: dict = {"nlist": 1024, "train_size": None, "iters": 20}
SEARCH_PARAMS: dict = {"nprobe": 8}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    raise NotImplementedError("not implemented")


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("not implemented")


def index_bytes(index: dict) -> int:
    raise NotImplementedError("not implemented")
