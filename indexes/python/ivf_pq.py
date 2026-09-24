"""IVF with PQ-coded residuals (CONTRACT 6.5).

Recall floor (CONTRACT 9, dev set): defaults, recall@10 >= 0.45 (metric=ip and metric=l2).
"""

import numpy as np

BUILD_PARAMS: dict = {"nlist": 1024, "iters": 20, "train_size": 100000, "m": 48, "nbits": 8, "metric": "ip"}
SEARCH_PARAMS: dict = {"nprobe": 8, "rerank": 0}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    raise NotImplementedError("not implemented")


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("not implemented")


def index_bytes(index: dict) -> int:
    raise NotImplementedError("not implemented")
