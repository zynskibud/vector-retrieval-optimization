"""Product quantization index (CONTRACT 6.4).

Recall floor (CONTRACT 9, dev set): defaults, recall@10 >= 0.50 (metric=ip and metric=l2).
"""

import numpy as np

BUILD_PARAMS: dict = {"m": 48, "nbits": 8, "metric": "ip", "train_size": 100000, "iters": 20}
SEARCH_PARAMS: dict = {"rerank": 0}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    raise NotImplementedError("not implemented")


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("not implemented")


def index_bytes(index: dict) -> int:
    raise NotImplementedError("not implemented")
