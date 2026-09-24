"""DiskANN / Vamana graph index with on-disk records (CONTRACT 6.7).

Recall floor (CONTRACT 9, dev set): l=100, recall@10 >= 0.90 (metric=ip and metric=l2).
"""

import numpy as np

BUILD_PARAMS: dict = {"r": 64, "l_build": 100, "alpha": 1.2, "pq_m": 48, "metric": "ip"}
SEARCH_PARAMS: dict = {"l": 100, "beam": 4, "rerank": 100, "io": "mmap"}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    raise NotImplementedError("not implemented")


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("not implemented")


def index_bytes(index: dict) -> int:
    raise NotImplementedError("not implemented")
