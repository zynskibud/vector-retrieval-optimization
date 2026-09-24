"""HNSW graph index (CONTRACT 6.6).

Recall floor (CONTRACT 9, dev set): ef=64, recall@10 >= 0.95.
"""

import numpy as np

BUILD_PARAMS: dict = {"m": 16, "ef_construct": 100}
SEARCH_PARAMS: dict = {"ef": 64}


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    raise NotImplementedError("not implemented")


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("not implemented")


def index_bytes(index: dict) -> int:
    raise NotImplementedError("not implemented")
