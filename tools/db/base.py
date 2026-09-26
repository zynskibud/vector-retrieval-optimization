"""Client interface for the database benches (tools/db/README.md) and shared helpers."""

from __future__ import annotations

import importlib
import time
from typing import Protocol

import numpy as np
import pyarrow as pa

DATABASES = ("qdrant", "pgvector", "milvus")

# Index types each database can run (README, parameter mapping).
SUPPORTED = {
    "qdrant": {"flat", "pq", "hnsw"},
    "pgvector": {"flat", "ivf", "hnsw"},
    "milvus": {"flat", "ivf", "ivf_pq", "hnsw", "diskann"},
}

# Metadata columns loaded next to each vector, for the Phase 3 filters.
META_COLUMNS = ("views", "title", "wiki_id", "paragraph_id", "langs")

# Contract defaults (indexes/CONTRACT.md section 6). Clients map these names.
BUILD_DEFAULTS = {
    "flat": {},
    "ivf": {"nlist": 1024, "iters": 20},
    "pq": {"m": 48, "nbits": 8, "metric": "ip", "quant": "product"},
    "ivf_pq": {"nlist": 1024, "m": 48, "nbits": 8, "metric": "ip"},
    "hnsw": {"m": 16, "ef_construct": 100, "quant": "none"},
    "diskann": {"r": 64, "l_build": 100},
}
SEARCH_DEFAULTS = {
    "flat": {},
    "ivf": {"nprobe": 8},
    "pq": {"rerank": 0},
    "ivf_pq": {"nprobe": 8, "rerank": 0},
    "hnsw": {"ef": 64, "rescore": 0},
    "diskann": {"l": 100, "beam": 4},
}


class Client(Protocol):
    """One database. Host names are the compose service names: qdrant, pgvector, milvus."""

    name: str

    def connect(self) -> None: ...
    def reset(self) -> None: ...
    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float: ...
    def build_index(self, index: str, params: dict) -> float: ...
    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...


def get_client(name: str) -> Client:
    if name not in DATABASES:
        raise ValueError(f"unknown database {name!r}; known: {DATABASES}")
    return importlib.import_module(f"tools.db.{name}").make_client()


def timed(fn, *args, **kwargs) -> tuple[float, object]:
    """Return (seconds, result) of one call."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def wait_until(check, timeout_s: float, every_s: float = 0.5, what: str = "condition") -> None:
    """Poll `check()` until it returns True or `timeout_s` passes."""
    t0 = time.perf_counter()
    while not check():
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"{what} not reached within {timeout_s:.0f}s")
        time.sleep(every_s)


def pad(ids: list[int], scores: list[float], k: int) -> tuple[list[int], list[float | None]]:
    """Pad a short result to k with -1 and null (CONTRACT section 6)."""
    ids = list(ids)[:k]
    scores = list(scores)[:k]
    return ids + [-1] * (k - len(ids)), [float(s) for s in scores] + [None] * (k - len(scores))
