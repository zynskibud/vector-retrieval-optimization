"""Client interface for the database benches (tools/db/README.md) and shared helpers."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
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
    "flat": {"filter": "none"},
    "ivf": {"nprobe": 8, "filter": "none"},
    "pq": {"rerank": 0},
    "ivf_pq": {"nprobe": 8, "rerank": 0},
    "hnsw": {"ef": 64, "rescore": 0, "filter": "none"},
    "diskann": {"l": 100, "beam": 4},
}

# Per-database params on top of the defaults above (Phase 3, README "Filters"):
# pgvector's B-tree on views (build) and iterative index scans (search).
BUILD_EXTRA = {"pgvector": {"flat": {"views_index": 0}, "ivf": {"views_index": 0}, "hnsw": {"views_index": 0}}}
SEARCH_EXTRA = {"pgvector": {"ivf": {"iterative": 0}, "hnsw": {"iterative": 0}}}

FILTER_NAMES = ("none", "top50", "top10", "top1", "top01")


def build_defaults(db: str, index: str) -> dict:
    return {**BUILD_DEFAULTS[index], **BUILD_EXTRA.get(db, {}).get(index, {})}


def search_defaults(db: str, index: str) -> dict:
    return {**SEARCH_DEFAULTS[index], **SEARCH_EXTRA.get(db, {}).get(index, {})}


def views_min(data_dir, name: str) -> float | None:
    """The threshold t of filter `name` (views >= t) from <data_dir>/filters.json; None for "none".

    Every client calls this with its `data_dir`, which bench.py sets before load.
    """
    if name == "none":
        return None
    if name not in FILTER_NAMES:
        raise ValueError(f"unknown filter {name!r}; known: {FILTER_NAMES}")
    if data_dir is None:
        raise ValueError("filter needs the data directory: set client.data_dir first")
    return float(_filters(str(data_dir))[name]["views_min"])


_FILTERS_CACHE: dict[str, dict] = {}


def _filters(data_dir: str) -> dict:
    if data_dir not in _FILTERS_CACHE:
        _FILTERS_CACHE[data_dir] = json.loads((Path(data_dir) / "filters.json").read_text())
    return _FILTERS_CACHE[data_dir]


class Client(Protocol):
    """One database. Host names are the compose service names: qdrant, pgvector, milvus."""

    name: str
    data_dir: Path | None  # set by bench.py before load; the source of filters.json

    def connect(self) -> None: ...
    def reset(self) -> None: ...
    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float: ...
    def build_index(self, index: str, params: dict) -> float: ...
    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]: ...
    # Phase 4 (CONTRACT section 12.3, tools/load/bench.py):
    def attach(self, index: str) -> None: ...     # a second connection to a built index: set the search state only
    def insert(self, vectors: np.ndarray, meta_rows: pa.Table, ids: list[int]) -> float: ...  # add rows to a built index; seconds
    def finish_inserts(self) -> None: ...         # make every inserted row searchable (Milvus: flush)
    # Phase 5 (CONTRACT section 13.4): applied after build_index, before the searches.
    def delete(self, ids: np.ndarray) -> float: ...  # delete rows by ID; searchable state when it returns; seconds
    def update(self, ids: np.ndarray, vectors: np.ndarray, meta_rows: pa.Table) -> float: ...  # new vectors, same IDs; seconds
    def compact(self) -> dict: ...                # the database's repair; returns {"compact_s": s, ...detail}
    def stats(self) -> dict: ...
    def close(self) -> None: ...


DELETE_NAMES = ("del10", "del30", "del50")
UPDATE_NAMES = ("upd10",)


def delete_ids(data_dir, name: str, n: int) -> np.ndarray:
    """Row IDs deleted by change set `name` among the first n rows (CONTRACT section 13.1)."""
    if name not in DELETE_NAMES:
        raise ValueError(f"unknown delete set {name!r}; known: {DELETE_NAMES}")
    return np.flatnonzero(np.load(Path(data_dir) / f"delete_{name}.npy")[:n]).astype(np.int64)


def update_rows(data_dir, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(IDs, new vectors) of update set `name`, restricted to IDs < n."""
    if name not in UPDATE_NAMES:
        raise ValueError(f"unknown update set {name!r}; known: {UPDATE_NAMES}")
    ids = np.load(Path(data_dir) / f"update_{name}_ids.npy")
    vecs = np.load(Path(data_dir) / f"update_{name}_vectors.npy")
    keep = ids < n
    return ids[keep].astype(np.int64), np.ascontiguousarray(vecs[keep], dtype=np.float32)


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
