"""Flat index: exact search by scanning every row (CONTRACT 6.1). Recall@10 = 1.0.

Phase 5 (CONTRACT 13.3):
- delete(index, mask): tombstone bool array. Search still scores every row (distance_computations
  stays N), then selects the top k among live rows only.
- update(index, ids, vectors): overwrite the rows. The first change copies the corpus array, so
  the caller's array (shared with other indexes in tests) is never written.
- compact(index, mode): copy the live rows into a new array with an int64 ID map; the tombstone
  array is dropped. Both modes do the same for flat.
- index_bytes: 0 for a plain build (the corpus array is the index). After a change the index
  owns its row array, so it counts rows x dim x 4, plus the ID map (8 bytes per row) and the
  tombstone bit set (N/8). This is what makes compaction measurable for flat.
"""

import time

import numpy as np

from . import changes, distance, filters, vro

BUILD_PARAMS: dict = {}
SEARCH_PARAMS: dict = {"filter": "none"}
DATA_DIR = None  # set by bench; filter_<name>.npy lives here (CONTRACT 11)


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    t0 = time.perf_counter()
    index = {"vectors": vectors, "train_s": 0.0}
    index["add_s"] = time.perf_counter() - t0
    return index


def _allowed(index: dict, name: str):
    """(positions, ids) of rows that pass filter `name` and are not deleted, or None for all rows.
    Positions index index["vectors"]; ids are row IDs. Cached per filter name."""
    vectors = index["vectors"]
    n_rows = len(vectors)
    id_map = index.get("id_map")
    dead = index.get("deleted")
    fmask = None
    if name != "none":
        fmask = filters.mask(DATA_DIR, name, index.get("n_orig", n_rows))
        if id_map is not None:
            fmask = fmask[id_map]
    if fmask is None and dead is None:
        return None
    cache = index.setdefault("filtered", {})
    if name not in cache:
        allow = np.ones(n_rows, dtype=bool) if fmask is None else fmask.copy()
        if dead is not None:
            allow &= ~dead
        pos = np.flatnonzero(allow).astype(np.int64)
        ids = pos if id_map is None else id_map[pos]
        cache[name] = (pos, ids)
    return cache[name]


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    vectors = index["vectors"]
    name = str(params.get("filter", "none"))
    id_map = index.get("id_map")
    sel = _allowed(index, name)
    if sel is None:
        index["distance_computations"] = len(vectors)
        index["search_extra"] = {"filter_rows": len(vectors)}
        return distance.top_k(distance.scores(query, vectors), k, id_map)
    pos, ids = sel
    if index.get("deleted") is not None:
        # Tombstones: score every row, then keep only live (and passing) rows.
        index["distance_computations"] = len(vectors)
        index["search_extra"] = {"filter_rows": len(ids)}
        s = distance.scores(query, vectors)
        return distance.top_k(s[pos], k, ids)
    # Filtered, no tombstones: gather the passing rows once per filter (cached), then scan them.
    rows_cache = index.setdefault("filtered_rows", {})
    if name not in rows_cache:
        rows_cache[name] = np.ascontiguousarray(vectors[pos])
    index["distance_computations"] = len(ids)
    index["search_extra"] = {"filter_rows": len(ids)}
    return distance.top_k(distance.scores(query, rows_cache[name]), k, ids)


def _own(index: dict) -> None:
    if not index.get("owned"):
        index["vectors"] = index["vectors"].copy()
        index["owned"] = True
        index["n_orig"] = len(index["vectors"])


def delete(index: dict, mask: np.ndarray) -> dict:
    _own(index)
    index["deleted"] = np.asarray(mask, dtype=bool).copy()
    index.pop("filtered", None)
    index.pop("filtered_rows", None)
    return index


def update(index: dict, ids: np.ndarray, vectors: np.ndarray) -> dict:
    _own(index)
    if index.get("id_map") is not None:
        raise ValueError("flat.update: index is compacted; update before compaction")
    index["vectors"][np.asarray(ids, dtype=np.int64)] = vectors
    index.pop("filtered", None)
    index.pop("filtered_rows", None)
    return index


def compact(index: dict, mode: str = "rebuild") -> dict:
    _own(index)
    dead = index.pop("deleted", None)
    if dead is not None:
        live = np.flatnonzero(~dead).astype(np.int64)
        old_map = index.get("id_map")
        index["vectors"] = np.ascontiguousarray(index["vectors"][live])
        index["id_map"] = live if old_map is None else old_map[live]
    index.pop("filtered", None)
    index.pop("filtered_rows", None)
    return index


def index_bytes(index: dict) -> int:
    if not index.get("owned"):
        return 0
    total = int(index["vectors"].nbytes)
    if index.get("id_map") is not None:
        total += int(index["id_map"].nbytes)
    if index.get("deleted") is not None:
        total += changes.tombstone_bytes(len(index["deleted"]))
    return total


def save(index: dict, path, build_params: dict | None = None, seed: int = 0) -> int:
    """Write the .vro file (CONTRACT 15.1): vectors and tombstones. Returns the file size.

    A compacted index (it has an ID map) is expanded back to all N original rows in row order:
    a dropped row gets a zero vector and its tombstone bit (CONTRACT 15.1)."""
    v = index["vectors"]
    dead = index.get("deleted")
    id_map = index.get("id_map")
    if id_map is not None:
        n = int(index["n_orig"])
        full = np.zeros((n, v.shape[1]), dtype=np.float32)
        full[id_map] = v
        full_dead = np.ones(n, dtype=bool)
        full_dead[id_map] = False if dead is None else dead
        v, dead = full, full_dead
    n, dim = v.shape
    return vro.write(path, "flat", n, dim, {} if build_params is None else build_params, seed, [
        ("vectors", "f32", v),
        ("tombstones", "u8", vro.pack_tombstones(dead, n)),
    ])


def load(path, params: dict | None = None, dim: int | None = None) -> dict:
    """Read a .vro file written by any language. Refuses a wrong index, dim, or build_params."""
    r = vro.read(path, "flat", dim, params)
    n, d = r.header["n"], r.header["dim"]
    index = {"vectors": r.array("vectors", "f32", (n, d)), "train_s": 0.0, "add_s": 0.0}
    dead = vro.unpack_tombstones(r.array("tombstones", "u8", ((n + 7) // 8,)), n)
    if dead is not None:
        index.update(deleted=dead, owned=True, n_orig=n)
    index["header"] = r.header
    return index
