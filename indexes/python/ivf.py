"""Inverted file index (CONTRACT 6.3).

Recall floor (CONTRACT 9, dev set): nprobe=8, recall@10 >= 0.75.

In build_params, train_size None means the k-means default min(N, 256 * nlist).
bench fills it in before build (6.2).

Layout (CSR): list_ids is one contiguous int64 array of vector IDs, grouped by list.
The IDs of list c are list_ids[offsets[c] : offsets[c + 1]], in ascending ID order.

Phase 5 (CONTRACT 13.3):
- delete(index, mask): tombstone bool array; search drops tombstoned IDs from the probed lists
  before scoring, so distance_computations counts only scored rows (as for a filter).
- update(index, ids, vectors): overwrite the vectors (in a copy of the corpus array made on the
  first update), assign each updated ID to its new best center, and rebuild the CSR arrays with
  one stable sort by (list, ID), so each list keeps ascending IDs.
- compact(index, mode): drop tombstoned IDs from list_ids and recompute offsets; the tombstone
  array is dropped. The corpus array is not touched (lists hold row IDs). Both modes are the same.
- index_bytes adds the tombstone bit set (N/8) while tombstones exist.
"""

import time

import numpy as np

from . import changes, distance, filters, kmeans, vro

BUILD_PARAMS: dict = {"nlist": 1024, "train_size": None, "iters": 20}
SEARCH_PARAMS: dict = {"nprobe": 8, "filter": "none"}
DATA_DIR = None  # set by bench; filter_<name>.npy lives here (CONTRACT 11)


def build(vectors: np.ndarray, params: dict, threads: int, seed: int) -> dict:
    nlist = int(params["nlist"])
    iters = int(params["iters"])
    train_size = params.get("train_size")
    if train_size is None:
        train_size = kmeans.default_train_size(len(vectors), nlist)

    t0 = time.perf_counter()
    centers = kmeans.kmeans(vectors, nlist, iters=iters, seed=seed, train_size=int(train_size),
                            metric="ip", normalize=True)
    t1 = time.perf_counter()

    # Add: assign every corpus row to its best center, in blocks to bound the (block, nlist) matrix.
    n = len(vectors)
    labels = np.empty(n, dtype=np.int64)
    block = 65536
    for start in range(0, n, block):
        labels[start : start + block] = np.argmax(vectors[start : start + block] @ centers.T, axis=1)
    counts = np.bincount(labels, minlength=nlist)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    list_ids = np.argsort(labels, kind="stable").astype(np.int64)  # stable: IDs ascend in each list
    t2 = time.perf_counter()

    return {
        "vectors": vectors,
        "centers": centers,
        "list_ids": list_ids,
        "offsets": offsets,
        "nlist": nlist,
        "train_s": t1 - t0,
        "add_s": t2 - t1,
        "extra": {"id_type": "int64"},
    }


def search(index: dict, query: np.ndarray, k: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    nlist = index["nlist"]
    nprobe = max(1, min(int(params["nprobe"]), nlist))
    offsets = index["offsets"]
    list_ids = index["list_ids"]

    center_scores = distance.scores(query, index["centers"])
    # The nprobe best centers; ties to the lower center index (same rule as for IDs).
    probe, _ = distance.top_k(center_scores, nprobe)

    ids = np.concatenate([list_ids[offsets[c] : offsets[c + 1]] for c in probe])
    mask = filters.mask(DATA_DIR, params.get("filter", "none"), len(index["vectors"]))
    if mask is not None:
        ids = ids[mask[ids]]
    dead = index.get("deleted")
    if dead is not None:
        ids = ids[~dead[ids]]  # skip tombstoned rows  # skip rows that fail the filter; only passing rows are scored
    index["distance_computations"] = nlist + len(ids)
    index["search_extra"] = {"filter_rows": len(index["vectors"]) if mask is None else int(filters.count(mask))}
    return distance.top_k(distance.scores(query, index["vectors"][ids]), k, ids)


def index_bytes(index: dict) -> int:
    total = int(index["centers"].nbytes + index["list_ids"].nbytes)
    if index.get("deleted") is not None:
        total += changes.tombstone_bytes(len(index["deleted"]))
    return total


def _set_lists(index: dict, ids: np.ndarray, labels: np.ndarray) -> None:
    """Rebuild the CSR arrays from (ID, list) pairs; IDs ascend inside each list."""
    order = np.lexsort((ids, labels))
    index["list_ids"] = np.ascontiguousarray(ids[order]).astype(np.int64)
    counts = np.bincount(labels, minlength=index["nlist"])
    offsets = np.zeros(index["nlist"] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    index["offsets"] = offsets


def _labels(index: dict) -> np.ndarray:
    counts = np.diff(index["offsets"])
    return np.repeat(np.arange(index["nlist"], dtype=np.int64), counts)


def delete(index: dict, mask: np.ndarray) -> dict:
    index["deleted"] = np.asarray(mask, dtype=bool).copy()
    return index


def update(index: dict, ids: np.ndarray, vectors: np.ndarray) -> dict:
    ids = np.asarray(ids, dtype=np.int64)
    if not index.get("owned"):
        index["vectors"] = index["vectors"].copy()
        index["owned"] = True
    index["vectors"][ids] = vectors
    new = np.argmax(np.asarray(vectors) @ index["centers"].T, axis=1).astype(np.int64)
    list_ids = index["list_ids"]
    labels = _labels(index)
    by_id = np.full(len(index["vectors"]), -1, dtype=np.int64)
    by_id[list_ids] = labels
    present = by_id[ids] >= 0  # an ID dropped by compaction stays out
    by_id[ids[present]] = new[present]
    _set_lists(index, list_ids, by_id[list_ids])
    return index


def compact(index: dict, mode: str = "rebuild") -> dict:
    dead = index.pop("deleted", None)
    if dead is not None:
        list_ids = index["list_ids"]
        labels = _labels(index)
        keep = ~dead[list_ids]
        _set_lists(index, list_ids[keep], labels[keep])
    return index


def save(index: dict, path, build_params: dict | None = None, seed: int = 0) -> int:
    """Write the .vro file (CONTRACT 15.1). list_ids and list_offsets are stored as int32.

    After compaction, list_ids holds only the live IDs, so its shape is [live rows], not [N]. A row
    that is in no list (dropped by compaction) gets its tombstone bit (CONTRACT 15.1)."""
    v = index["vectors"]
    n, dim = v.shape
    dead = np.ones(n, dtype=bool)
    dead[index["list_ids"]] = False
    if index.get("deleted") is not None:
        dead |= index["deleted"]
    if n >= 2**31:
        raise ValueError("ivf.save: IDs do not fit in int32")
    bp = {"nlist": index["nlist"]} if build_params is None else build_params
    return vro.write(path, "ivf", n, dim, bp, seed, [
        ("vectors", "f32", v),
        ("tombstones", "u8", vro.pack_tombstones(dead, n)),
        ("centers", "f32", index["centers"]),
        ("list_ids", "int32", index["list_ids"]),
        ("list_offsets", "int32", index["offsets"]),
    ])


def load(path, params: dict | None = None, dim: int | None = None) -> dict:
    """Read a .vro file written by any language. Refuses a wrong index, dim, or build_params."""
    r = vro.read(path, "ivf", dim, params)
    n, d = r.header["n"], r.header["dim"]
    nlist = int(r.header["build_params"]["nlist"])
    offsets = r.array("list_offsets", "int32", (nlist + 1,)).astype(np.int64)
    list_ids = r.array("list_ids", "int32").astype(np.int64)
    if list_ids.shape != (int(offsets[-1]),) or offsets[0] != 0 or (np.diff(offsets) < 0).any():
        raise vro.FormatError(f"{path}: list_offsets and list_ids do not agree")
    index = {
        "vectors": r.array("vectors", "f32", (n, d)),
        "centers": r.array("centers", "f32", (nlist, d)),
        "list_ids": list_ids,
        "offsets": offsets,
        "nlist": nlist,
        "train_s": 0.0,
        "add_s": 0.0,
        "extra": {"id_type": "int64"},
        "header": r.header,
    }
    dead = vro.unpack_tombstones(r.array("tombstones", "u8", ((n + 7) // 8,)), n)
    if dead is not None:
        index["deleted"] = dead
    return index
