"""Qdrant client for the database benches (tools/db/README.md).

Collection `vro`: 384-dim vectors, distance Dot, point ID = row index, payload = META_COLUMNS.

Order of work. The bench calls reset, load, build_index. `load` creates the collection with
hnsw m=0 and indexing disabled (indexing_threshold=0), so the upserts do no index work.
`build_index` then updates the collection with the index config (hnsw m, ef_construct,
quantization) and enables indexing. Its seconds run from that update call until the
collection is green and every point is indexed: this is `server_build_s`.

Index mapping:
  flat  hnsw m=0; search with exact=True.
  hnsw  hnsw m, ef_construct; quant none|scalar (int8)|product (x8, pq m=48)|binary.
  pq    product quantization, compression from m (384/m = 8 -> x8); hnsw stays m=0.
Filters (Phase 3): load creates a float payload index on `views`, before the upserts, so the
HNSW build sees it (Qdrant adds extra graph edges per payload-index value block). A search with
filter=<name> passes Filter(must=[FieldCondition(views, Range(gte=t))]), t from filters.json.
Search: hnsw_ef = ef; quantization rescore = bool(rescore or rerank). Scores are raw Dot.

Updates and deletes (Phase 5, CONTRACT section 13.4). Qdrant never edits an HNSW graph in place.
A delete sets a bit in the segment's deleted-point bit set; the graph keeps the node, a search
still walks through it and drops it from the results. An upsert of an existing ID marks the old
copy deleted in its segment and writes the new copy to an appendable segment, which the optimizer
indexes later (Qdrant searches unindexed segments exactly until then).
The vacuum optimizer is Qdrant's compaction: it rebuilds a segment (a new HNSW graph from the
live points) when the segment holds more than vacuum_min_vector_number vectors and its deleted
fraction exceeds deleted_threshold (defaults 1000 and 0.2). So a default collection compacts on
its own after del30. To measure the tombstone state, delete() and update() first set
deleted_threshold = 1.0 (vacuum off); compact() then sets deleted_threshold = 0.01 and
vacuum_min_vector_number = 100 and waits for status green. stats() reads segment counts and
deleted-vector counts from GET /telemetry (details_level=10).
"""

from __future__ import annotations

import json
import time
import urllib.request
from importlib.metadata import version

import numpy as np
import pyarrow as pa
from qdrant_client import QdrantClient
from qdrant_client import models as m

from tools.db import base

COLLECTION = "vro"
DIM = 384
BUILD_TIMEOUT_S = 3600

_COMPRESSION = {4: m.CompressionRatio.X4, 8: m.CompressionRatio.X8, 16: m.CompressionRatio.X16,
                32: m.CompressionRatio.X32, 64: m.CompressionRatio.X64}


def _product(msub: int) -> m.ProductQuantization:
    ratio = DIM // msub
    if ratio not in _COMPRESSION:
        raise ValueError(f"qdrant: m={msub} gives compression x{ratio}; supported x4, x8, x16, x32, x64")
    return m.ProductQuantization(product=m.ProductQuantizationConfig(compression=_COMPRESSION[ratio], always_ram=True))


def _quantization(quant: str, msub: int):
    if quant == "none":
        return None
    if quant == "scalar":
        return m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, always_ram=True))
    if quant == "product":
        return _product(msub)
    if quant == "binary":
        return m.BinaryQuantization(binary=m.BinaryQuantizationConfig(always_ram=True))
    raise ValueError(f"qdrant: unknown quant {quant!r}; known: none, scalar, product, binary")


class QdrantDB:
    name = "qdrant"

    def __init__(self, host: str = "qdrant"):
        self.host = host
        self.client: QdrantClient | None = None
        self.index = "flat"
        self.data_dir = None

    def connect(self) -> None:
        self.client = QdrantClient(host=self.host, port=6333, grpc_port=6334, prefer_grpc=True, timeout=600)

    def reset(self) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)

    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float:
        """Create the collection (no index work) and upsert all rows. Returns seconds."""
        t0 = time.perf_counter()
        self.client.create_collection(
            COLLECTION,
            vectors_config=m.VectorParams(size=DIM, distance=m.Distance.DOT),
            hnsw_config=m.HnswConfigDiff(m=0),
            optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0),
        )
        self.client.create_payload_index(COLLECTION, "views", field_schema=m.PayloadSchemaType.FLOAT, wait=True)
        cols = {c: meta.column(c).to_pylist() for c in base.META_COLUMNS}
        n = len(vectors)
        for s in range(0, n, batch):
            e = min(s + batch, n)
            payload = [{c: cols[c][i] for c in base.META_COLUMNS} for i in range(s, e)]
            self.client.upsert(COLLECTION, points=m.Batch(ids=list(range(s, e)),
                               vectors=vectors[s:e].tolist(), payloads=payload), wait=True)
        self._wait_green(require_indexed=False)
        return time.perf_counter() - t0

    def _wait_green(self, require_indexed: bool) -> None:
        time.sleep(0.5)  # let the optimizer pick up a config change before the first poll

        def ready() -> bool:
            info = self.client.get_collection(COLLECTION)
            if info.status != m.CollectionStatus.GREEN:
                return False
            return not require_indexed or (info.indexed_vectors_count or 0) >= (info.points_count or 0)

        base.wait_until(ready, BUILD_TIMEOUT_S, what="qdrant collection green")

    def build_index(self, index: str, params: dict) -> float:
        """Update the collection with the index config and enable indexing.

        Returns seconds from the update call (end of load) until status green and,
        for hnsw, every vector indexed.
        """
        self.index = index
        t0 = time.perf_counter()
        if index == "flat":
            self._wait_green(require_indexed=False)
            return time.perf_counter() - t0
        if index == "hnsw":
            hnsw = m.HnswConfigDiff(m=int(params.get("m", 16)), ef_construct=int(params.get("ef_construct", 100)))
            # hnsw m is the graph degree, not a PQ setting: quant=product uses the pq default m=48 (x8).
            quant = _quantization(str(params.get("quant", "none")), base.BUILD_DEFAULTS["pq"]["m"])
        elif index == "pq":
            hnsw = None
            quant = _product(int(params.get("m", 48)))
        else:
            raise ValueError(f"qdrant has no {index}")
        # indexing_threshold=1 (KB): index every segment, also the small ones.
        self.client.update_collection(COLLECTION, hnsw_config=hnsw, quantization_config=quant,
                                      optimizers_config=m.OptimizersConfigDiff(indexing_threshold=1))
        self._wait_green(require_indexed=(index == "hnsw"))
        return time.perf_counter() - t0

    def attach(self, index: str) -> None:
        self.index = index

    def insert(self, vectors: np.ndarray, meta_rows: pa.Table, ids: list[int]) -> float:
        """Upsert rows into the built collection (wait=True: the call returns when the rows are
        searchable; the optimizer indexes them later, and until then Qdrant scans them exactly)."""
        t0 = time.perf_counter()
        cols = {c: meta_rows.column(c).to_pylist() for c in base.META_COLUMNS}
        payload = [{c: cols[c][i] for c in base.META_COLUMNS} for i in range(len(ids))]
        self.client.upsert(COLLECTION, points=m.Batch(ids=[int(i) for i in ids], vectors=vectors.tolist(),
                           payloads=payload), wait=True)
        return time.perf_counter() - t0

    def finish_inserts(self) -> None:
        pass  # upsert(wait=True) already made each batch searchable

    # -- Phase 5 -------------------------------------------------------------
    def _vacuum_off(self) -> None:
        self.client.update_collection(COLLECTION, optimizers_config=m.OptimizersConfigDiff(deleted_threshold=1.0))

    def delete(self, ids: np.ndarray) -> float:
        """Delete points by ID in batches of 10,000 (wait=True). Returns seconds."""
        self._vacuum_off()
        t0 = time.perf_counter()
        ids = [int(i) for i in ids]
        for s in range(0, len(ids), 10000):
            self.client.delete(COLLECTION, points_selector=m.PointIdsList(points=ids[s:s + 10000]), wait=True)
        return time.perf_counter() - t0

    def update(self, ids: np.ndarray, vectors: np.ndarray, meta_rows: pa.Table) -> float:
        """Upsert the new vectors under the same IDs (payload rewritten unchanged), batches of 2,000.
        Seconds cover the upserts and the wait until the optimizer has indexed the new copies (green)."""
        self._vacuum_off()
        t0 = time.perf_counter()
        cols = {c: meta_rows.column(c).to_pylist() for c in base.META_COLUMNS}
        for s in range(0, len(ids), 2000):
            e = min(s + 2000, len(ids))
            payload = [{c: cols[c][i] for c in base.META_COLUMNS} for i in range(s, e)]
            self.client.upsert(COLLECTION, points=m.Batch(ids=[int(i) for i in ids[s:e]], vectors=vectors[s:e].tolist(),
                               payloads=payload), wait=True)
        self._wait_green(require_indexed=(self.index == "hnsw"))
        return time.perf_counter() - t0

    def compact(self) -> dict:
        """Turn the vacuum optimizer on with low thresholds and wait until the collection is green
        and every vector is indexed. Seconds run from the config change."""
        t0 = time.perf_counter()
        self.client.update_collection(COLLECTION, optimizers_config=m.OptimizersConfigDiff(
            deleted_threshold=0.01, vacuum_min_vector_number=100))
        time.sleep(2.0)  # the optimizer loop picks up the config change asynchronously
        self._wait_green(require_indexed=(self.index == "hnsw"))
        return {"compact_s": time.perf_counter() - t0, "method": "vacuum optimizer: deleted_threshold=0.01, vacuum_min_vector_number=100"}

    def _segments(self) -> dict:
        try:
            with urllib.request.urlopen(f"http://{self.host}:6333/telemetry?details_level=10", timeout=60) as r:
                doc = json.load(r)
        except Exception as e:  # noqa: BLE001
            return {"telemetry_error": str(e)}
        cols = doc["result"]["collections"]["collections"]
        col = next((c for c in cols if c.get("id") == COLLECTION), None)
        if col is None:
            return {}
        segs = [s["info"] for sh in col["shards"] if sh.get("local") for s in sh["local"].get("segments", [])]
        disk = sum(int(s.get("disk_usage_bytes") or 0) for s in segs)
        vec_payload = sum(int(s.get("vectors_size_bytes") or 0) + int(s.get("payloads_size_bytes") or 0) for s in segs)
        return {
            "segments_count": len(segs),
            "segments_indexed": sum(1 for s in segs if s.get("segment_type") == "indexed"),
            "deleted_vectors": sum(int(s.get("num_deleted_vectors") or 0) for s in segs),
            "stored_vectors": sum(int(s.get("num_vectors") or 0) for s in segs),
            "disk_bytes": disk if disk > 0 else vec_payload,
            "disk_bytes_source": "segment disk_usage_bytes" if disk > 0 else
                                 "segment vectors_size_bytes + payloads_size_bytes (telemetry; HNSW links not included)",
            "ram_bytes": sum(int(s.get("ram_usage_bytes") or 0) for s in segs),
        }

    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]:
        rescore = bool(params.get("rescore") or params.get("rerank"))
        sp = m.SearchParams(
            hnsw_ef=int(params["ef"]) if "ef" in params else None,
            exact=(self.index == "flat"),
            quantization=m.QuantizationSearchParams(rescore=rescore),
        )
        t = base.views_min(self.data_dir, str(params.get("filter", "none")))
        flt = None if t is None else m.Filter(must=[m.FieldCondition(key="views", range=m.Range(gte=t))])
        res = self.client.query_points(COLLECTION, query=query.tolist(), limit=k, search_params=sp,
                                       query_filter=flt, with_payload=False).points
        return [int(p.id) for p in res], [float(p.score) for p in res]

    def stats(self) -> dict:
        info = self.client.get_collection(COLLECTION)
        try:
            server_version = self.client.info().version
        except Exception:  # noqa: BLE001
            server_version = "unknown"
        return {
            "rows": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "server_version": server_version,
            "index_bytes": 0,
            "index_bytes_source": "unavailable",
            "client_lib": f"qdrant-client {version('qdrant-client')}",
            **self._segments(),
        }

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def make_client() -> QdrantDB:
    return QdrantDB()
