"""Milvus client for the database benches (tools/db/README.md).

Server: Milvus standalone (compose service `milvus`, port 19530). Client: pymilvus MilvusClient.
Metric: IP everywhere. IVF_PQ rerank: Milvus has no rerank option for IVF_PQ, so the client does
it: fetch the top `rerank` IDs, read their vectors with `get`, sort by the exact dot product.
Filters (Phase 3): `views` is a FLOAT field (float32, as in metadata.parquet; an INT64 field
would truncate and move rows across the threshold). filter=<name> searches with
filter="views >= t", t from filters.json. No scalar index on views (Milvus scans the field).
"""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pymilvus
from pymilvus import DataType, MilvusClient

from tools.db import base

COLLECTION = "vro"
DIM = 384
TIMEOUT_S = 3600


class MilvusDB:
    name = "milvus"

    def __init__(self, uri: str = "http://milvus:19530"):
        self.uri = uri
        self.client: MilvusClient | None = None
        self.index = None
        self.data_dir = None
        self.consistency: str | None = None  # None = the collection default (Bounded)

    def connect(self) -> None:
        self.client = MilvusClient(uri=self.uri, timeout=60)

    def reset(self) -> None:
        if self.client.has_collection(COLLECTION):
            self.client.drop_collection(COLLECTION)
        self.index = None

    def _create(self) -> None:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("views", DataType.FLOAT)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("wiki_id", DataType.INT64)
        schema.add_field("paragraph_id", DataType.INT64)
        schema.add_field("langs", DataType.INT64)
        self.client.create_collection(COLLECTION, schema=schema)

    @staticmethod
    def _rows(vectors: np.ndarray, cols: dict, ids, offset: int = 0) -> list[dict]:
        rows = []
        for j, i in enumerate(ids):
            c = j + offset
            rows.append({
                "id": int(i),
                "embedding": vectors[j].tolist(),
                "views": float(cols["views"][c] or 0.0),
                "title": (cols["title"][c] or "")[:512].encode("utf-8")[:512].decode("utf-8", "ignore"),
                "wiki_id": int(cols["wiki_id"][c] or 0),
                "paragraph_id": int(cols["paragraph_id"][c] or 0),
                "langs": int(cols["langs"][c] or 0),
            })
        return rows

    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float:
        t0 = time.perf_counter()
        self._create()
        cols = {c: meta.column(c).to_pylist() for c in base.META_COLUMNS}
        for start in range(0, len(vectors), batch):
            end = min(start + batch, len(vectors))
            rows = self._rows(vectors[start:end], cols, range(start, end), offset=start)
            self.client.insert(COLLECTION, rows)
        self.client.flush(COLLECTION)
        return time.perf_counter() - t0

    def build_index(self, index: str, params: dict) -> float:
        if index == "flat":
            itype, p = "FLAT", {}
        elif index == "ivf":
            itype, p = "IVF_FLAT", {"nlist": int(params["nlist"])}
        elif index == "ivf_pq":
            itype, p = "IVF_PQ", {"nlist": int(params["nlist"]), "m": int(params["m"]), "nbits": int(params["nbits"])}
        elif index == "hnsw":
            itype, p = "HNSW", {"M": int(params["m"]), "efConstruction": int(params["ef_construct"])}
        elif index == "diskann":
            itype, p = "DISKANN", {}
        else:
            raise ValueError(f"milvus has no {index}")
        ip = self.client.prepare_index_params()
        ip.add_index(field_name="embedding", index_type=itype, metric_type="IP", params=p, index_name="embedding")
        t0 = time.perf_counter()
        self.client.create_index(COLLECTION, ip)

        def built() -> bool:
            d = self.client.describe_index(COLLECTION, "embedding")
            if d.get("state") == "Failed":
                raise RuntimeError(f"index build failed: {d}")
            return d.get("state") == "Finished" and d.get("pending_index_rows", 0) == 0

        base.wait_until(built, TIMEOUT_S, what="index built")
        self.client.load_collection(COLLECTION)
        base.wait_until(lambda: str(self.client.get_load_state(COLLECTION).get("state")).endswith("Loaded"),
                        TIMEOUT_S, what="collection loaded")
        self.index = index
        return time.perf_counter() - t0

    def attach(self, index: str) -> None:
        self.index = index

    def insert(self, vectors: np.ndarray, meta_rows: pa.Table, ids: list[int]) -> float:
        """Insert into the loaded collection. The rows land in a growing segment, which Milvus
        searches by brute force; with the default Bounded consistency a search can miss rows
        inserted in the last moments. No flush per batch (a flush seals a segment and is slow)."""
        cols = {c: meta_rows.column(c).to_pylist() for c in base.META_COLUMNS}
        t0 = time.perf_counter()
        self.client.insert(COLLECTION, self._rows(vectors, cols, ids))
        return time.perf_counter() - t0

    def finish_inserts(self) -> None:
        """Flush once after the inserter, and search with Strong consistency from now on, so
        the after-inserts pass sees every inserted row."""
        self.client.flush(COLLECTION)
        self.consistency = "Strong"

    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]:
        sp: dict = {}
        limit = k
        if self.index in ("ivf", "ivf_pq"):
            sp["nprobe"] = int(params.get("nprobe", 8))
        elif self.index == "hnsw":
            sp["ef"] = max(int(params.get("ef", 64)), k)
        elif self.index == "diskann":
            sp["search_list"] = max(int(params.get("l", 100)), k)
        rerank = int(params.get("rerank", 0)) if self.index == "ivf_pq" else 0
        if rerank > k:
            limit = rerank
        t = base.views_min(self.data_dir, str(params.get("filter", "none")))
        expr = "" if t is None else f"views >= {t!r}"
        res = self.client.search(COLLECTION, data=[query.tolist()], anns_field="embedding", limit=limit, filter=expr,
                                 output_fields=[], search_params={"metric_type": "IP", "params": sp},
                                 **({"consistency_level": self.consistency} if self.consistency else {}))
        hits = res[0]
        ids = [int(h["id"]) for h in hits]
        scores = [float(h["distance"]) for h in hits]
        if rerank > k and ids:
            rows = self.client.get(COLLECTION, ids=ids, output_fields=["embedding"])
            vec = {int(r["id"]): np.asarray(r["embedding"], dtype=np.float32) for r in rows}
            q = np.asarray(query, dtype=np.float32)
            exact = [(-float(vec[i] @ q), i) for i in ids]
            exact.sort()
            ids = [i for _, i in exact[:k]]
            scores = [-s for s, _ in exact[:k]]
        return ids, scores

    def stats(self) -> dict:
        rows = int(self.client.get_collection_stats(COLLECTION).get("row_count", 0))
        info = {}
        if self.index is not None:
            d = self.client.describe_index(COLLECTION, "embedding")
            info = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in d.items()}
        return {
            "rows": rows,
            "index_info": info,
            "server_version": str(self.client.get_server_version()),
            "index_bytes": 0,
            "index_bytes_source": "unavailable",
            "client_lib": f"pymilvus {pymilvus.__version__}",
        }

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def make_client() -> MilvusDB:
    return MilvusDB()
