"""pgvector client for the database benches (tools/db/README.md).

Server: Postgres 17 with pgvector 0.8.6, compose service `pgvector`, user/password/db `vro`.
Client: psycopg 3.

Table: items(id bigint primary key, embedding vector(384), views real, title text,
wiki_id bigint, paragraph_id int, langs int). id = row index in vectors.npy.
views is real (float4), not int: metadata.parquet stores it as float32 page views.

Index mapping (metric is always the inner product, operator class vector_ip_ops):
- flat:  no index. The query runs a sequential scan and an exact sort. build_index returns 0.
- ivf:   ivfflat with lists = nlist. Search sets ivfflat.probes = nprobe.
- hnsw:  hnsw with m = m, ef_construction = ef_construct. Search sets hnsw.ef_search = ef.

Build: the service sets maintenance_work_mem = 1GB, and build_index sets it again for the
session. pgvector builds HNSW (and IVFFlat) in parallel with up to
max_parallel_maintenance_workers workers (4 in docker-compose.yml), so server_build_s is
a multi-core wall time.

Scores: the `<#>` operator returns the negative inner product. The query negates it, so
a higher score is better, as in CONTRACT section 3.

Filters (Phase 3): filter=<name> adds `WHERE views >= t` (t from filters.json, sent as float8;
views is real, so the comparison runs in double precision and matches the float32 mask).
Build param views_index=1 adds a B-tree on views (the planner can then choose it over the
vector index). Search param iterative=1 sets hnsw.iterative_scan / ivfflat.iterative_scan =
relaxed_order for the transaction; the index scan then continues past ef_search / probes until
k rows pass the filter (up to hnsw.max_scan_tuples, default 20,000). relaxed_order can return
rows slightly out of order, so the client re-sorts by score. iterative=0 sets it off.

Search: one prepared statement, binary protocol. The query vector goes as a binary
pgvector value (int16 dim, int16 unused, dim big-endian float4).
"""

from __future__ import annotations

import struct
import time

import numpy as np
import psycopg
import pyarrow as pa
from psycopg.adapt import Dumper, PyFormat
from psycopg.pq import Format
from psycopg.types import TypeInfo

from tools.db import base

DIM = 384
DSN = "host=pgvector port=5432 user=vro password=vro dbname=vro"
SEARCH_SQL = (
    "SELECT id, (embedding <#> %(q)s) * -1 AS score FROM items "
    "ORDER BY embedding <#> %(q)s LIMIT %(k)s"
)
FILTER_SQL = (
    "SELECT id, (embedding <#> %(q)s) * -1 AS score FROM items WHERE views >= %(t)s "
    "ORDER BY embedding <#> %(q)s LIMIT %(k)s"
)


class Vec:
    """A float32 vector to send as a pgvector binary value."""

    __slots__ = ("a",)

    def __init__(self, a: np.ndarray):
        self.a = a


def _vec_bytes(a: np.ndarray) -> bytes:
    return struct.pack(">HH", a.shape[0], 0) + np.asarray(a, dtype=">f4").tobytes()


class PgvectorClient:
    name = "pgvector"

    def __init__(self, dsn: str = DSN):
        self.dsn = dsn
        self.conn: psycopg.Connection | None = None
        self.index_name: str | None = None
        self._search_key: tuple | None = None
        self.data_dir = None

    # -- connection -------------------------------------------------------
    def connect(self, timeout_s: float = 120.0) -> None:
        t0 = time.perf_counter()
        while True:
            try:
                self.conn = psycopg.connect(self.dsn, autocommit=True)
                break
            except psycopg.OperationalError:
                if time.perf_counter() - t0 > timeout_s:
                    raise
                time.sleep(1.0)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        info = TypeInfo.fetch(self.conn, "vector")
        oid = info.oid

        class VecDumper(Dumper):
            format = Format.BINARY

            def dump(self, obj):
                return _vec_bytes(obj.a)

        VecDumper.oid = oid
        self.conn.adapters.register_dumper(Vec, VecDumper)
        self.vector_oid = oid

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # -- data -------------------------------------------------------------
    def reset(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS items")
        self.index_name = None
        self._search_key = None

    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float:
        """COPY in binary format, one COPY per batch of rows. Returns seconds (includes CREATE TABLE and ANALYZE)."""
        n, dim = vectors.shape
        cols = {c: meta.column(c).to_pylist() for c in ("views", "title", "wiki_id", "paragraph_id", "langs")}
        t0 = time.perf_counter()
        self.conn.execute(
            f"CREATE TABLE items (id bigint PRIMARY KEY, embedding vector({dim}), views real, "
            "title text, wiki_id bigint, paragraph_id int, langs int)"
        )
        with self.conn.cursor() as cur:
            for start in range(0, n, batch):
                stop = min(start + batch, n)
                with self.conn.transaction():
                    with cur.copy(
                        "COPY items (id, embedding, views, title, wiki_id, paragraph_id, langs) FROM STDIN (FORMAT BINARY)"
                    ) as cp:
                        cp.set_types(["int8", self.vector_oid, "float4", "text", "int8", "int4", "int4"])
                        for i in range(start, stop):
                            cp.write_row((
                                i, Vec(vectors[i]), cols["views"][i], cols["title"][i],
                                cols["wiki_id"][i], cols["paragraph_id"][i], cols["langs"][i],
                            ))
        self.conn.execute("ANALYZE items")
        return time.perf_counter() - t0

    def build_index(self, index: str, params: dict) -> float:
        """Vector index, plus (views_index=1) a B-tree on views. Seconds cover both."""
        t0 = time.perf_counter()
        self.conn.execute("DROP INDEX IF EXISTS items_views")
        if int(params.get("views_index", 0)):
            self.conn.execute("CREATE INDEX items_views ON items (views)")
            self.conn.execute("ANALYZE items")
        views_s = time.perf_counter() - t0
        if index == "flat":
            self.index_name = None
            return views_s
        self.conn.execute("SET maintenance_work_mem = '1GB'")
        if index == "ivf":
            sql = f"CREATE INDEX items_ivf ON items USING ivfflat (embedding vector_ip_ops) WITH (lists = {int(params['nlist'])})"
            name = "items_ivf"
        elif index == "hnsw":
            sql = (
                "CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_ip_ops) "
                f"WITH (m = {int(params['m'])}, ef_construction = {int(params['ef_construct'])})"
            )
            name = "items_hnsw"
        else:
            raise ValueError(f"pgvector has no {index}")
        t0 = time.perf_counter()
        self.conn.execute(sql)
        dt = time.perf_counter() - t0
        self.index_name = name
        return dt + views_s

    # -- search -----------------------------------------------------------
    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]:
        with self.conn.transaction():
            with self.conn.cursor(binary=True) as cur:
                if "nprobe" in params:
                    cur.execute(f"SET LOCAL ivfflat.probes = {int(params['nprobe'])}")
                if "ef" in params:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {int(params['ef'])}")
                iterative = int(params.get("iterative", 0))
                if "iterative" in params:
                    mode = "relaxed_order" if iterative else "off"
                    cur.execute(f"SET LOCAL hnsw.iterative_scan = {mode}")
                    cur.execute(f"SET LOCAL ivfflat.iterative_scan = {mode}")
                t = base.views_min(self.data_dir, str(params.get("filter", "none")))
                if t is None:
                    cur.execute(SEARCH_SQL, {"q": Vec(query), "k": k}, prepare=True)
                else:
                    cur.execute(FILTER_SQL, {"q": Vec(query), "k": k, "t": t}, prepare=True)
                rows = cur.fetchall()
        if iterative:
            rows.sort(key=lambda r: (-r[1], r[0]))
        return [int(r[0]) for r in rows], [float(r[1]) for r in rows]

    # -- stats ------------------------------------------------------------
    def stats(self) -> dict:
        c = self.conn
        rows = c.execute("SELECT count(*) FROM items").fetchone()[0]
        table_bytes = c.execute("SELECT pg_relation_size('items')").fetchone()[0]
        total_bytes = c.execute("SELECT pg_total_relation_size('items')").fetchone()[0]
        index_bytes = 0
        if self.index_name:
            index_bytes = c.execute("SELECT pg_relation_size(%s::regclass)", (self.index_name,)).fetchone()[0]
        server = c.execute("SHOW server_version").fetchone()[0]
        ext = c.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        out = {
            "rows": int(rows),
            "index_bytes": int(index_bytes),
            "table_bytes": int(table_bytes),
            "disk_bytes": int(total_bytes),
            "server_version": f"PostgreSQL {server}",
            "pgvector_version": ext,
            "client_lib": f"psycopg {psycopg.__version__}",
            "index_name": self.index_name or "none (sequential scan)",
        }
        if not self.index_name:
            out["index_bytes_source"] = "unavailable"
        return out


def make_client() -> PgvectorClient:
    return PgvectorClient()
