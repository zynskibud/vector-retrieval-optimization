# Database benches (Phase 2)

The same six index types, run inside three self-hosted databases, measured with the same runner, schema, and report as the hand-built indexes. This file is the contract for `tools/db/`. `indexes/CONTRACT.md` sections 1 to 5 apply unchanged; this file says how a database bench maps to them.

## Layout

```
tools/db/
  README.md          this contract
  base.py            the client interface (Protocol) and shared helpers
  qdrant.py          one module per database, same interface
  pgvector.py
  milvus.py
  bench.py           CLI: same flags as CONTRACT section 2, plus --db
  tests/test_<db>.py pytest against a running database, 20k rows
```

Everything runs inside the `dbbench` container (`make dbbench ARGS=...`, `make db-test DB=...`), which sits on the internal Docker network with the database containers and has no other network. A database is started with `make db-up DB=<name>` and stopped with `make db-down DB=<name>`. One database at a time.

## Client interface (`base.py`)

```python
class Client(Protocol):
    name: str                                    # "qdrant" | "pgvector" | "milvus"
    def connect(self) -> None: ...               # host names: qdrant, pgvector, milvus (compose service names)
    def reset(self) -> None: ...                 # drop the collection/table if it exists
    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float:
        ...                                      # insert rows; ID = row index; returns seconds
    def build_index(self, index: str, params: dict) -> float:
        ...                                      # create the index, wait until it is built; returns seconds
    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]:
        ...                                      # one query; IDs best first, padded with -1; scores as the DB reports them (see Scores)
    def stats(self) -> dict: ...                 # server-side numbers: row count, index size or segment info, server version, disk bytes if reported
    def close(self) -> None: ...
```

`meta` is the metadata table from `data/processed/<set>/metadata.parquet`; load its `views`, `title`, `wiki_id`, `paragraph_id`, and `langs` columns as payload/columns so Phase 3 filters can use them. Text is not loaded (it is 378 MB and no test needs it).

## Command line

`python -m tools.db.bench --db <name> --index <name> --data DIR --out FILE [--k] [--build K=V]... [--search K=V]... [--seed] [--warmup] [--limit] [--batch 2000]`

Same meaning as CONTRACT section 2. `--threads` is accepted and ignored (the server chooses). Unsupported (db, index) pairs exit 2 with the message `<db> has no <index>`.

## Output JSON

CONTRACT section 3, with:

- `language` = the database name.
- `build.train_s` = 0, `build.add_s` = load seconds + index build seconds, `build.peak_rss_mb` = the client process (not the server), `build.index_bytes` = what the server reports for the index, or 0 with `extra.index_bytes_source = "unavailable"`.
- `extra`: `load_s`, `server_build_s`, `server_version`, `rows`, `disk_bytes` (server-reported, if any), `client_lib` (name and version), and any index-specific info the server gives.
- `searches[i].extra`: `{}` unless the server reports per-query counters.
- `distance_computations`: `null` (the server does not report it).

## Parameter mapping (contract name → database setting)

| Index | Contract params | Qdrant | pgvector | Milvus |
|---|---|---|---|---|
| flat | — | collection with `hnsw_config.m = 0` (exact) | no index, seq scan | `FLAT` |
| ivf | nlist, nprobe | — | `ivfflat` with `lists = nlist`, `SET ivfflat.probes = nprobe` | `IVF_FLAT` with `nlist`, search `nprobe` |
| pq | m, nbits, rerank | `product` quantization (`compression` from m: 384/m = 8 → `x8`), `rescore = rerank > 0`, `always_ram` | — | — (no standalone PQ) |
| ivf_pq | nlist, nprobe, m, nbits, rerank | — | — | `IVF_PQ` with `nlist`, `m`, `nbits`, search `nprobe` |
| hnsw | m, ef_construct, ef | `hnsw_config.m`, `ef_construct`; search `hnsw_ef = ef` | `hnsw` with `m`, `ef_construction`; `SET hnsw.ef_search = ef` | `HNSW` with `M`, `efConstruction`; search `ef` |
| diskann | r, l_build, l, beam | — | — | `DISKANN` (search `search_list = l`); `queryNode.enableDisk` and `common.diskIndex.enable` are set in `tools/db/milvus-config/user.yaml` |

Qdrant extras (Phase 1's "scalar, product, binary quantization"): run `hnsw` with `--build quant=none|scalar|product|binary` and `--search rescore=0|1`; report them as index `hnsw` with those build params. The metric is always the dot product (`Dot` / `vector_ip_ops` / `IP`).

## Measurement

- Search: one query at a time through the client library, timed around the client call (so the localhost round trip is included; the report must say so). 100 warm-up queries first.
- Load in batches of `--batch` rows (default 2000); `load_s` is the wall time of all inserts plus any flush the database needs before an index build.
- `server_build_s`: from the index-create call until the server reports the index ready (Qdrant: collection status green; pgvector: `CREATE INDEX` returns; Milvus: `index_building_progress` at 100% and the collection loaded).
- Ties and padding: as CONTRACT section 6 (lower ID first where the client can enforce it; pad with -1 / null).

## Tests (`tests/test_<db>.py`)

Run with the database up, against `data/processed/dev` with `--limit 20000` and brute-force truth computed in the test:

1. `reset`, `load` 20,000 rows, `stats()["rows"] == 20000`.
2. For each supported index at default params: recall@10 ≥ the CONTRACT section 9 floor (hnsw ≥ 0.95 at ef=64; ivf ≥ 0.75 at nprobe=8; ivf_pq ≥ 0.45; diskann ≥ 0.90; flat = 1.0).
3. A `bench` subprocess run produces JSON that passes `tools.bench.schema.validate`.
4. The test leaves the collection dropped (`reset` at the end).

The test must finish in under 10 minutes. After the tests, stop the database (`make db-down`).
