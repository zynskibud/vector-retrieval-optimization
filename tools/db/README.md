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
    data_dir: Path | None                        # set by bench.py (client.data_dir = args.data) before load; filters.json is read from it
    def connect(self) -> None: ...               # host names: qdrant, pgvector, milvus (compose service names)
    def reset(self) -> None: ...                 # drop the collection/table if it exists
    def load(self, vectors: np.ndarray, meta: pa.Table, batch: int) -> float:
        ...                                      # insert rows; ID = row index; returns seconds
    def build_index(self, index: str, params: dict) -> float:
        ...                                      # create the index, wait until it is built; returns seconds
    def search(self, query: np.ndarray, k: int, params: dict) -> tuple[list[int], list[float]]:
        ...                                      # one query; IDs best first, padded with -1; scores as the DB reports them (see Scores)
    # Phase 4 (CONTRACT section 12.3, tools/load/bench.py):
    def attach(self, index: str) -> None: ...    # a new connection to an index another connection built: set the search state only
    def insert(self, vectors: np.ndarray, meta_rows: pa.Table, ids: list[int]) -> float:
        ...                                      # add rows to the built index (ID = row index); returns seconds
    def finish_inserts(self) -> None: ...        # make every inserted row searchable before the after-inserts pass
    # Phase 5 (CONTRACT section 13.4), after build_index and before the searches:
    def delete(self, ids: np.ndarray) -> float: ...                  # delete rows by ID, searchable state on return; seconds
    def update(self, ids: np.ndarray, vectors: np.ndarray, meta_rows: pa.Table) -> float: ...  # new vectors, same IDs; seconds
    def compact(self) -> dict: ...                                   # the database's repair; {"compact_s": s, ...detail}
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
| filter (flat, ivf, hnsw) | filter = none\|top50\|top10\|top1\|top01 (search) | payload index on `views` (float, created at load); `Filter(must=[FieldCondition("views", Range(gte=t))])` | `WHERE views >= t`; build `views_index=0\|1` (B-tree on views); search `iterative=0\|1` (`SET LOCAL hnsw.iterative_scan` / `ivfflat.iterative_scan` = `relaxed_order` or `off`) | `views` is a FLOAT field; search `filter="views >= t"` |
| diskann | r, l_build, l, beam | — | — | `DISKANN` (search `search_list = l`); `queryNode.enableDisk` and `common.diskIndex.enable` are set in `tools/db/milvus-config/user.yaml` |

Qdrant extras (Phase 1's "scalar, product, binary quantization"): run `hnsw` with `--build quant=none|scalar|product|binary` and `--search rescore=0|1`; report them as index `hnsw` with those build params. The metric is always the dot product (`Dot` / `vector_ip_ops` / `IP`).

## Filters (Phase 3, CONTRACT section 11)

- The threshold `t` of each filter comes from `<data>/filters.json` through `base.views_min(client.data_dir, name)`. `bench.py` sets `client.data_dir = args.data` before `connect`; a test sets it before the first filtered search.
- `searches[i].extra.filter_rows` = rows among the loaded rows that pass the filter (from `filter_<name>.npy`, so it is right with `--limit`). It is present when the search params carry `filter`.
- pgvector only: `views_index` and `iterative` are extra params (`base.BUILD_EXTRA`, `base.SEARCH_EXTRA`). With `iterative=1` the rows can come back slightly out of order (`relaxed_order`), so the client re-sorts them by score.
- Test: hnsw with `filter=top10` at ef=64 has recall@10 >= 0.85 against the exact filtered top-10 of the 20,000 loaded rows (`tests/filtering.py`), and every returned ID passes the filter; `top01` returns only passing IDs. pgvector runs the four combinations of `views_index` and `iterative`.

## Measurement

- Search: one query at a time through the client library, timed around the client call (so the localhost round trip is included; the report must say so). 100 warm-up queries first.
- Load in batches of `--batch` rows (default 2000); `load_s` is the wall time of all inserts plus any flush the database needs before an index build.
- `server_build_s`: from the index-create call until the server reports the index ready (Qdrant: collection status green; pgvector: `CREATE INDEX` returns; Milvus: `index_building_progress` at 100% and the collection loaded).
- Ties and padding: as CONTRACT section 6 (lower ID first where the client can enforce it; pad with -1 / null).

## Load runs (Phase 4, CONTRACT section 12)

`python -m tools.load.bench` takes the `tools.db.bench` flags plus `--clients C` (default 1), `--duration S` (default 20), and `--insert-rate R` (default 0). Each of the C worker threads opens its own connection (`get_client`, `connect`, `attach`). The protocol and the JSON keys are in the module docstring and CONTRACT section 12. `make load-db ARGS="--data data/processed/dev --languages <db>"` runs the runner's load sweep for one database.

Inserts during searches (`--insert-rate`), one batch of 100 rows per `insert` call:

- Qdrant: `upsert(wait=True)`. The rows are searchable when the call returns; the optimizer indexes them later, and Qdrant scans unindexed segments exactly until then. `finish_inserts` does nothing.
- pgvector: one binary `COPY` per batch, in one transaction. Postgres adds each row to the HNSW or IVFFlat index inside the `COPY`, so the rows are searchable at commit. `finish_inserts` does nothing.
- Milvus: `insert` without a flush. The rows go to a growing segment, which Milvus searches by brute force. With the default `Bounded` consistency, a search can miss rows inserted in the last moments: in a test, 88 of 100 rows were found as their own top-1 right after the insert, and 100 of 100 with `consistency_level="Strong"`, without a flush. `finish_inserts` flushes once and sets `Strong` consistency for the after-inserts pass.

If the database inserts slower than R, the inserter continues after the loop until every row is in. `extra.inserted_during_loop` counts the rows added inside the loop.

## Updates, deletes, compaction (Phase 5, CONTRACT section 13)

`python -m tools.db.bench ... --delete del10|del30|del50 | --update upd10 [--compact]`. The change runs after the build and before the warm-up. `base.delete_ids` / `base.update_rows` read `delete_<name>.npy` and `update_<name>_{ids,vectors}.npy` and keep only IDs below `n` (so `--limit` works). JSON:

- `extra`: `delete_s`, `deleted_rows` or `update_s`, `updated_rows`; with `--compact` `compact_s` and `compact_detail` (the steps and their seconds). `extra.disk_bytes` (and `build.index_bytes`) are the stats after the build; `extra.disk_bytes_after`, `index_bytes_after`, `table_bytes_after`, `segments_count_after`, `rows_after` are the stats when the searches start (after the compaction with `--compact`, else right after the change). The full `stats()` dicts are `extra.stats_changed` and `extra.stats_after`.
- Every search run carries `search_params.deleted` or `updated`, and `compacted` = 0 or 1. The report scores it against `ground_truth_<name>.npy` and counts `deleted_returned`.

Each database's own background compaction is switched off for the change, so a run without `--compact` measures the tombstone state:

| | delete | update | compact | disk_bytes |
|---|---|---|---|---|
| Qdrant | `update_collection(deleted_threshold=1.0)` (vacuum off), then `delete(PointIdsList)` in batches of 10,000, `wait=True`. The point is a set bit in the segment's deleted bit set; the HNSW node stays and searches walk through it. | vacuum off, `upsert` (same ID, same payload) in batches of 2,000; the old copy is marked deleted, the new one lands in an appendable segment; the time includes the wait until green and every vector indexed. | `update_collection(deleted_threshold=0.01, vacuum_min_vector_number=100)`, wait for green and all vectors indexed. The vacuum optimizer rebuilds each segment with deleted points from its live points (a new HNSW graph). With the defaults (0.2, 1000) Qdrant does this on its own after del30. | `GET /telemetry?details_level=10`: segment `disk_usage_bytes` if non-zero, else `vectors_size_bytes + payloads_size_bytes` of the segments (`disk_bytes_source` says which). v1.19.1 reports `disk_usage_bytes = 0`, and `vectors_size_bytes` counts live vectors only, so this number drops at the delete, not at the compaction; `deleted_vectors` and `segments_count` show the compaction. |
| pgvector | `ALTER TABLE items SET (autovacuum_enabled = false)`, then `DELETE ... WHERE id = ANY(...)` in chunks of 10,000, one transaction. | autovacuum off, `COPY` into a temp table, one `UPDATE ... FROM`. Each row gets a new tuple and a new index entry; the old entry stays. | `VACUUM (ANALYZE) items` (`vacuum_s`), then `REINDEX INDEX items_hnsw` / `items_ivf` (`reindex_s`), autovacuum reset. | `pg_total_relation_size('items')`; also `table_bytes` (`pg_relation_size`), `index_bytes`, `dead_tuples`. |
| Milvus | `collection.autocompaction.enabled = false`, `delete(filter="id in [...]")` in chunks of 10,000, one `flush`; searches then use Strong consistency. | same property, `upsert` in batches of 2,000, `flush`, wait until the new segment is indexed and loaded. | `compact(is_l0=True)` (moves the L0 deletes into the segments), then `compact()` (mix compaction rewrites the segments without deleted rows), each polled with `get_compaction_state`; then wait until indexed and loaded. | Prometheus metrics on port 9091: `milvus_datacoord_stored_binlog_size` (Flushed) + `milvus_datacoord_stored_index_files_size`. The index files of dropped segments count until garbage collection, so `disk_bytes_after` can be larger than `disk_bytes`. Also `segments_count`, `segment_rows`. |

What the tombstone state means for each graph index:

- Qdrant: the deleted node stays in the HNSW graph of its segment. A search expands it and follows its edges, and drops it from the result (the same as hnswlib `markDelete`).
- pgvector: the HNSW graph keeps the dead tuple's element. A search expands it, and the heap visibility check drops it, so the `ef_search` candidates hold fewer live rows. `VACUUM` marks the elements deleted and repairs their neighbors' edges (slow: 111 s for 6,076 deleted rows of 20,000); the index file shrinks only at `REINDEX`.
- Milvus: sealed segments are immutable. The delete is a bit set applied at search time; the mix compaction writes new segments and builds a new index for each.

`make changes-db ARGS="--data data/processed/dev --languages <db>"` runs the runner's changes sweep (CONTRACT 13; `chg-<db>-<index>-<change>[-compact].json`).

## Backup and restore (Phase 7, CONTRACT section 15.3)

`tools/backup/bench.py` loads and builds, runs the 1,000 queries (search run 1), backs up, drops, restores, runs the queries again (search run 2, `search_params.phase = "after_restore"`), and writes CONTRACT section 3 JSON with `extra.backup_s`, `backup_bytes`, `restore_s`, `rebuild_needed`, `cold`, `rows_before`, `rows_after`, `restore_identical`, `ids_equal_fraction`, `ids_overlap`. Cases: flat, ivf, hnsw where the database has them (Qdrant: flat, hnsw).

The dbbench container has no Docker socket, so the work runs in stages, started on the host by `scripts/backup_db.sh` (`make backup-db DB=<db> ARGS="--data DIR"`):

| Database | Method | Where | rebuild_needed |
|---|---|---|---|
| Qdrant | `create_snapshot`, HTTP download to `<out stem>.snapshot` on the raw volume, `delete_collection`, upload with `POST /collections/vro/snapshots/upload?priority=snapshot`, wait green | one dbbench process, `--stage all` | false if `indexed_vectors_count` = points right after the upload |
| pgvector | `pg_dump -Fc -t items` to `/tmp` in the pgvector container, `DROP TABLE items`, `pg_restore` | host: `docker compose exec pgvector` between `--stage before` and `--stage after` | true for ivf and hnsw (the dump has no index pages), false for flat |
| Milvus | cold: stop milvus, `tar` the milvus-data volume to `<out stem>.tar` on the raw volume, start, wait healthy; restore = stop, wipe the volume, untar, start, wait healthy | host: `docker compose stop/up` and a `docker run --rm vro-bench:latest` container on the two volumes, between the stages | false (the volume holds the index files); `cold = true` |

Stage files: `--stage before` writes `<out stem>.stage1.json` (the document with search run 1) on the raw volume. The host step times itself and passes its numbers as `--stage after --host-json '{...}'`; stage after reconnects (`client.reopen(index)`: Milvus loads the collection, and those seconds are added to `restore_s`; pgvector runs `ANALYZE`), runs search run 2, merges, writes `<out>`, and deletes the stage file. The dump and the tar are deleted after the restore. `backup()` and `restore()` on the pgvector and Milvus clients raise `HostStepRequired` inside dbbench (pgvector uses `pg_dump` if it is on PATH).

pgvector seeds neither its IVFFlat k-means sample nor its HNSW levels, so the index that `pg_restore` builds differs from the first one and `restore_identical` is false for ivf and hnsw. The test (`tools/backup/tests/test_db_roundtrip.py`, `make backup-test DB=<db>`, 20,000 rows) then requires recall@10 after the restore within 0.02 of recall@10 before (exact top 10 over the loaded rows), and `restore_identical` for every case without a rebuild.

## Tests (`tests/test_<db>.py`)

Run with the database up, against `data/processed/dev` with `--limit 20000` and brute-force truth computed in the test:

1. `reset`, `load` 20,000 rows, `stats()["rows"] == 20000`.
2. For each supported index at default params: recall@10 ≥ the CONTRACT section 9 floor (hnsw ≥ 0.95 at ef=64; ivf ≥ 0.75 at nprobe=8; ivf_pq ≥ 0.45; diskann ≥ 0.90; flat = 1.0).
3. A `bench` subprocess run produces JSON that passes `tools.bench.schema.validate`.
4. The test leaves the collection dropped (`reset` at the end).
5. `test_load` (Phase 4, `tests/load.py`): hnsw ef=64, `--clients 8 --duration 5`: zero errors, qps > 0, first-pass recall >= 0.95. Then `--clients 4 --duration 10 --insert-rate 2000` (build on 18,000): zero errors, `inserted_rows` = 2,000, after-inserts recall within 0.01 of a static 20,000-row build (both computed in the test). Run: `pytest tools/db/tests/test_<db>.py -k test_load`.

6. `test_changes` (Phase 5, `tests/changes.py`), hnsw at ef=64 on 20,000 rows: after `del30` no deleted ID is returned and recall@10 against the remaining-rows truth is at most 0.03 below the recall before the delete; `disk_bytes` is reported before and after `compact()`, and the compacted recall is within 0.01 of a fresh build on the remaining rows (CONTRACT 13.5); after `upd10`, 100 sampled updated rows are the top-1 for their new vectors. Run: `pytest tools/db/tests/test_<db>.py -k changes`.

The test must finish in under 10 minutes. After the tests, stop the database (`make db-down`).
