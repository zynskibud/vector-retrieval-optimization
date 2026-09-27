# Open questions and defaults taken

Decisions that need the human, the default the session took, and its effect. Newest first. Remove an entry when the human decides.

## 2026-09-26: Two container caps: day (3 CPUs, 6 GB) and TIMING (6 CPUs, 12 GB)

- **Question:** the coordinator's rule (PROTOCOL.md): by day one container at a time at 3 CPUs and 6 GB; a TIMING job on GO at 6 CPUs and up to 12 GB, with the caps in the compose file.
- **Default taken:** `docker-compose.yml` reads `VRO_CPUS`, `VRO_MEM`, `VRO_DB_CPUS` with the day caps as defaults; `scripts/run.sh` exports the TIMING values for every job except `test`. `VRO_THREADS` equals the CPU cap, so tests and builds by day use 3 threads and the sweeps 6. The database containers get 3 CPUs by day and 4 in a sweep (the Phase 2 design). The 6 GB day cap is under the 9 GB that Python IVF needs on the full corpus, so full-corpus work is TIMING only.
- **Effect:** a test suite by day takes about twice as long as at 6 threads. Sweep numbers are unchanged (6 threads, as before). A database test by day runs two containers (the database and dbbench); this is the one exception to one container at a time.

## 2026-09-26: Phase 7 design taken without human review

- **Question:** the backup format for the hand-built indexes and the method per database.
- **Default taken (CONTRACT section 15):** one little-endian `.vro` file format with a JSON header and a section table, shared by the four languages (a Rust file loads in Go, C++, Python), for flat, ivf, hnsw; a loaded index needs no rebuild and returns identical IDs. Databases: Qdrant snapshots, pg_dump / pg_restore (which rebuilds the HNSW index, measured), Milvus cold backup of its data volume because milvus-backup needs object storage that the local-storage setup does not have. Five Opus 5.5 agents.
- **Effect:** the cross-language load is a strong check that the four implementations hold the same graph; the pgvector restore time includes an index rebuild, which the report marks.

## 2026-09-26: Phase 6 design taken without human review

- **Question:** how to measure an embedding cache.
- **Default taken (CONTRACT section 14):** all-MiniLM-L6-v2 on CPU inside the container (downloaded once by the setup service into a volume), a Zipf-distributed request stream over a pool of 5,000 texts, a hashed key that includes the model version so a version change invalidates by miss, an in-process LRU and a Redis 7 container (512 MB, allkeys-lru), FAISS HNSW for the search stage, a runner sweep over backend and capacity. Two Opus 5.5 agents. Running the embedding model on CPU in a container is treated as CPU-BULK, not GPU, because it uses neither Ollama nor MPS; if the coordinator disagrees, the Phase 6 runs move to the GPU class.
- **Effect:** the cache numbers show CPU embedding cost (about 5-20 ms per text on this machine) against a cache hit (microseconds for LRU, about 0.2 ms for Redis on localhost), and the hit rate the workload's repetition allows.

## 2026-09-26: Phase 5 design taken without human review

- **Question:** how deletes, updates, and compaction are defined and measured.
- **Default taken (CONTRACT section 13):** nested random delete sets of 10 / 30 / 50%, an update set of 10% with new vectors that are related to the old ones, per-set ground truth from `tools/data/changes.py`; tombstone deletes (hnswlib's markDelete: deleted nodes still take part in the walk), update = delete + re-insert under the same ID, compaction = rebuild from live rows for hnsw and array cleanup for flat and ivf; databases through delete / update / compact with VACUUM + REINDEX for pgvector and compact() for Milvus. Five Opus 5.5 agents. PQ, IVF_PQ, DiskANN stay out.
- **Effect:** recall after deletes is measured against the truth over the remaining rows, so the numbers show graph damage, not the missing rows; the compaction cost shows as the rebuild time.

## 2026-09-26: Phase 4 design taken without human review

- **Question:** how to measure concurrency for the hand-built indexes and the databases, and what "searches during inserts" means.
- **Default taken (CONTRACT section 12):** closed-loop worker threads (1 to 64) over the query set for a fixed duration inside the same `bench` program, reporting QPS, all latencies, errors, and CPU%; an inserter that adds the last 10% of the rows at a fixed rate during the loop, with a one-thread pass afterwards whose recall must match a static build; lock-free searches over atomic neighbor slots in Go, C++, Rust, with the build's per-node lock stripe for inserts; Python with threads and a lock per node list, GIL-serialized. Only hnsw and the databases take part. Five Opus 5.5 agents.
- **Effect:** the language comparison gains a concurrency axis (thread scaling and tail latency) that the earlier phases could not show; Python's numbers will show the GIL. IVF, PQ, DiskANN stay read-only.

## 2026-09-26: Phase 3 design taken without human review

- **Question:** what "metadata filtering" measures and how the filter reaches each index.
- **Default taken (CONTRACT section 11):** one predicate, `views >= t`, with t chosen for selectivity 0.5, 0.1, 0.01, 0.001; masks and per-filter ground truth produced by `tools/data/filters.py`; a search-time key `filter=` on flat, ivf, hnsw in all four languages and in the three databases; hnsw uses the hnswlib rule (only passing nodes enter the result list, all visited nodes are expanded), with no brute-force fallback, so the low-selectivity degradation is visible; pgvector also runs with a B-tree index on `views` and with `iterative_scan`. Five Opus 5.5 agents: one per language, one for tools and the database clients.
- **Effect:** filtering is measured on 3 index types, not 6. PQ, IVF_PQ, and DiskANN skip Phase 3 (they can be added later with the same key).

## 2026-09-26: Phase 2 design taken without human review

- **Question:** the design of the database phase (services, client interface, what is measured, agents). CLAUDE.md asks for approval before a large change; PROTOCOL.md says to take a default and log it.
- **Default taken:** Qdrant v1.19, pgvector 0.8.6 on Postgres 17, Milvus standalone with embedded etcd and local file storage (MinIO's images are no longer pullable from Docker Hub or quay.io, so the two side containers are gone; this is how Milvus's own standalone_embed.sh runs it), each as a compose service under a profile, on an internal Docker network with no external access, capped at 4 GB. One Python client module per database in `tools/db/` with the same four functions (load, build_index, search, stats). `tools/db/bench.py` uses the same command line and output JSON as the language benches with `language = qdrant | pgvector | milvus`, so the runner, schema check, and report work unchanged. Extra measurements: load time, server-side index build time, data-directory disk use, container memory. Search latency includes a localhost network round trip. Three Opus 5.5 agents, one per database, each stops its database after its tests.
- **Effect:** the database results appear as more lines on the Phase 1 plots. The network round trip adds about 0.1 to 0.3 ms per query to every database number; the write-up must say so. If you want a different database set or client design, say so before the Phase 2 sweep runs.

## 2026-09-25: Full-corpus Python HNSW and DiskANN use the 100k subset

- **Question:** the pure-Python builds take 30 to 60 minutes per run at 1.2M rows (HNSW sequential insert, DiskANN two Vamana passes), and each sweep needs several builds.
- **Default taken (coordinator decision 6c, allowed by CONTRACT section 8):** Python HNSW and DiskANN run on `data/processed/dev` (100k rows) in the full-corpus sweep; the other languages and FAISS run on the full corpus. The report marks these rows with `n = 100000`.
- **Effect:** Python's HNSW and DiskANN latency and recall are not directly comparable with the 1.2M rows of the other languages; the write-up must say so.

## 2026-09-25: Subagent model rules

- **Question:** PROTOCOL.md says Sonnet 5 by default and Opus 5.5 only for hard design or debugging. The user asked this session for Opus 5.5 on every index agent.
- **Default taken:** the coordinator (ai-engineering-6c) said to keep the user's subagent settings and never use Fable 5.1 for subagents. So: Opus 5.5 for index and optimization agents, at most 8 at a time.
- **Effect:** higher token cost per wave than Sonnet 5; same quality of the implementations as Waves 1 to 2b.

## 2026-09-25: Container memory limit

- **Question:** Docker Desktop's VM now has 16 GB. `docker-compose.yml` still caps the container at 6 GB.
- **Default taken:** 6 GB for the dev-set work (enough: the largest dev-set build, Python PQ l2, peaks at 3.2 GB). Raise `mem_limit` to 12 GB before the full-corpus runs (the 1.2M corpus alone is 1.8 GB, and Python IVF peaks at about 9 GB there).
- **Effect:** none on the dev set. The full-corpus runs must not start before the change.

## 2026-09-25: Host results kept as a second environment

- **Question:** the dev-set sweeps for flat, IVF, PQ, HNSW, and 10 IVF_PQ and DiskANN cases ran on the macOS host before isolation.
- **Default taken:** keep them under `results/summary/dev-macos-native/` and `docs/results-dev.md`; rerun everything inside the container into `results/summary/dev/`. FAISS is 1.12.0 in the container and 1.15.1 on the host.
- **Effect:** the write-up gets two environments for the same code; the container numbers are the primary ones.
