# Open questions and defaults taken

Decisions that need the human, the default the session took, and its effect. Newest first. Remove an entry when the human decides.

## 2026-09-26: Phase 2 design taken without human review

- **Question:** the design of the database phase (services, client interface, what is measured, agents). CLAUDE.md asks for approval before a large change; PROTOCOL.md says to take a default and log it.
- **Default taken:** Qdrant v1.19, pgvector 0.8.6 on Postgres 17, Milvus standalone (with etcd and MinIO), each as a compose service under a profile, on an internal Docker network with no external access, capped at 4 GB. One Python client module per database in `tools/db/` with the same four functions (load, build_index, search, stats). `tools/db/bench.py` uses the same command line and output JSON as the language benches with `language = qdrant | pgvector | milvus`, so the runner, schema check, and report work unchanged. Extra measurements: load time, server-side index build time, data-directory disk use, container memory. Search latency includes a localhost network round trip. Three Opus 5.5 agents, one per database, each stops its database after its tests.
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
