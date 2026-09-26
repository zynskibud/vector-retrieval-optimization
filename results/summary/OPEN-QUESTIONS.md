# Open questions and defaults taken

Decisions that need the human, the default the session took, and its effect. Newest first. Remove an entry when the human decides.

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
