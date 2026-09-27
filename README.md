# Vector Retrieval Optimization

Six vector search indexes built by hand in Python, Go, C++, and Rust, checked against FAISS, and compared with Qdrant, pgvector, and Milvus. The data is 1.2 million Wikipedia paragraphs with 384-dimension embeddings.

## Status

| Phase | Topic | Status |
|---|---|---|
| 0 | Setup and data | Done |
| 1 | Indexes by hand, checked against FAISS | In progress: all 24 implementations (6 indexes × 4 languages) done and tested; dev-set sweeps for IVF_PQ and DiskANN running; full-corpus runs next |
| 2 | Three databases | In progress: Qdrant, pgvector, Milvus clients and tests done inside Docker; the database sweeps wait for the machine lock |
| 3 | Metadata filtering | Code done: filter= on flat, IVF, HNSW in all four languages, FAISS, and the three databases; sweeps wait for the machine lock |
| 4 | Concurrency | Code done: closed-loop clients 1..64 and inserts during search for HNSW in all four languages and for the three databases; the load sweep waits for the machine lock |
| 5 | Updates and deletes | Code done: tombstone deletes, updates, and compaction for flat, IVF, HNSW in all four languages and for the three databases, with per-change ground truth; the changes sweep waits for the machine lock |
| 6 | Embedding cache | Code done: all-MiniLM-L6-v2 on CPU in the container, LRU and Redis backends, Zipf and uniform workloads, invalidation by model version; the cache sweep waits for the machine lock |
| 7 | Backup and restore | In progress: the `.vro` index file (flat, IVF, HNSW) is written and read by all four languages, and a file written by one language loads in the other three with identical results; the database snapshot tests are next |
| 8 | Write-up | Not started |

The full plan is in [`docs/plan.html`](docs/plan.html).

## Indexes

| Index | Idea |
|---|---|
| Flat | Compare the query to every vector. Exact. |
| IVF | Cluster the vectors with k-means. Search only the nearest clusters. |
| PQ | Compress each vector into 1-byte codes. Compare with lookup tables. |
| IVF_PQ | IVF plus PQ. |
| HNSW | Layered nearest-neighbor graph. Greedy search along edges. |
| DiskANN | One graph on SSD, with compressed vectors in RAM. |

Each index is written in four languages with the same algorithm and settings. The goal is to see how each language's memory model affects speed, memory use, and tail latency.

## Data

5 of the 145 files of [`maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2`](https://huggingface.co/datasets/maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2) on Hugging Face: about 1.2M paragraphs. 1,000 vectors are held out as queries. The exact top-100 neighbors of each query are the ground truth for recall.

## Layout

```
tools/       Python: data preparation, benchmark runner, metrics, database clients
indexes/     Hand-built indexes: python/, go/, cpp/, rust/
data/        Downloaded and prepared data (not in git)
results/     Benchmark output: raw/ (not in git) and summary/
docs/        The plan, and notes on the algorithms and the results
```

## Requirements

- [uv](https://docs.astral.sh/uv/) for Python
- Go, a C++17 compiler with CMake, and Rust (added in Phase 1)
- Docker (added in Phase 2)
- About 15 GB of free disk

## Quick start

```bash
uv sync
uv run python -m tools.data.prepare
uv run python -m tools.data.ground_truth
uv run python -m tools.data.verify      # 10 checks on the outputs
uv run python -m tools.data.subset      # 100k-row dev dataset for fast tests
```

The download is about 2.8 GB. Prepared files: `vectors.npy` (1.7 GB), `metadata.parquet` with the paragraph text (378 MB), `queries.npy`, `query_meta.parquet`, `ground_truth.npy`, and `ground_truth_scores.npy`.
