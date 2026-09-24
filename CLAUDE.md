# CLAUDE.md

Instructions for Claude and for subagents that work in this repository. Read this file, then `docs/plan.html`, before you change code.

## Project

Benchmark vector search on 1.2M Wikipedia vectors (384 dimensions, `all-MiniLM-L6-v2`). The work has three parts:

1. Six indexes built by hand in four languages: Python, Go, C++, and Rust. FAISS is the reference.
2. The same index types in three self-hosted databases: Qdrant, pgvector, and Milvus.
3. Tests of filtering, concurrency, updates and deletes, an embedding cache, and backup and restore.

`docs/plan.html` has the full plan in 9 phases (0 to 8). Each phase has a "Done when" check. Do not start a phase until the previous phase it depends on meets its check.

## The user

The user is learning these topics. The user directs the work and reads every design decision. Claude writes the code.

- Explain design decisions in standard technical terms: node, edge, neighbor, array, heap memory. Do not use analogies.
- Define each new term the first time it appears.
- Before a large change, state the design and wait for approval.
- Report results as measured numbers. If a test fails, say so and show the output.

## Directory layout

```
.
├── CLAUDE.md, README.md
├── pyproject.toml, uv.lock        # Python project (uv): tools and Python indexes
├── docker-compose.yml             # Phase 2: one service per database
├── data/                          # gitignored
│   ├── raw/                       #   downloaded Parquet shards
│   └── processed/                 #   vectors.npy, queries.npy, metadata.parquet, query_meta.parquet,
│                                  #   ground_truth.npy, ground_truth_scores.npy
├── tools/                         # Python package: everything that is not an index
│   ├── data/                      #   Phase 0: prepare.py, ground_truth.py
│   ├── bench/                     #   runner, recall and latency metrics, plots
│   ├── db/                        #   Phase 2: one client module per database
│   ├── load/                      #   Phase 4: concurrent load generator
│   ├── cache/                     #   Phase 6: embedding cache
│   └── backup/                    #   Phase 7: backup and restore tests
├── indexes/                       # hand-built indexes, one folder per language
│   ├── CONTRACT.md                #   rules that all four languages follow
│   ├── python/
│   ├── go/
│   ├── cpp/
│   └── rust/
├── results/
│   ├── raw/                       #   gitignored: one JSON file per run
│   └── summary/                   #   committed: tables and plots
└── docs/                          #   plan.html, and notes on each algorithm and result
```

Inside each language folder, use one file or module per index, with the same names in every language: `flat`, `ivf`, `pq`, `ivf_pq`, `hnsw`, `diskann`. Shared code goes in `npy` (the .npy reader), `distance`, and `kmeans`. Each language has one `bench` program as its entry point.

## Fixed conventions

These hold everywhere. Do not change them without approval from the user.

- **Vector ID** = row index in `data/processed/vectors.npy`. Query ID = row index in `queries.npy`.
- **Vectors are L2-normalized.** Similarity is the dot product (inner product). Higher is better.
- **Ground truth** = exact top-100 by brute force, in `ground_truth.npy`, shape (1000, 100), int64. Recall@10 uses the first 10 columns.
- **Data type:** float32 for vectors. int64 for IDs in files.
- **Recall is computed only by the tools**, never inside an index program. This keeps the measurement the same for all languages.
- **Random seeds are fixed** and passed in, so every run is reproducible.

## Rules for the hand-built indexes

- **No vector search libraries** in `indexes/`: no FAISS, hnswlib, Annoy, ScaNN, or usearch. The index logic must be written by hand.
- **Allowed libraries:** the standard library of each language, plus:
  - Python: NumPy, for arrays and for the distance inner loop.
  - Rust: `rayon` for threads, `serde_json` for output.
  - C++: a JSON writer. Threads come from `std::thread` or OpenMP.
  - Go: the standard library only.
- **Write the .npy reader by hand** in each language. The format is a short header followed by raw little-endian float32 data.
- **The same algorithm and the same settings** in all four languages. If one language needs a different approach, write down why in `docs/`.
- **Follow `indexes/CONTRACT.md`** for the command-line arguments, the output JSON, and the index file format.

## Commands

```bash
uv sync                                   # install the Python environment
uv run python -m tools.data.prepare     # Phase 0: download and prepare data
uv run python -m tools.data.ground_truth
uv run python -m tools.data.verify       # Phase 0 checks; must print 10 PASS lines
```

Add each new command here when it is created.

## Machine limits

- Apple M4, 24 GB RAM, about 44 GB free disk. Docker Desktop gives containers about 7.6 GB of RAM.
- Run one database container at a time.
- Delete old data before a large download. Check free disk with `df -h ~` first.

## Git

- The repository is public: github.com/zynskibud/vector-retrieval-optimization
- Never commit `data/` or `results/raw/`.

### Commit after every change

When a change is complete, commit it and push it at once. Do not wait for the user to ask. A change is one logical unit of work, for example one index in one language, one script, one fix, or one doc update.

Before each commit:

1. Run the checks for the code that changed: build, tests, and `tools.data.verify` when data code changed. If a check fails, fix it first. Never commit broken code.
2. Run `git status` and `git diff --staged`. Stage only the files of this change. Never use `git add -A` without reading the list.
3. Confirm that no data, build output, secrets, or large files are staged.

Commit message format:

```
<area>: <what changed, imperative, max 72 characters>

<why, and anything a reviewer must know>
```

`<area>` is one of: `tools`, `indexes/python`, `indexes/go`, `indexes/cpp`, `indexes/rust`, `db`, `docs`, `repo`. Example: `indexes/rust: add IVF build and search`.

Rules:

- One logical change per commit. Do not mix a fix with a new feature.
- Update `README.md` status and `CLAUDE.md` commands in the same commit as the change that makes them true.
- Never rewrite pushed history: no `--amend`, rebase, or force push after a push, unless the user asks.
- Subagents do not commit and do not push. Each subagent reports its changed files to the main session. The main session reviews, runs the checks, and commits each change separately.
