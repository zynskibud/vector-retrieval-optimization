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
│       │                          #   ground_truth.npy, ground_truth_scores.npy
│       └── dev/                   #   same files for a 100k-row sample, plus dev_ids.npy
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
- **Develop and test on `data/processed/dev/`** (100k vectors, same 1,000 queries, own ground truth). Use the full data only for the benchmark runs that the tools runner starts.

## Rules for the hand-built indexes

- **No vector search libraries** in `indexes/`: no FAISS, hnswlib, Annoy, ScaNN, or usearch. The index logic must be written by hand.
- **Allowed libraries:** the standard library of each language, plus:
  - Python: NumPy, for arrays and for the distance inner loop.
  - Rust: `rayon` for threads, `serde_json` for output.
  - C++: a JSON writer. Threads come from `std::thread` or OpenMP.
  - Go: the standard library only.
- **Write the .npy reader by hand** in each language. The format is a short header followed by raw little-endian float32 data.
- **The same algorithm and the same settings** in all four languages. If one language needs a different approach, write down why in `docs/`.
- **All six indexes exist in all four languages.** No stub remains; `bench --index <name>` runs every index. DiskANN writes `<out>.diskann` (4 KB per row: 0.4 GB on dev, 4.9 GB on the full corpus) next to the output JSON; delete it after a run.
- **Follow `indexes/CONTRACT.md`.** It fixes the command line, the output JSON, the `.npy` reader, the PRNG, every index algorithm and its parameter names, the measurement rules, the directory layout, and the required tests. Read it in full before you write index code.

## Commands

```bash
uv sync                                   # host: only for tools.data.* (download needs network)
uv run python -m tools.data.prepare     # Phase 0: download and prepare data (host)
uv run python -m tools.data.ground_truth
uv run python -m tools.data.verify       # Phase 0 checks; must print 10 PASS lines
uv run python -m tools.data.subset       # 100k-row dev dataset in data/processed/dev/
uv run python -m tools.data.verify --data data/processed/dev
uv run python -m tools.data.filters [--data DIR]   # Phase 3: filter masks and per-filter ground truth (host)
uv run python -m tools.data.changes [--data DIR]   # Phase 5: delete sets, update set, and their ground truth (host)

uv run python -m tools.bench.schema FILE.json           # validate a bench output against CONTRACT section 3
uv run python -m tools.bench.faiss_ref --index hnsw --data data/processed/dev --out r.json --search ef=64   # FAISS reference, same CLI as bench
uv run python -m tools.bench.runner --data data/processed/dev --languages faiss,python --indexes flat,ivf [--repeat 3] [--dry-run]   # sequential runs, median of 3, refuses when load > 2 -> results/raw/<data>/
uv run python -m tools.bench.report --data data/processed/dev   # results/summary/<data>/: results.csv, results.md, <index>.png
uv run pytest indexes/python/tests tools/bench/tests -q

(cd indexes/go && go vet ./... && go test ./... && go build -o bin/bench ./cmd/bench)
cmake -S indexes/cpp -B indexes/cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build indexes/cpp/build -j && ctest --test-dir indexes/cpp/build
(cd indexes/rust && cargo build --release && cargo test --release)   # inside the folder, so .cargo/config.toml applies
```

Add each new command here when it is created.

## Isolation: everything runs in the container

No toolchain runs on the host. Every build, test, benchmark, and report runs inside the Docker container defined in `docker-compose.yml`, through the `Makefile`:

```bash
make setup                      # one time: image, deps (with network), builds (without)
make build                      # rebuild Go, C++, Rust after code changes
make test                       # all suites; or run one, e.g.:
docker compose run --rm bench uv run --frozen pytest indexes/python/tests/test_ivf.py -q
docker compose run --rm bench sh -c 'cd indexes/rust && cargo test --release --test hnsw'
docker compose run --rm bench sh -c 'cd indexes/go && go test -timeout 60m ./hnsw/'
docker compose run --rm bench ctest --test-dir indexes/cpp/build -R '^hnsw$' --output-on-failure
docker compose run --rm bench indexes/rust/target/release/bench --index flat --data data/processed/dev --out results/raw/dev/x.json
make bench ARGS="--data data/processed/dev --languages rust,cpp --indexes flat --repeat 3"
make report ARGS="--data data/processed/dev"
make db-up DB=qdrant && make db-test DB=qdrant; make db-down DB=qdrant        # Phase 2/3: one database at a time
make dbbench ARGS="--db qdrant --index hnsw --data data/processed/dev --out results/raw/dev/x.json --search ef=64"
make load ARGS="--data data/processed/dev --languages rust,cpp,go,python"     # Phase 4: hnsw load runs, clients 1..64 (timing: lock)
make load-db ARGS="--data data/processed/dev --languages qdrant"               # Phase 4: databases, dbbench container
docker compose --profile db run --rm dbbench uv run --frozen python -m tools.load.bench --db qdrant --index hnsw --data data/processed/dev --out results/raw/dev/x.json --clients 8 --duration 20
scripts/run.sh dev-sweep | db-sweep | load-sweep                               # timing jobs, only after the coordinator's GO
```

Rules:

- The repository is mounted **read-only** in the container. Only `results/summary/` is writable on the host. Raw results go to the `raw` volume (`results/raw` inside the container), build outputs and `.venv` to named volumes, scratch to `/tmp` (tmpfs, 2 GB). Code that runs inside cannot change the repository.
- The `bench` service has **no network**. The `setup` service has network and is used only for `uv sync` and `cargo fetch`.
- Resource limits: `docker-compose.yml` defaults to the day caps (one container at a time, 3 CPUs, 6 GB); `scripts/run.sh` sets the TIMING caps (6 CPUs, 12 GB, databases 4 CPUs) for a job on the coordinator's GO. `VRO_THREADS` follows the CPU cap. Benchmarks measure the Linux VM (Ubuntu 24.04 arm64): NumPy uses OpenBLAS, not Accelerate; FAISS is 1.12.0 (the last with a Linux arm64 wheel); DiskANN's `nocache` uses `O_DIRECT` on the `raw` volume. Numbers from the earlier host runs are kept under `results/summary/dev-macos-native/` and are not comparable.
- Agents: use the commands above. Do not call `uv run`, `go`, `cargo`, `cmake`, or a bench binary on the host. If a container command fails because a dependency is missing, report it; do not install anything on the host.
- Output files written inside the container that the host must see (result JSON for a report) are read through `make report` or `docker compose run --rm bench cat results/raw/...`.

## Heavy jobs: the machine lock and the coordinator

This machine is shared with two other projects. A coordinator session schedules jobs by resource class through `../.coord/PROTOCOL.md` (read it; `../.coord/STATUS.md` names the coordinator). A benchmark timing run is class TIMING: it runs alone on the machine under `../.coord/timing.lock`, with the GPU lock (`gpu.lock`, old name `heavy.lock`) free and the load under 2.

- Every timing run starts through `scripts/run.sh <job>` and never by hand. The script runs `scripts/preflight.sh`, takes `../.coord/timing.lock` with owner `vector-retrieval <job> <ISO time>`, runs the job detached under `caffeinate -i`, logs to `results/logs/`, and releases the lock when the job ends or is stopped (`scripts/run.sh --stop`).
- Start a timing run only after the coordinator sends `GO <job>`. Report to the coordinator when a job starts, ends, or fails, with the log path.
- Tests and builds are CPU-BULK or LIGHT work: run them without a lock, never while a timing run holds `timing.lock`.
- Any container this project starts outside a timing run stays under 4 GB and is stopped before a sweep.
- `scripts/status.sh` shows the lock, running containers, the newest log, and the result counts.
- Decisions that need the human, with the default taken, go in `results/summary/OPEN-QUESTIONS.md`.

## Machine limits

- Apple M4, 24 GB RAM, about 48 GB free disk. Docker Desktop's VM memory must be set to 16 GB in Docker Desktop > Settings > Resources for the full-corpus runs (8 GB is enough for the dev set).
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
