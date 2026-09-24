# Index contract

Rules that every hand-built index follows, in all four languages. The benchmark runner in `tools/bench/` drives each language's `bench` program through this interface. If an implementation and this file disagree, this file wins. Change this file only with the user's approval.

Terms: **corpus** = the vectors to index. **query** = a vector to search for. **k** = number of results per query. **ID** = row index of a vector in `vectors.npy`, starting at 0.

## 1. Data

All files are in one data directory, either `data/processed/` (full, 1,211,690 rows) or `data/processed/dev/` (100,000 rows). The `bench` program takes the directory as an argument and reads only two files from it:

| File | Type | Shape | Meaning |
|---|---|---|---|
| `vectors.npy` | float32 | (N, 384) | The corpus. Row i has ID i. |
| `queries.npy` | float32 | (Q, 384) | The queries. Q = 1,000. |

All vectors are L2-normalized (length 1). **Similarity is the dot product.** Higher is better. Do not use Euclidean distance. Do not normalize again.

`ground_truth.npy` (int64, (Q, 100), best first) exists in the same directory. The `bench` program **must not read it**. Tests may read it to assert correctness (section 9).

### 1.1 The .npy format

Each language writes its own reader. The files use NumPy format version 1.0:

1. Bytes 0–5: the magic string `\x93NUMPY`.
2. Byte 6: major version (1). Byte 7: minor version (0).
3. Bytes 8–9: header length as an unsigned 16-bit little-endian integer, `HLEN`.
4. Bytes 10 to 10+HLEN: an ASCII Python dict, padded with spaces and ending in `\n`, for example `{'descr': '<f4', 'fortran_order': False, 'shape': (1211690, 384), }`.
5. The rest of the file: the array data, row-major (C order), little-endian.

`descr` is `<f4` for float32 and `<i8` for int64. The reader must check `descr`, check `fortran_order` is `False`, parse `shape`, and fail with a clear message otherwise. Data offset = 10 + HLEN. The total header (10 + HLEN) is always a multiple of 64 bytes.

Load `vectors.npy` into one contiguous array in RAM (row-major float32). DiskANN is the exception (section 6.6).

## 2. Command line

Each language has one program, `bench`, that builds an index, runs the queries, and writes one JSON file. Build and search happen in the same process, so a build is never repeated for a parameter sweep.

```
bench --index <name> --data <dir> --out <file.json> [options]

--index NAME        flat | ivf | pq | ivf_pq | hnsw | diskann
--data DIR          data directory (section 1)
--out FILE          output JSON path (section 3)
--k INT             results per query. Default 10.
--build KEY=VALUE   build parameter. Repeatable. See section 6 for names.
--search KEY=VALUE  search parameter set. Repeatable. Each occurrence is one
                    search run over all queries. May hold several keys,
                    comma-separated: --search nprobe=8,rerank=0
--threads INT       threads for build. Default: number of CPU cores.
--seed INT          seed for every random choice. Default 42.
--warmup INT        queries run before timing starts, not reported. Default 100.
--limit INT         use only the first INT corpus rows. Default: all. For tests.
```

Example, one HNSW build and four search settings:

```
bench --index hnsw --data data/processed --out results/raw/rust-hnsw.json \
      --build m=16 --build ef_construct=100 \
      --search ef=16 --search ef=32 --search ef=64 --search ef=128
```

Unknown index names, unknown parameter keys, and missing required parameters are errors: print the message to stderr and exit with code 2. Any other failure exits with code 1. Success exits with code 0 and prints nothing to stdout except, optionally, progress lines to stderr.

If `--search` is not given, run once with the defaults in section 6.

## 3. Output JSON

One file per `bench` invocation. Keys and types are exact. Extra keys are allowed under `"extra"` only.

```json
{
  "contract_version": 1,
  "language": "rust",                  // python | go | cpp | rust | faiss (the reference, tools/bench/faiss_ref.py)
  "index": "hnsw",
  "data_dir": "data/processed",
  "n": 1211690,                        // corpus rows used (after --limit)
  "dim": 384,
  "q": 1000,
  "k": 10,
  "threads": 10,
  "seed": 42,
  "build_params": {"m": 16, "ef_construct": 100},   // every build param, defaults filled in
  "build": {
    "train_s": 0.0,                    // time in train step (0 for indexes with no train)
    "add_s": 412.7,                    // time to insert all vectors
    "total_s": 412.7,                  // train_s + add_s
    "peak_rss_mb": 3120.5,             // peak resident memory of the process so far
    "index_bytes": 214380000           // memory of the index structure only, computed, not measured (section 4)
  },
  "searches": [
    {
      "search_params": {"ef": 16},     // every search param, defaults filled in
      "ids": [[12, 9981, ...], ...],   // Q rows, k IDs each, best first; -1 pads a short result
      "scores": [[0.91, 0.90, ...], ...],  // same shape, the index's own score for each result
      "latency_ms": [0.412, 0.398, ...],   // Q values, one per query, in query order
      "total_s": 0.41,                 // wall time of the timed loop
      "qps": 2439.0,                   // q / total_s
      "distance_computations": 3120,   // mean per query, if the index counts them; else null
      "extra": {}                      // per-search-run counters, mean per query, e.g. diskann disk_reads, disk_bytes_read
    }
  ],
  "machine": {"os": "darwin", "arch": "arm64", "cpu": "Apple M4", "cores": 10},
  "extra": {}
}
```

`ids` must be int, `scores` float. Write the JSON with the language's standard library or the allowed JSON library. Floats print with enough digits to round-trip (default of each library is fine).

## 4. Measurement rules

- **Timing** uses a monotonic wall clock: `time.perf_counter`, `time.Now`, `std::chrono::steady_clock`, `std::time::Instant`.
- **Search latency** is per query. Run the queries one at a time, in order, on **one thread**, after the warm-up queries. Time only the search call: not the JSON writing, not the result copying into the output list.
- **Warm-up:** run the first `--warmup` queries once, untimed. Then run all Q queries timed. The warm-up runs use the first search parameter set.
- **Build time:** `train_s` covers everything that learns from data before insertion (k-means, codebooks). `add_s` covers the insertion of all vectors, including graph construction and encoding. Loading the `.npy` file is not timed.
- **`peak_rss_mb`:** the process's peak resident set size after the build, in MB (1 MB = 2^20 bytes). Use `getrusage(RUSAGE_SELF).ru_maxrss` (bytes on macOS, KB on Linux; convert). Python: `resource.getrusage`. Go: `syscall.Getrusage`. The value includes the 1.8 GB corpus array, so `index_bytes` is reported as well.
- **`index_bytes`:** the memory of the index structure itself, computed from its contents. Rules per index:
  - flat: 0 (the corpus array is the index).
  - ivf: centers (nlist × dim × 4) + one int32 or int64 ID per vector in the lists (state which in `extra`).
  - pq: codebooks (m × 256 × (dim/m) × 4) + codes (N × m bytes).
  - ivf_pq: ivf part + pq part.
  - hnsw: sum over all layers of (number of edges × 4 bytes) + per-node level bookkeeping.
  - diskann: bytes held in RAM only (PQ codes + codebooks + entry point), plus the on-disk file size reported separately in `extra.disk_bytes`.
- **`distance_computations`:** if the index counts dot products per query, report the mean. If not, `null`. HNSW, DiskANN, and IVF must count. Flat reports N.

## 5. Determinism

Every random choice uses the `--seed` value and the same PRNG in all four languages, so the four implementations can be compared choice by choice:

**PRNG: SplitMix64.** State `s` (uint64) starts at `seed`. Each call:

```
s  = s + 0x9E3779B97F4A7C15
z  = s
z  = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
z  = (z ^ (z >> 27)) * 0x94D049BB133111EB
return z ^ (z >> 31)                       // uint64, wrapping arithmetic throughout
```

- `next_u64()` is the value above.
- `next_f64()` = `(next_u64() >> 11) * 2^-53`, a float in [0, 1).
- `next_below(n)` = `next_u64() % n`.

Where the contract says "random", use this generator with the run's seed, in the stated order. Then Python, Go, C++, and Rust produce the same random choices. With `--threads 1`, all four must produce the same index and the same results. With more threads, insertion order may differ, and small differences are acceptable.

## 6. The indexes

Score is always the dot product `q · x`. Each index returns the k highest-scoring IDs, best first. **Ties:** on equal scores, the lower ID comes first. If the index finds fewer than k candidates, pad with `-1` and score `-inf` (write `null` for the score in JSON).

### 6.1 flat

No parameters. Search computes `q · x` for every row and returns the top k. Use a partial selection (heap or partition), not a full sort.

### 6.2 k-means (shared by ivf, pq, ivf_pq, diskann)

One procedure, in `kmeans`:

```
kmeans(points, k, iters, seed) -> centers (k, d)
```

1. **Training set:** the first `train_size` rows of the corpus. Not random. Default `train_size` = min(N, 256 × k), but never below k.
2. **Init:** pick k distinct training rows by `next_below(train_n)` with the PRNG seeded by `seed`; on a repeat, draw again.
3. **Iterate at most `iters` times** (default 20). Each iteration, in this order:
   1. Assign each training point to the center with the highest dot product (lowest squared distance in `l2` mode, 6.4.1).
   2. If no label changed compared with the previous iteration, stop. Compare the labels from step 1, before the empty-cluster fix.
   3. Fix empty clusters (step 4 below).
   4. Set each center to the mean of its points, then **L2-normalize the center** (all data is normalized, and normalized centers keep dot-product ranking consistent). PQ codebooks skip the normalization (6.4).
   The centers after the last iteration's step 4 are the result.
4. **Empty cluster:** for each empty cluster in index order, take the worst-fit training point: the point with the lowest score to its own assigned center (highest squared distance in `l2` mode), among points whose current cluster has at least 2 members, so no cluster becomes empty by donating. Move that point to the empty cluster (update its label and the member counts) before handling the next empty cluster.

### 6.3 ivf

| Param | Phase | Default | Meaning |
|---|---|---|---|
| `nlist` | build | 1024 | number of clusters |
| `train_size` | build | see 6.2 | training rows |
| `iters` | build | 20 | k-means iterations |
| `nprobe` | search | 8 | clusters searched per query |

**Build:** train = k-means. Add = assign every corpus row to its best center, append its ID to that center's list. Store lists as contiguous ID arrays with offsets (CSR layout), not one dynamic list per cluster.

**Search:** score the query against all centers, take the `nprobe` best, scan all IDs in those lists with the full vectors, return the top k. `distance_computations` = nlist + number of scanned rows.

### 6.4 pq

| Param | Phase | Default | Meaning |
|---|---|---|---|
| `m` | build | 48 | sub-vectors per vector; 384 / m must be an integer |
| `nbits` | build | 8 | bits per code; only 8 is supported (256 centroids) |
| `metric` | build | `ip` | `ip` (dot product) or `l2` (squared Euclidean distance). See 6.4.1. |
| `train_size` | build | 100000 | training rows |
| `iters` | build | 20 | k-means iterations per codebook |
| `rerank` | search | 0 | candidates re-scored with full vectors; 0 = off |

**Build:** split each training vector into m sub-vectors of d/m dimensions. For sub-vector j, run k-means (6.2, with `next_below` from a PRNG seeded `seed + j`) with k = 256 on the training sub-vectors, **without the normalization step 3** (sub-vectors are not unit length), with assignment by the `metric` (6.4.1). Codebook j is (256, d/m). Encode every corpus row: code[j] = index of the best centroid for sub-vector j under the `metric`. Store codes as a contiguous (N, m) uint8 array.

**Search:** build a table T (m, 256) from the query and the codebooks (6.4.1). The score of row i is `sum_j T[j][code[i][j]]`. Scan all N rows, keep the top k (or the top `rerank` if rerank > 0, then re-score those with the full vectors and return the top k). `distance_computations` = N (table-based scores count as 1 each).

#### 6.4.1 The metric dimension

For exact scores on unit vectors, `q · x` and `−‖q − x‖²` give the same ranking, because `‖q − x‖² = 2 − 2 q · x`. For PQ they differ: the codebooks are trained and the codes are chosen on sub-vectors, which are not unit length, so `ip` and `l2` give different codebooks, different codes, and different recall. **Both are measured.** The runner runs every PQ-based index (pq, ivf_pq, diskann) with `metric=ip` and `metric=l2`, and the report shows both. The FAISS reference uses `METRIC_INNER_PRODUCT` for `ip` and `METRIC_L2` for `l2`.

| | `metric=ip` | `metric=l2` |
|---|---|---|
| k-means assignment (codebooks only) | highest `p · c` | lowest `‖p − c‖²` |
| code choice | highest `x_j · codebook[j][c]` | lowest `‖x_j − codebook[j][c]‖²` |
| table entry `T[j][c]` | `q_j · codebook[j][c]` | `−‖q_j − codebook[j][c]‖²` |
| reported `scores` | approximate dot product | negative approximate squared distance |
| rerank score | `q · x` | `−‖q − x‖²` |

In both modes, a higher score is better, so the top-k logic is the same. The IVF coarse centers (6.2) always use the shared k-means with dot-product assignment, in both modes, so only the PQ part changes.

### 6.5 ivf_pq

Params: all of ivf (`nlist`, `nprobe`, `iters`) and pq (`m`, `nbits`, `metric`, `rerank`), plus `train_size` (default 100000).

**Build:** train ivf centers (6.2, normalized). Compute the residual `r = x − c` for each training row, with c its assigned center, and train the codebooks on residuals (6.4, under the `metric`). Encode each corpus row's residual. Store per list: IDs and codes, CSR layout.

**Search:** choose `nprobe` lists as in ivf.

- `metric=ip`: the score of row i in a list with center c is `q · c + sum_j T[j][code[i][j]]`, with T built from q against the residual codebooks. T is the same for every list.
- `metric=l2`: for each chosen list, set `q' = q − c` and build `T_c[j][k] = −‖q'_j − codebook[j][k]‖²`. The score of row i is `sum_j T_c[j][code[i][j]]`. One table per probed list, so `nprobe` tables per query.

Return the top k, with optional rerank.

### 6.6 hnsw

| Param | Phase | Default | Meaning |
|---|---|---|---|
| `m` | build | 16 | max edges per node on layers ≥ 1; layer 0 allows 2m |
| `ef_construct` | build | 100 | candidate list size during insertion |
| `ef` | search | 64 | candidate list size during search; effective value is max(ef, k) |

**Levels:** for corpus row i, draw `u = next_f64()` from one PRNG seeded `seed`, drawn in row order i = 0, 1, 2, ..., before any insertion. `level(i) = floor(−ln(u) × mL)` with `mL = 1 / ln(m)`. The node exists on layers 0..level(i). The entry point is the node with the highest level, ties to the lowest row index.

**Insert row i** (rows in order, as in the paper, Malkov and Yashunin 2018, Algorithm 1):
1. From the entry point, on each layer above level(i), greedy descent with ef = 1.
2. On each layer from min(level(i), top) down to 0: search-layer (Algorithm 2) with `ef_construct` candidates, select neighbors with the **heuristic** (Algorithm 4, `extendCandidates = false`, `keepPrunedConnections = false`), limit `m` (or 2m on layer 0), and add bidirectional edges. If a neighbor now exceeds its limit, shrink its list with the same heuristic over its current neighbors.
3. If level(i) > top, the node becomes the entry point.

**Search:** greedy descent from the entry point to layer 1 with ef = 1, then search-layer on layer 0 with `max(ef, k)` candidates, return the top k. `distance_computations` = number of dot products in the whole search.

**Storage:** node IDs are the row IDs. Each layer's adjacency is a flat int32 array with fixed slots per node (2m on layer 0, m above), with a per-node count, so no per-node heap allocation during search. Build may use threads (insert in parallel with a lock per node's neighbor list). With `--threads 1`, insertion is strictly in row order.

### 6.7 diskann

| Param | Phase | Default | Meaning |
|---|---|---|---|
| `r` | build | 64 | max out-degree |
| `l_build` | build | 100 | candidate list size during build |
| `alpha` | build | 1.2 | pruning slack |
| `pq_m` | build | 48 | PQ sub-vectors for the in-RAM codes |
| `metric` | build | `ip` | metric for the PQ codes and tables (6.4.1); the graph build and rerank always use the dot product |
| `l` | search | 100 | candidate list size during search |
| `beam` | search | 4 | nodes expanded per step |
| `rerank` | search | 100 | candidates re-scored with full vectors from disk |
| `io` | search | `mmap` | `mmap` (memory-mapped file, OS cache allowed) or `nocache` (uncached reads, see 6.7.1) |

**Build (Vamana, Subramanya et al. 2019):**
1. Entry point = the corpus row with the highest dot product with the mean of all rows (the medoid).
2. Init: each node gets `r` random out-edges, drawn with `next_below(N)` from a PRNG seeded `seed`, in row order, skipping self and repeats.
3. Two passes over all rows in row order, first with alpha = 1.0, then with `alpha`. For row i: greedy search from the entry point with list size `l_build` to get the visited set V; robust-prune(i, V, alpha, r); for each new out-neighbor j of i, add the edge j→i and, if j now has more than r edges, robust-prune j.
4. Train PQ codes (6.4, `m = pq_m`) on all rows; encode all rows.
5. Write `<out>.diskann` next to the output JSON: for each node, its full vector then its out-edges (fixed `r` slots, int32, −1 for empty), so a node is one contiguous record. Report the file size in `extra.disk_bytes`.

**Search:** the corpus array is **released** after the file is written; search must not hold the full vectors in RAM. Hold PQ codes and codebooks in RAM. Beam search: a candidate list of size `l`; each step expands the `beam` best unexpanded candidates, reads their records from the file (6.7.1), scores their out-neighbors with the PQ table; stop when the list holds no unexpanded node. Re-score the top `rerank` candidates with their full vectors from the file, return the top k. Report, in that search run's `extra` object, `disk_reads` = mean records read per query and `disk_bytes_read` = mean bytes read per query.

#### 6.7.1 The I/O dimension: warm cache and real disk reads

macOS and Linux keep recently read file pages in RAM (the page cache). After one pass, a memory-mapped file is served from RAM, and the timings no longer include the SSD. **Both cases are measured**, through the `io` search parameter:

| `io` | How records are read | What the timing shows |
|---|---|---|
| `mmap` | Memory-map the whole file; read records through the map. | Warm cache: the algorithm and the RAM cost of the graph walk. |
| `nocache` | Open the file with caching disabled and read each record with `pread`. macOS: `fcntl(fd, F_NOCACHE, 1)`. Linux: `O_DIRECT`, with the read buffer and the offset aligned to 4096 bytes (pad each record to a multiple of 4096 in the file layout, so this holds on both systems). | Real SSD latency per record. |

Rules:

1. The runner runs DiskANN with `io=mmap` and `io=nocache` at every search setting. Before an `io=nocache` run, the `bench` program itself must not have touched the file through a map in the same process (the file is written, closed, and then opened with caching disabled), so the OS has no warm pages from this process.
2. The two modes must return **identical `ids`** for every query. A test asserts this.
3. A test asserts that `io=nocache` reports `extra.disk_reads > 0` in its search entry and a **higher p50 latency** than `io=mmap` at the same setting. If the two are equal, the reads are cached and the test fails.
4. The report shows both latencies side by side, and the ratio, so the SSD cost is visible.

Record layout on disk: each node record is `dim × 4` bytes of vector, then `r` int32 out-edges, padded to a multiple of 4096 bytes. With dim = 384 and r = 64, a record is 1,536 + 256 = 1,792 bytes, padded to 4,096, so the file is N × 4 KB (4.9 GB for the full corpus, 0.4 GB for dev). `extra.disk_bytes` reports the file size.

## 7. Language rules

From `CLAUDE.md`: no vector search libraries; write the `.npy` reader, k-means, and every index by hand. Allowed libraries:

| Language | Allowed beyond the standard library | Threads |
|---|---|---|
| Python 3.12 | NumPy (arrays, matrix products, argpartition). NumPy may do the inner loops; the algorithm structure must be explicit Python. | `multiprocessing` or `threading`; NumPy releases the GIL |
| Go 1.22+ | none | goroutines |
| C++17 | `nlohmann/json` (single header, vendored in `indexes/cpp/third_party/`) | `std::thread` |
| Rust 2021 | `rayon`, `serde`, `serde_json`, `memmap2`, `libc` (for `fcntl`, `pread`, `getrusage`) | `rayon` |

Compiler settings: C++ `-O3 -march=native`; Rust release profile with `opt-level=3`, `codegen-units=1`, `target-cpu=native` in `.cargo/config.toml`; Go default with `GOAMD64`/`GOARM64` defaults.

## 8. Directory layout and build commands

Same module names in every language.

```
indexes/python/           package; run with: uv run python -m indexes.python.bench
  bench.py  npy.py  distance.py  kmeans.py  splitmix.py
  flat.py  ivf.py  pq.py  ivf_pq.py  hnsw.py  diskann.py
  tests/test_*.py         pytest
indexes/go/               module vro/indexes/go; build: cd indexes/go && go build -o bin/bench ./cmd/bench
  cmd/bench/main.go
  npy/  distance/  kmeans/  splitmix/  flat/  ivf/  pq/  ivfpq/  hnsw/  diskann/   (one package each)
  *_test.go               go test ./...
indexes/cpp/              build: cmake -S indexes/cpp -B indexes/cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build indexes/cpp/build
  CMakeLists.txt  src/bench.cpp  src/{npy,distance,kmeans,splitmix,flat,ivf,pq,ivf_pq,hnsw,diskann}.{hpp,cpp}
  tests/                  ctest
indexes/rust/             build: cargo build --release --manifest-path indexes/rust/Cargo.toml
  Cargo.toml  src/main.rs  src/{npy,distance,kmeans,splitmix,flat,ivf,pq,ivf_pq,hnsw,diskann}.rs
  tests/                  cargo test --release
```

The runner calls: `uv run python -m indexes.python.bench`, `indexes/go/bin/bench`, `indexes/cpp/build/bench`, `indexes/rust/target/release/bench`.

A language may keep its shared types (matrix, params, build context with threads, seed, and the output path, result structs) in one extra module, for example `common`. Each index module exposes the same interface (adapted to the language):

```
build(vectors, params, threads, seed) -> index      // runs train + add, records train_s and add_s
search(index, query, k, params) -> (ids, scores)    // one query
index_bytes(index) -> int
```

`bench` dispatches on `--index` to the module. In the skeleton (Wave 1), `flat` is implemented, so the whole path from `.npy` to output JSON is tested end to end. The other five index modules exist and exit with code 1 and the message "not implemented". Wave 2 agents replace one module body each and must not edit `bench`, the shared modules, or the build files, except to add a dependency listed in section 7.

## 9. Tests

Each language has tests that run on `data/processed/dev/` with `--limit 20000` where speed matters. Required tests:

1. **npy:** reading `queries.npy` gives shape (1000, 384), and row 0's first three values match a hard-coded reference printed by `uv run python -c "import numpy as np; print(np.load('data/processed/dev/queries.npy')[0,:3].tolist())"`.
2. **splitmix:** seed 42 gives first `next_u64()` = `13679457532755275413`; seed 0 gives `16294208416658607535`.
3. **flat:** on the dev set, recall@10 against `ground_truth.npy` = 1.0 (tests may read ground truth).
4. **Each other index:** recall@10 ≥ a floor stated in the module's docstring at default params on the dev set (ivf nprobe=8 ≥ 0.75, pq ≥ 0.50, ivf_pq ≥ 0.45, hnsw ef=64 ≥ 0.95, diskann l=100 ≥ 0.90; FAISS on dev gives ivf nprobe=8 = 0.80 and hnsw ef=64 = 0.97, so these floors leave room for the different k-means and HNSW RNG). Also: the top result of the flat index equals the top result of ground truth for every query.
5. **Metric dimension (pq, ivf_pq, diskann):** both `metric=ip` and `metric=l2` meet the recall floor. Both produce a full run without error.
6. **I/O dimension (diskann):** `io=mmap` and `io=nocache` return identical `ids`; `io=nocache` reports `disk_reads > 0` and a higher p50 latency than `io=mmap` (6.7.1).
7. **Output JSON** validates against section 3: all keys present, shapes right.

Recall for reporting is computed only by `tools/bench/`. Tests use it only as an assertion.

## 10. Checklist for a Wave 2 agent

1. Read this file and `CLAUDE.md`.
2. Implement only your index module in your language. Do not touch other files unless section 8 allows it.
3. Run the tests for your module on the dev set. Report the numbers, not "it works".
4. Run one full `bench` invocation on the dev set with the default parameters and one sweep. Attach the output JSON path.
5. Report: files changed, test output, any deviation from this contract and why.
6. Do not commit. The main session commits.
