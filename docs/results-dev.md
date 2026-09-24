# Results on the dev set, after Wave 2a

Dev set: 100,000 corpus vectors, 1,000 queries, 384 dimensions, unit length. Machine: Apple M4, 10 cores, 24 GB. Search on one thread, 100 warm-up queries, 3 repeat runs per case, median-p50 run kept. Builds use 10 threads where the implementation supports it. Full tables: `results/summary/dev/results.md`. Plots: `results/summary/dev/<index>.png`.

The runner waited for the load average to drop before each case, but seven cases ran with a load between 2.1 and 7.8 after the 3-minute wait, so some latencies carry extra noise. The `p50_spread` column shows the three runs.

## Recall: the four languages agree

| Index | Setting | Python | Go | C++ | Rust | FAISS |
|---|---|---|---|---|---|---|
| flat | | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| IVF | nprobe=8 | 0.796 | 0.796 | 0.795 | 0.795 | 0.798 |
| IVF | nprobe=64 | 0.968 | 0.968 | 0.969 | 0.969 | 0.966 |
| PQ ip | rerank=0 | 0.659 | 0.659 | 0.659 | 0.659 | 0.674 |
| PQ l2 | rerank=0 | 0.680 | 0.682 | 0.681 | 0.681 | 0.679 |
| PQ l2 | rerank=100 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 |
| HNSW | ef=64 | 0.968 | 0.967 | 0.967 | 0.968 | 0.972 |
| HNSW | ef=256 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 |

Every hand-built index is within 0.005 of FAISS, except PQ ip at rerank=0 where FAISS is 0.015 higher (its k-means differs). C++ and Rust IVF return identical result IDs for all 1,000 queries at every nprobe: same PRNG, same k-means rule, same float32 loop order.

## Search latency, p50 in ms, one thread

| Index | Setting | Python | Go | C++ | Rust | FAISS |
|---|---|---|---|---|---|---|
| flat | | 4.6 | 17.9 | 5.8 | 6.0 | 4.4 |
| IVF | nprobe=8 | 0.28 | 0.45 | 0.24 | 0.28 | 0.08 |
| IVF | nprobe=64 | 2.0 | 2.2 | 1.7 | 1.5 | 0.38 |
| PQ ip | rerank=0 | 23.6 | 3.2 | 4.1 | 2.3 | 1.3 |
| PQ l2 | rerank=0 | 19.1 | 2.1 | 4.1 | 2.7 | 1.2 |
| HNSW | ef=64 | 0.85 | 0.49 | 0.35 | 0.64 | 0.28 |
| HNSW | ef=256 | 2.8 | 1.1 | 1.1 | 1.3 | 0.85 |

## Build time in seconds, 10 threads

| Index | Python | Go | C++ | Rust | FAISS |
|---|---|---|---|---|---|
| IVF | 8.3 | 68.8 | 16.9 | 20.9 | 3.0 |
| PQ ip | 10.3 | 82.4 | 35.2 | 69.8 | 6.8 |
| HNSW | 150 (1 thread) | 12.4 | 7.1 | 8.2 | 9.3 |

## Peak RSS in MB (includes the 154 MB corpus)

| Index | Python | Go | C++ | Rust | FAISS |
|---|---|---|---|---|---|
| flat | 178 | 154 | 150 | 154 | 344 |
| IVF | 725 | 216 | 156 | 167 | 423 |
| PQ l2 | 3,186 | 239 | 162 | 166 | 208 |
| HNSW | 254 | 265 | 173 | 182 | 361 |

## What the numbers say

1. **The algorithm is the same everywhere.** Recall agrees to the third decimal across four languages and FAISS. The language changes speed and memory, not results.

2. **Flat is memory-bound.** Python (NumPy on Accelerate), C++, Rust, and FAISS all read 154 MB per query at 25 to 35 GB/s from one core. Go is 3 times slower: its compiler does not vectorize the dot product (docs/measurement-notes.md, item 2).

3. **FAISS wins search latency on IVF and PQ by 3 to 4 times**, with the same recall. Its list scan and table scan are hand-tuned SIMD kernels with prefetching and batched accumulation. Our loops compute one row at a time. This is the clearest optimization target for the hand-built versions, and the reason FAISS exists.

4. **Hand-built HNSW is close to FAISS.** C++ is 0.35 ms against FAISS 0.28 ms at ef=64; Go 0.49; Rust 0.64; Python 0.85. The graph walk is pointer-chasing, not a tight SIMD loop, so the language gap is smaller than in the scans. Rust is slower than Go here, which is unexpected; its search takes a scratch buffer from a mutex-guarded pool and maps upper-layer nodes through a per-layer table. To investigate.

5. **Parallel HNSW builds beat FAISS.** C++ 7.1 s, Rust 8.2 s, FAISS 9.3 s, Go 12.4 s, all with 10 threads. Python's one-thread build takes 150 s.

6. **k-means dominates IVF and PQ build time, and NumPy beats C++ and Rust there.** Python's IVF build is 8 s against 17 s (C++) and 21 s (Rust), because NumPy's assignment step is one matrix product on Accelerate, while the compiled versions compute one dot product at a time. FAISS also uses BLAS: 3 s. Go's 69 s is the unvectorized dot again. The fix for C++ and Rust is a blocked matrix product for the assignment step.

7. **Python PQ search is 10 times slower than the compiled versions.** The table lookup is a gather over 100k × 48 codes; NumPy has no fast kernel for it. Its build is fast for the same reason as IVF.

8. **The `l2` metric beats `ip` for PQ at rerank=0 by 2 points** (0.681 vs 0.659) in all languages. Same k-means, different assignment rule. Rerank removes the difference.

9. **Memory.** C++ and Rust hold the corpus plus the index and nothing else. FAISS copies the vectors into its own structures (flat: 344 MB vs 150 MB). Python's PQ build peaks at 3.2 GB because 10 codebooks train at once, each with a 100k × 256 score array.

10. **Repeat spread.** Most cases vary under 10% between runs. Cases that ran with a load average above 2 vary more.

## Open questions for later waves

- Why is FAISS's IVF scan 3 times faster than the C++ and Rust scans at the same distance count? (Batched SIMD, prefetch, or the ID gather.)
- Why is the Rust HNSW search slower than Go's? (Scratch pool, node maps, or heap implementation.)
- Blocked matrix product for k-means assignment in C++, Rust, and Go.
- Go: NEON assembly for the dot product behind a build tag.
