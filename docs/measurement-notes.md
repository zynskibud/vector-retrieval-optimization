# Measurement notes

Open problems in how the benchmarks are measured, with the data that showed them and the planned fix. Update this file when a problem is closed.

## 1. Run-to-run variance on macOS (fix in place, effect not yet measured)

**Observed.** Rust flat on the dev set (100k rows, one thread): the Wave 1 agent measured p50 = 2.48 ms. Three runs by the main session on an idle machine gave 4.80, 4.41, and 4.59 ms. The runner's run gave 5.09 ms. Same binary, same data, same code path. That is a 2× spread.

**Cause, as far as known.** macOS schedules a process on performance or efficiency cores and changes the clock as it likes. A flat scan is memory-bound (154 MB per query), so bandwidth from the chosen core sets the latency. The process cannot pin itself to a core on macOS.

**Effect.** A single run cannot separate a 20% code difference from scheduler noise.

**Fix (runner.py).**
1. `--repeat N` (default 3). Each case runs N times as separate processes. The runner keeps the run with the median p50 and stores every run's p50 in `extra.runner.p50_ms_runs`, so the spread is visible.
2. `--max-load` (default 2.0). The runner refuses to run a case while the 1-minute load average is above it, so runs never overlap with builds or agents. The load at start is stored in `extra.runner.load1_at_start`.

The report prints the spread of the repeat runs (`p50_spread`, min-max of each run's mean p50) next to the p50. Still to do: the full-corpus runs should confirm that 3 repeats are enough.

## 2. Go's dot product is not vectorized (open)

**Observed.** Flat on dev: Go p50 = 12.6 ms, against 4.9 to 5.1 ms for Python (NumPy), C++, Rust, and FAISS. Recall is identical, so this is speed only.

**Cause.** The Go compiler does not auto-vectorize floating-point loops. The Go skeleton's `distance.Dot` keeps four scalar accumulators, which helped (28 ms to 10 ms) but still runs one multiply-add per instruction. C++ and Rust compile the same loop to 4-lane NEON. NumPy calls Apple Accelerate.

**Options.**
1. Leave it. "What the compiler gives you" is a fair data point for the language comparison, and it is what most Go code does.
2. Hand-written NEON assembly for `Dot` in a `.s` file, which is how Go's own math libraries do it. Standard library only, so allowed by the contract.
3. Both: keep the plain loop as the default, add the assembly behind a build tag, and report both in the write-up.

**Decision.** Option 3, in Wave 2 or later. Until then, Go's latency numbers show the plain loop.

## 3. NumPy's BLAS on macOS (closed, documented in CONTRACT section 4)

**Observed.** Python flat p50 was 3.2 ms with default settings, faster than C++ and Rust.

**Cause.** NumPy on macOS arm64 uses Apple Accelerate, which runs matrix products on the AMX unit and on several threads.

**Fix.** `indexes/python/bench.py` sets the BLAS thread variables to 1 before NumPy loads. Python flat is now 4.9 ms. The report must say that Python's flat speed comes from Accelerate, not from Python.
