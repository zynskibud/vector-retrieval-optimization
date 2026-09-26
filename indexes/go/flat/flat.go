// Package flat is exact search (CONTRACT.md section 6.1): the query is scored
// against every corpus row and the k best are kept with a bounded heap.
// Recall@10 on the dev set must be 1.0.
package flat

import (
	"time"

	"vro/indexes/go/distance"
	"vro/indexes/go/npy"
)

// BuildDefaults lists every build parameter with its default. Flat has none.
var BuildDefaults = map[string]any{}

// SearchDefaults lists every search parameter with its default. filter names
// a mask filter_<name>.npy in DataDir (CONTRACT.md section 11); "none" = no filter.
var SearchDefaults = map[string]any{"filter": "none"}

// DataDir is the data directory that holds filter_<name>.npy. bench sets it.
var DataDir string

// Index holds a reference to the corpus. It copies nothing.
type Index struct {
	vectors []float32
	n, dim  int
	trainS  float64
	addS    float64
	dists   int64
	passed  int64 // rows that passed the filter, summed over searches
	topk    *distance.TopK
}

// Build wraps the corpus. vectors is row-major (n, dim).
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	start := time.Now()
	ix := &Index{vectors: vectors, n: n, dim: dim}
	ix.addS = time.Since(start).Seconds()
	return ix, nil
}

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	if ix.topk == nil || ix.topk.K() != k {
		ix.topk = distance.NewTopK(k)
	}
	tk := ix.topk
	tk.Reset()
	d := ix.dim
	mask := filterMask(params)
	if mask == nil {
		for i := 0; i < ix.n; i++ {
			tk.Push(int64(i), distance.Dot(query, ix.vectors[i*d:(i+1)*d]))
		}
		ix.dists += int64(ix.n)
		ix.passed += int64(ix.n)
		return tk.Results()
	}
	// Filtered: iterate all rows and skip the rows that fail (section 11.3).
	scored := 0
	for i := 0; i < ix.n; i++ {
		if !mask[i] {
			continue
		}
		tk.Push(int64(i), distance.Dot(query, ix.vectors[i*d:(i+1)*d]))
		scored++
	}
	ix.dists += int64(scored)
	ix.passed += int64(scored)
	return tk.Results()
}

// IndexBytes is 0: the corpus array is the index (section 4).
func IndexBytes(ix *Index) int64 { return 0 }

// TrainSeconds returns the train time of the build.
func (ix *Index) TrainSeconds() float64 { return ix.trainS }

// AddSeconds returns the add time of the build.
func (ix *Index) AddSeconds() float64 { return ix.addS }

// DistanceComputations returns the total dot products computed by Search so far.
func (ix *Index) DistanceComputations() int64 { return ix.dists }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any { return map[string]any{} }

// SearchCounters returns cumulative per-search counters. filter_rows is the
// number of rows that passed the filter (all rows for filter=none).
// bench reports the mean per query in each search run's "extra".
func (ix *Index) SearchCounters() map[string]float64 {
	return map[string]float64{"filter_rows": float64(ix.passed)}
}

// filterMask returns the mask for params["filter"], or nil for no filter.
// bench loads every mask before the build, so an error here is a program bug.
func filterMask(params map[string]any) []bool {
	name, _ := params["filter"].(string)
	m, err := npy.FilterMask(DataDir, name)
	if err != nil {
		panic(err)
	}
	return m
}
