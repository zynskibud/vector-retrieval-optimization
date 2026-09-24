// Package flat is exact search (CONTRACT.md section 6.1): the query is scored
// against every corpus row and the k best are kept with a bounded heap.
// Recall@10 on the dev set must be 1.0.
package flat

import (
	"time"

	"vro/indexes/go/distance"
)

// BuildDefaults lists every build parameter with its default. Flat has none.
var BuildDefaults = map[string]any{}

// SearchDefaults lists every search parameter with its default. Flat has none.
var SearchDefaults = map[string]any{}

// Index holds a reference to the corpus. It copies nothing.
type Index struct {
	vectors []float32
	n, dim  int
	trainS  float64
	addS    float64
	dists   int64
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
	for i := 0; i < ix.n; i++ {
		tk.Push(int64(i), distance.Dot(query, ix.vectors[i*d:(i+1)*d]))
	}
	ix.dists += int64(ix.n)
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

// SearchCounters returns cumulative per-search counters, for example
// disk_reads. bench reports the mean per query in each search run's "extra".
func (ix *Index) SearchCounters() map[string]float64 { return map[string]float64{} }
