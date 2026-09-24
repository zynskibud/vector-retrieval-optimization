// Package ivf is the IVF index (CONTRACT.md section 6.3).
//
// Recall floor (CONTRACT.md section 9): recall@10 >= 0.80 with nprobe=8 on the dev set at default params.
//
// Wave 1 stub: Build returns "not implemented".
package ivf

import "errors"

// ErrNotImplemented is returned by Build until the index is written.
var ErrNotImplemented = errors.New("not implemented")

// BuildDefaults lists every build parameter with its default.
// train_size -1 means "default of section 6.2": min(n, 256*nlist), not below nlist.
var BuildDefaults = map[string]any{"nlist": 1024, "train_size": -1, "iters": 20}

// SearchDefaults lists every search parameter with its default.
var SearchDefaults = map[string]any{"nprobe": 8}

// Index is the IVF index.
type Index struct {
	trainS, addS float64
	dists        int64
}

// Build runs train and add. vectors is row-major (n, dim).
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	return nil, ErrNotImplemented
}

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	return nil, nil
}

// IndexBytes returns the computed memory of the index structure (section 4).
func IndexBytes(ix *Index) int64 { return 0 }

// TrainSeconds returns the train time of the build.
func (ix *Index) TrainSeconds() float64 { return ix.trainS }

// AddSeconds returns the add time of the build.
func (ix *Index) AddSeconds() float64 { return ix.addS }

// DistanceComputations returns the total dot products computed by Search so
// far, or -1 if the index does not count them.
func (ix *Index) DistanceComputations() int64 { return ix.dists }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any { return map[string]any{} }

// SearchCounters returns cumulative per-search counters, for example
// disk_reads. bench reports the mean per query in each search run's "extra".
func (ix *Index) SearchCounters() map[string]float64 { return map[string]float64{} }
