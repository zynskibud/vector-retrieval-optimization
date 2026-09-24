package main

import (
	"vro/indexes/go/diskann"
	"vro/indexes/go/flat"
	"vro/indexes/go/hnsw"
	"vro/indexes/go/ivf"
	"vro/indexes/go/ivfpq"
	"vro/indexes/go/kmeans"
	"vro/indexes/go/pq"
)

// built is the common view of a built index. Every index type implements it.
type built interface {
	TrainSeconds() float64
	AddSeconds() float64
	DistanceComputations() int64 // cumulative; -1 if not counted
	Extra() map[string]any
	SearchCounters() map[string]float64 // cumulative; mean per query goes to searches[i].extra
}

// instance couples a built index with its package-level Search and IndexBytes.
type instance struct {
	idx    built
	search func(q []float32, k int, p map[string]any) ([]int64, []float32)
	bytes  int64
}

// buildFunc runs the package Build and wraps the result.
type buildFunc func(v []float32, n, dim int, p map[string]any, threads int, seed uint64) (instance, error)

type indexSpec struct {
	buildDefaults, searchDefaults map[string]any
	build                         buildFunc
}

// wrap adapts one index package to buildFunc. The package functions follow the
// contract interface: Build, Search, IndexBytes.
func wrap[I built](
	b func([]float32, int, int, map[string]any, int, uint64) (I, error),
	s func(I, []float32, int, map[string]any) ([]int64, []float32),
	ib func(I) int64,
) buildFunc {
	return func(v []float32, n, dim int, p map[string]any, threads int, seed uint64) (instance, error) {
		ix, err := b(v, n, dim, p, threads, seed)
		if err != nil {
			return instance{}, err
		}
		return instance{
			idx:    ix,
			search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return s(ix, q, k, sp) },
			bytes:  ib(ix),
		}, nil
	}
}

var registry = map[string]indexSpec{
	"flat":    {flat.BuildDefaults, flat.SearchDefaults, wrap(flat.Build, flat.Search, flat.IndexBytes)},
	"ivf":     {ivf.BuildDefaults, ivf.SearchDefaults, wrap(ivf.Build, ivf.Search, ivf.IndexBytes)},
	"pq":      {pq.BuildDefaults, pq.SearchDefaults, wrap(pq.Build, pq.Search, pq.IndexBytes)},
	"ivf_pq":  {ivfpq.BuildDefaults, ivfpq.SearchDefaults, wrap(ivfpq.Build, ivfpq.Search, ivfpq.IndexBytes)},
	"hnsw":    {hnsw.BuildDefaults, hnsw.SearchDefaults, wrap(hnsw.Build, hnsw.Search, hnsw.IndexBytes)},
	"diskann": {diskann.BuildDefaults, diskann.SearchDefaults, wrap(diskann.Build, diskann.Search, diskann.IndexBytes)},
}

// fillDerived resolves defaults that depend on the data: ivf's train_size
// (section 6.2) is min(n, 256*nlist), not below nlist. It also sets the
// diskann output path, which Build needs for its record file.
func fillDerived(index string, p map[string]any, n int) {
	if ts, ok := p["train_size"].(int64); ok && ts < 0 {
		p["train_size"] = int64(kmeans.TrainSize(n, int(p["nlist"].(int64))))
	}
}
