// Package ivf is the IVF index (CONTRACT.md section 6.3).
//
// Build trains nlist centers with the shared k-means (section 6.2), then
// assigns every corpus row to its best center. The inverted lists are stored
// in CSR layout: one contiguous []int32 of row IDs grouped by list, plus an
// offsets array of length nlist+1. List c holds ids[offsets[c]:offsets[c+1]].
//
// Search scores the query against all centers, keeps the nprobe best, and
// scans every row of those lists with the full vectors.
//
// Recall floor (CONTRACT.md section 9): recall@10 >= 0.75 with nprobe=8 on the dev set at default params.
package ivf

import (
	"fmt"
	"math"
	"sync"
	"time"

	"vro/indexes/go/distance"
	"vro/indexes/go/kmeans"
)

// BuildDefaults lists every build parameter with its default.
// train_size -1 means "default of section 6.2": min(n, 256*nlist), not below nlist.
var BuildDefaults = map[string]any{"nlist": int64(1024), "train_size": int64(-1), "iters": int64(20)}

// SearchDefaults lists every search parameter with its default. Values are
// int64 because bench parses command-line values to the type of the default.
var SearchDefaults = map[string]any{"nprobe": int64(8)}

// Index is the IVF index.
type Index struct {
	vectors []float32 // corpus, row-major (n, dim); not copied
	n, dim  int
	nlist   int
	centers []float32 // row-major (nlist, dim), L2-normalized
	ids     []int32   // row IDs grouped by list (CSR values)
	offsets []int32   // length nlist+1; list c is ids[offsets[c]:offsets[c+1]]

	trainS, addS float64
	dists        int64

	topk   *distance.TopK // result selector, reused across queries
	probeK *distance.TopK // center selector, reused across queries
}

// intParam reads an integer parameter. bench passes int64; tests may pass int.
func intParam(p map[string]any, key string, def int) (int, error) {
	v, ok := p[key]
	if !ok || v == nil {
		return def, nil
	}
	switch x := v.(type) {
	case int:
		return x, nil
	case int64:
		return int(x), nil
	case int32:
		return int(x), nil
	case float64:
		return int(x), nil
	}
	return 0, fmt.Errorf("ivf: parameter %s has type %T, want an integer", key, v)
}

// Build runs train and add. vectors is row-major (n, dim).
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	nlist, err := intParam(params, "nlist", 1024)
	if err != nil {
		return nil, err
	}
	iters, err := intParam(params, "iters", 20)
	if err != nil {
		return nil, err
	}
	trainSize, err := intParam(params, "train_size", -1)
	if err != nil {
		return nil, err
	}
	if nlist < 1 {
		return nil, fmt.Errorf("ivf: nlist must be >= 1, got %d", nlist)
	}
	if trainSize < 0 {
		trainSize = kmeans.TrainSize(n, nlist)
	}
	if trainSize > n {
		trainSize = n
	}
	if n > math.MaxInt32 {
		return nil, fmt.Errorf("ivf: n=%d does not fit int32 IDs", n)
	}
	if threads < 1 {
		threads = 1
	}
	ix := &Index{vectors: vectors, n: n, dim: dim, nlist: nlist}

	// Train: k-means on the first trainSize rows (section 6.2).
	start := time.Now()
	ix.centers, err = kmeans.Train(vectors[:trainSize*dim], trainSize, dim, kmeans.Config{
		K: nlist, Iters: iters, Seed: seed, Threads: threads,
	})
	if err != nil {
		return nil, err
	}
	ix.trainS = time.Since(start).Seconds()

	// Add: assign every row, then build the CSR lists.
	start = time.Now()
	labels := make([]int32, n)
	chunk := (n + threads - 1) / threads
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		lo, hi := t*chunk, min((t+1)*chunk, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				c, _ := kmeans.Nearest(vectors[i*dim:(i+1)*dim], ix.centers, nlist, dim, false)
				labels[i] = int32(c)
			}
		}(lo, hi)
	}
	wg.Wait()
	// Counting sort by label: rows stay in ascending ID order inside each list,
	// so the layout does not depend on the thread count.
	ix.offsets = make([]int32, nlist+1)
	for _, c := range labels {
		ix.offsets[c+1]++
	}
	for c := 0; c < nlist; c++ {
		ix.offsets[c+1] += ix.offsets[c]
	}
	ix.ids = make([]int32, n)
	next := make([]int32, nlist)
	copy(next, ix.offsets[:nlist])
	for i, c := range labels {
		ix.ids[next[c]] = int32(i)
		next[c]++
	}
	ix.addS = time.Since(start).Seconds()
	return ix, nil
}

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	nprobe, err := intParam(params, "nprobe", 8)
	if err != nil || nprobe < 1 {
		nprobe = 8
	}
	if nprobe > ix.nlist {
		nprobe = ix.nlist
	}
	if ix.probeK == nil || ix.probeK.K() != nprobe {
		ix.probeK = distance.NewTopK(nprobe)
	}
	if ix.topk == nil || ix.topk.K() != k {
		ix.topk = distance.NewTopK(k)
	}
	d := ix.dim

	// Pick the nprobe best centers; on equal scores the lower list index wins.
	pk := ix.probeK
	pk.Reset()
	for c := 0; c < ix.nlist; c++ {
		pk.Push(int64(c), distance.Dot(query, ix.centers[c*d:(c+1)*d]))
	}
	lists, _ := pk.Results()

	tk := ix.topk
	tk.Reset()
	scanned := 0
	for _, c := range lists {
		if c < 0 {
			continue
		}
		for _, id := range ix.ids[ix.offsets[c]:ix.offsets[c+1]] {
			row := int(id)
			tk.Push(int64(id), distance.Dot(query, ix.vectors[row*d:(row+1)*d]))
		}
		scanned += int(ix.offsets[c+1] - ix.offsets[c])
	}
	ix.dists += int64(ix.nlist + scanned)
	return tk.Results()
}

// IndexBytes returns the computed memory of the index structure (section 4):
// centers (nlist*dim*4) + one int32 ID per row + the int32 offsets array.
func IndexBytes(ix *Index) int64 {
	return int64(ix.nlist)*int64(ix.dim)*4 + int64(len(ix.ids))*4 + int64(len(ix.offsets))*4
}

// TrainSeconds returns the train time of the build.
func (ix *Index) TrainSeconds() float64 { return ix.trainS }

// AddSeconds returns the add time of the build.
func (ix *Index) AddSeconds() float64 { return ix.addS }

// DistanceComputations returns the total dot products computed by Search so
// far: per query, nlist + scanned rows.
func (ix *Index) DistanceComputations() int64 { return ix.dists }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any {
	minL, maxL := int32(math.MaxInt32), int32(0)
	empty := 0
	for c := 0; c < ix.nlist; c++ {
		l := ix.offsets[c+1] - ix.offsets[c]
		minL, maxL = min(minL, l), max(maxL, l)
		if l == 0 {
			empty++
		}
	}
	return map[string]any{
		"id_type":       "int32",
		"list_size_min": minL,
		"list_size_max": maxL,
		"empty_lists":   empty,
	}
}

// SearchCounters returns cumulative per-search counters. IVF has none.
func (ix *Index) SearchCounters() map[string]float64 { return map[string]float64{} }
