// Package pq is the PQ index (CONTRACT.md section 6.4).
//
// Build splits each vector into m sub-vectors of dim/m dimensions, trains one
// codebook of 256 centroids per sub-vector with the shared k-means (no center
// normalization, assignment by the metric), and encodes every corpus row as m
// uint8 codes. Search builds a lookup table T (m, 256) per query and scores
// every row as sum_j T[j][code[j]]. With rerank > 0, the top rerank candidates
// are re-scored with the full vectors.
//
// Recall floor (CONTRACT.md section 9): recall@10 >= 0.50, for metric=ip and metric=l2 on the dev set at default params.
package pq

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"vro/indexes/go/distance"
	"vro/indexes/go/kmeans"
)

// ksub is the number of centroids per codebook (nbits = 8).
const ksub = 256

// Defaults are int64 so that bench parses command-line values as integers.

// BuildDefaults lists every build parameter with its default.
var BuildDefaults = map[string]any{"m": int64(48), "nbits": int64(8), "metric": "ip", "train_size": int64(100000), "iters": int64(20)}

// SearchDefaults lists every search parameter with its default.
var SearchDefaults = map[string]any{"rerank": int64(0)}

// Index is the PQ index.
type Index struct {
	vectors   []float32 // full corpus, referenced for rerank (not copied)
	n, dim    int
	m, dsub   int
	l2        bool
	codebooks []float32 // (m, 256, dsub) row-major
	codes     []uint8   // (n, m) row-major
	table     []float32 // (m, 256) scratch table, reused per query
	topk      *distance.TopK
	final     *distance.TopK

	trainS, addS float64
	dists        int64
}

// intParam reads an integer parameter from params, or from defaults if absent.
func intParam(params, defaults map[string]any, key string) (int, error) {
	v, ok := params[key]
	if !ok {
		v = defaults[key]
	}
	switch x := v.(type) {
	case int:
		return x, nil
	case int64:
		return int(x), nil
	case string:
		i, err := strconv.Atoi(x)
		if err != nil {
			return 0, fmt.Errorf("pq: parameter %s needs an integer, got %q", key, x)
		}
		return i, nil
	case float64:
		if x != float64(int(x)) {
			return 0, fmt.Errorf("pq: parameter %s needs an integer, got %v", key, x)
		}
		return int(x), nil
	}
	return 0, fmt.Errorf("pq: parameter %s has bad type %T", key, v)
}

// Build runs train and add. vectors is row-major (n, dim).
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	m, err := intParam(params, BuildDefaults, "m")
	if err != nil {
		return nil, err
	}
	nbits, err := intParam(params, BuildDefaults, "nbits")
	if err != nil {
		return nil, err
	}
	trainSize, err := intParam(params, BuildDefaults, "train_size")
	if err != nil {
		return nil, err
	}
	iters, err := intParam(params, BuildDefaults, "iters")
	if err != nil {
		return nil, err
	}
	metric, ok := params["metric"].(string)
	if !ok {
		metric = BuildDefaults["metric"].(string)
	}
	if nbits != 8 {
		return nil, fmt.Errorf("pq: nbits=%d is not supported, only 8", nbits)
	}
	if metric != "ip" && metric != "l2" {
		return nil, fmt.Errorf("pq: metric must be ip or l2, got %q", metric)
	}
	if m <= 0 || dim%m != 0 {
		return nil, fmt.Errorf("pq: dim %d is not divisible by m=%d", dim, m)
	}
	if threads < 1 {
		threads = 1
	}
	trainN := min(trainSize, n)
	if trainN < ksub {
		return nil, fmt.Errorf("pq: need at least %d training rows, have %d", ksub, trainN)
	}
	dsub := dim / m
	ix := &Index{vectors: vectors, n: n, dim: dim, m: m, dsub: dsub, l2: metric == "l2"}

	// Train: one codebook per sub-vector position.
	start := time.Now()
	ix.codebooks = make([]float32, m*ksub*dsub)
	sub := make([]float32, trainN*dsub)
	for j := 0; j < m; j++ {
		for i := 0; i < trainN; i++ {
			copy(sub[i*dsub:(i+1)*dsub], vectors[i*dim+j*dsub:i*dim+(j+1)*dsub])
		}
		cfg := kmeans.Config{K: ksub, Iters: iters, Seed: seed + uint64(j), Threads: threads, L2: ix.l2, NoNormalize: true}
		cb, err := kmeans.Train(sub, trainN, dsub, cfg)
		if err != nil {
			return nil, fmt.Errorf("pq: codebook %d: %w", j, err)
		}
		copy(ix.codebooks[j*ksub*dsub:(j+1)*ksub*dsub], cb)
	}
	ix.trainS = time.Since(start).Seconds()

	// Add: encode every row, rows split across goroutines.
	start = time.Now()
	ix.codes = make([]uint8, n*m)
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
				row := vectors[i*dim : (i+1)*dim]
				for j := 0; j < m; j++ {
					c, _ := kmeans.Nearest(row[j*dsub:(j+1)*dsub], ix.codebooks[j*ksub*dsub:(j+1)*ksub*dsub], ksub, dsub, ix.l2)
					ix.codes[i*m+j] = uint8(c)
				}
			}
		}(lo, hi)
	}
	wg.Wait()
	ix.addS = time.Since(start).Seconds()
	ix.table = make([]float32, m*ksub)
	return ix, nil
}

// Codes returns the (n, m) code array. Tests use it.
func (ix *Index) Codes() []uint8 { return ix.codes }

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	rerank, err := intParam(params, SearchDefaults, "rerank")
	if err != nil || rerank < 0 {
		rerank = 0
	}
	m, dsub := ix.m, ix.dsub
	// Table T[j][c]: dot product (ip) or negative squared distance (l2).
	T := ix.table
	for j := 0; j < m; j++ {
		qj := query[j*dsub : (j+1)*dsub]
		cb := ix.codebooks[j*ksub*dsub : (j+1)*ksub*dsub]
		for c := 0; c < ksub; c++ {
			T[j*ksub+c] = kmeans.Score(qj, cb[c*dsub:(c+1)*dsub], ix.l2)
		}
	}
	keep := k
	if rerank > 0 {
		keep = rerank
	}
	if ix.topk == nil || ix.topk.K() != keep {
		ix.topk = distance.NewTopK(keep)
	}
	tk := ix.topk
	tk.Reset()
	codes := ix.codes
	for i := 0; i < ix.n; i++ {
		code := codes[i*m : (i+1)*m]
		var s float32
		for j, c := range code {
			s += T[j*ksub+int(c)]
		}
		tk.Push(int64(i), s)
	}
	ix.dists += int64(ix.n)
	if rerank == 0 {
		return tk.Results()
	}
	cand, _ := tk.Results()
	if ix.final == nil || ix.final.K() != k {
		ix.final = distance.NewTopK(k)
	}
	fk := ix.final
	fk.Reset()
	d := ix.dim
	for _, id := range cand {
		if id < 0 {
			continue
		}
		fk.Push(id, kmeans.Score(query, ix.vectors[int(id)*d:(int(id)+1)*d], ix.l2))
		ix.dists++
	}
	return fk.Results()
}

// IndexBytes returns the computed memory of the index structure (section 4):
// codebooks (m * 256 * dsub * 4) + codes (n * m).
func IndexBytes(ix *Index) int64 {
	return int64(ix.m*ksub*ix.dsub*4) + int64(ix.n*ix.m)
}

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
