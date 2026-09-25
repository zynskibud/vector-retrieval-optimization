// Package ivfpq is the IVF-PQ index (CONTRACT.md section 6.5).
//
// Build trains nlist coarse centers with the shared k-means (dot-product
// assignment, normalized centers) on the first train_size rows. Each training
// row gets its residual r = x - c, with c its assigned center. One codebook of
// 256 centroids per sub-vector is trained on the residuals (seed + j, no
// normalization, assignment by the metric). Add assigns every corpus row,
// encodes its residual as m uint8 codes, and stores the lists in CSR layout:
// ids ([]int32, grouped by list), codes (n*m bytes, same order as ids), and
// offsets (nlist+1).
//
// Search probes the nprobe best centers. metric=ip: score = q.c + sum_j
// T[j][code_j], one table T per query. metric=l2: per probed list q' = q - c,
// T_c[j][k] = -||q'_j - codebook[j][k]||^2, score = sum_j T_c[j][code_j].
// With rerank > 0, the top rerank candidates are re-scored with the full
// vectors (q.x for ip, -||q-x||^2 for l2).
//
// Recall floor (CONTRACT.md section 9): recall@10 >= 0.45, for metric=ip and metric=l2 on the dev set at default params.
package ivfpq

import (
	"fmt"
	"math"
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
var BuildDefaults = map[string]any{"nlist": int64(1024), "iters": int64(20), "m": int64(48), "nbits": int64(8), "metric": "ip", "train_size": int64(100000)}

// SearchDefaults lists every search parameter with its default.
var SearchDefaults = map[string]any{"nprobe": int64(8), "rerank": int64(0)}

// Index is the IVF-PQ index.
type Index struct {
	vectors   []float32 // full corpus, referenced for rerank (not copied)
	n, dim    int
	nlist     int
	m, dsub   int
	l2        bool
	centers   []float32 // (nlist, dim), L2-normalized
	codebooks []float32 // (m, 256, dsub), trained on residuals
	ids       []int32   // row IDs grouped by list
	codes     []uint8   // (n, m), row p holds the codes of ids[p]
	offsets   []int32   // nlist+1; list c is positions offsets[c]..offsets[c+1]

	table  []float32 // (m, 256) scratch
	qres   []float32 // (dim) scratch for q - c
	probeK *distance.TopK
	topk   *distance.TopK
	final  *distance.TopK

	trainS, addS float64
	dists        int64
}

// intParam reads an integer parameter from params, or from defaults if absent.
func intParam(params, defaults map[string]any, key string) (int, error) {
	v, ok := params[key]
	if !ok || v == nil {
		v = defaults[key]
	}
	switch x := v.(type) {
	case int:
		return x, nil
	case int64:
		return int(x), nil
	case int32:
		return int(x), nil
	case string:
		i, err := strconv.Atoi(x)
		if err != nil {
			return 0, fmt.Errorf("ivfpq: parameter %s needs an integer, got %q", key, x)
		}
		return i, nil
	case float64:
		if x != math.Trunc(x) {
			return 0, fmt.Errorf("ivfpq: parameter %s needs an integer, got %v", key, x)
		}
		return int(x), nil
	}
	return 0, fmt.Errorf("ivfpq: parameter %s has bad type %T", key, v)
}

// parallel runs f(lo, hi) over [0, n) split into threads contiguous chunks.
func parallel(n, threads int, f func(lo, hi int)) {
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
			f(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// Build runs train and add. vectors is row-major (n, dim).
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	var vals [5]int
	for i, key := range []string{"nlist", "iters", "m", "nbits", "train_size"} {
		v, err := intParam(params, BuildDefaults, key)
		if err != nil {
			return nil, err
		}
		vals[i] = v
	}
	nlist, iters, m, nbits, trainSize := vals[0], vals[1], vals[2], vals[3], vals[4]
	metric, ok := params["metric"].(string)
	if !ok {
		metric = BuildDefaults["metric"].(string)
	}
	if nbits != 8 {
		return nil, fmt.Errorf("ivfpq: nbits=%d is not supported, only 8", nbits)
	}
	if metric != "ip" && metric != "l2" {
		return nil, fmt.Errorf("ivfpq: metric must be ip or l2, got %q", metric)
	}
	if m <= 0 || dim%m != 0 {
		return nil, fmt.Errorf("ivfpq: dim %d is not divisible by m=%d", dim, m)
	}
	if nlist < 1 {
		return nil, fmt.Errorf("ivfpq: nlist must be >= 1, got %d", nlist)
	}
	if n > math.MaxInt32 {
		return nil, fmt.Errorf("ivfpq: n=%d does not fit int32 IDs", n)
	}
	if threads < 1 {
		threads = 1
	}
	trainN := min(trainSize, n)
	if trainN < max(nlist, ksub) {
		return nil, fmt.Errorf("ivfpq: need at least %d training rows, have %d", max(nlist, ksub), trainN)
	}
	dsub := dim / m
	ix := &Index{vectors: vectors, n: n, dim: dim, nlist: nlist, m: m, dsub: dsub, l2: metric == "l2"}

	// Train, step 1: coarse centers (dot product, normalized).
	start := time.Now()
	var err error
	ix.centers, err = kmeans.Train(vectors[:trainN*dim], trainN, dim, kmeans.Config{
		K: nlist, Iters: iters, Seed: seed, Threads: threads,
	})
	if err != nil {
		return nil, err
	}
	// Train, step 2: residuals of the training rows.
	labels := make([]int32, n)
	resid := make([]float32, trainN*dim)
	parallel(trainN, threads, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			x := vectors[i*dim : (i+1)*dim]
			c, _ := kmeans.Nearest(x, ix.centers, nlist, dim, false)
			labels[i] = int32(c)
			cv := ix.centers[c*dim : (c+1)*dim]
			r := resid[i*dim : (i+1)*dim]
			for d := range r {
				r[d] = x[d] - cv[d]
			}
		}
	})
	// Train, step 3: one codebook per sub-vector on the residuals.
	ix.codebooks = make([]float32, m*ksub*dsub)
	sub := make([]float32, trainN*dsub)
	for j := 0; j < m; j++ {
		for i := 0; i < trainN; i++ {
			copy(sub[i*dsub:(i+1)*dsub], resid[i*dim+j*dsub:i*dim+(j+1)*dsub])
		}
		cfg := kmeans.Config{K: ksub, Iters: iters, Seed: seed + uint64(j), Threads: threads, L2: ix.l2, NoNormalize: true}
		cb, err := kmeans.Train(sub, trainN, dsub, cfg)
		if err != nil {
			return nil, fmt.Errorf("ivfpq: codebook %d: %w", j, err)
		}
		copy(ix.codebooks[j*ksub*dsub:(j+1)*ksub*dsub], cb)
	}
	resid, sub = nil, nil
	ix.trainS = time.Since(start).Seconds()

	// Add: assign the rows not assigned in train (the first trainN rows have
	// the same labels), then build the CSR lists and encode the residuals.
	start = time.Now()
	parallel(n-trainN, threads, func(lo, hi int) {
		for i := trainN + lo; i < trainN+hi; i++ {
			c, _ := kmeans.Nearest(vectors[i*dim:(i+1)*dim], ix.centers, nlist, dim, false)
			labels[i] = int32(c)
		}
	})
	// Counting sort by label: rows stay in ascending ID order in each list,
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
	ix.codes = make([]uint8, n*m)
	parallel(n, threads, func(lo, hi int) {
		r := make([]float32, dim)
		for p := lo; p < hi; p++ {
			i := int(ix.ids[p])
			x := vectors[i*dim : (i+1)*dim]
			c := int(labels[i])
			cv := ix.centers[c*dim : (c+1)*dim]
			for d := range r {
				r[d] = x[d] - cv[d]
			}
			for j := 0; j < m; j++ {
				k, _ := kmeans.Nearest(r[j*dsub:(j+1)*dsub], ix.codebooks[j*ksub*dsub:(j+1)*ksub*dsub], ksub, dsub, ix.l2)
				ix.codes[p*m+j] = uint8(k)
			}
		}
	})
	ix.addS = time.Since(start).Seconds()
	ix.table = make([]float32, m*ksub)
	ix.qres = make([]float32, dim)
	return ix, nil
}

// Codes returns the (n, m) code array in list order. Tests use it.
func (ix *Index) Codes() []uint8 { return ix.codes }

// Lists returns the CSR ids and offsets. Tests use them.
func (ix *Index) Lists() (ids, offsets []int32) { return ix.ids, ix.offsets }

// fillTable writes T[j][k] = score of q_j against codebook[j][k] under the metric.
func (ix *Index) fillTable(q []float32) {
	m, dsub := ix.m, ix.dsub
	T := ix.table
	for j := 0; j < m; j++ {
		qj := q[j*dsub : (j+1)*dsub]
		cb := ix.codebooks[j*ksub*dsub : (j+1)*ksub*dsub]
		for c := 0; c < ksub; c++ {
			T[j*ksub+c] = kmeans.Score(qj, cb[c*dsub:(c+1)*dsub], ix.l2)
		}
	}
}

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	nprobe, err := intParam(params, SearchDefaults, "nprobe")
	if err != nil || nprobe < 1 {
		nprobe = 8
	}
	nprobe = min(nprobe, ix.nlist)
	rerank, err := intParam(params, SearchDefaults, "rerank")
	if err != nil || rerank < 0 {
		rerank = 0
	}
	if ix.probeK == nil || ix.probeK.K() != nprobe {
		ix.probeK = distance.NewTopK(nprobe)
	}
	keep := k
	if rerank > 0 {
		keep = rerank
	}
	if ix.topk == nil || ix.topk.K() != keep {
		ix.topk = distance.NewTopK(keep)
	}
	d, m := ix.dim, ix.m

	// Probe: the nprobe best centers by dot product; ties go to the lower list.
	pk := ix.probeK
	pk.Reset()
	for c := 0; c < ix.nlist; c++ {
		pk.Push(int64(c), distance.Dot(query, ix.centers[c*d:(c+1)*d]))
	}
	lists, qc := pk.Results()

	if !ix.l2 {
		ix.fillTable(query) // same table for every list
	}
	T := ix.table
	tk := ix.topk
	tk.Reset()
	scanned := 0
	for li, c := range lists {
		if c < 0 {
			continue
		}
		var base float32
		if ix.l2 {
			cv := ix.centers[int(c)*d : (int(c)+1)*d]
			for i := range ix.qres {
				ix.qres[i] = query[i] - cv[i]
			}
			ix.fillTable(ix.qres)
		} else {
			base = qc[li]
		}
		lo, hi := int(ix.offsets[c]), int(ix.offsets[c+1])
		for p := lo; p < hi; p++ {
			code := ix.codes[p*m : (p+1)*m]
			s := base
			for j, cj := range code {
				s += T[j*ksub+int(cj)]
			}
			tk.Push(int64(ix.ids[p]), s)
		}
		scanned += hi - lo
	}
	ix.dists += int64(ix.nlist + scanned)
	if rerank == 0 {
		return tk.Results()
	}
	cand, _ := tk.Results()
	if ix.final == nil || ix.final.K() != k {
		ix.final = distance.NewTopK(k)
	}
	fk := ix.final
	fk.Reset()
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
// ivf part (centers + int32 IDs + int32 offsets) + pq part (codebooks + codes).
func IndexBytes(ix *Index) int64 {
	return int64(ix.nlist)*int64(ix.dim)*4 + int64(len(ix.ids))*4 + int64(len(ix.offsets))*4 +
		int64(ix.m)*ksub*int64(ix.dsub)*4 + int64(len(ix.codes))
}

// TrainSeconds returns the train time of the build.
func (ix *Index) TrainSeconds() float64 { return ix.trainS }

// AddSeconds returns the add time of the build.
func (ix *Index) AddSeconds() float64 { return ix.addS }

// DistanceComputations returns the total distance computations of Search so
// far: per query, nlist + scanned rows (+ reranked rows).
func (ix *Index) DistanceComputations() int64 { return ix.dists }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any {
	return map[string]any{"id_type": "int32", "id_bytes": 4}
}

// SearchCounters returns cumulative per-search counters. IVF-PQ has none.
func (ix *Index) SearchCounters() map[string]float64 { return map[string]float64{} }
