// Package kmeans implements the shared k-means of CONTRACT.md section 6.2.
// It is used by ivf, pq, ivf_pq and diskann. PQ codebooks use the L2 and
// NoNormalize options (section 6.4.1).
package kmeans

import (
	"fmt"
	"math"
	"sync"

	"vro/indexes/go/distance"
	"vro/indexes/go/splitmix"
)

// Config holds the k-means settings.
type Config struct {
	K       int    // number of centers
	Iters   int    // maximum iterations (contract default 20)
	Seed    uint64 // seed of the PRNG used for initialization
	Threads int    // goroutines for the assignment step
	// L2 assigns by lowest squared Euclidean distance instead of highest dot product.
	L2 bool
	// NoNormalize skips the L2 normalization of centers (PQ sub-vectors).
	NoNormalize bool
}

// TrainSize returns the default number of training rows: min(n, 256*k), but not below k.
func TrainSize(n, k int) int {
	t := 256 * k
	if n < t {
		t = n
	}
	if t < k {
		t = k
	}
	return t
}

// Train runs k-means on points, a row-major (n, dim) array. The caller passes
// the training rows only (the first train_size corpus rows). It returns the
// centers as a row-major (K, dim) array.
func Train(points []float32, n, dim int, cfg Config) ([]float32, error) {
	if cfg.K <= 0 || n < cfg.K {
		return nil, fmt.Errorf("kmeans: need at least k=%d training rows, have %d", cfg.K, n)
	}
	if cfg.Threads < 1 {
		cfg.Threads = 1
	}
	centers := initCenters(points, n, dim, cfg)
	assign := make([]int32, n)
	for i := range assign {
		assign[i] = -1
	}
	labels := make([]int32, n)
	scores := make([]float32, n)
	// Per iteration: (a) assign, (b) stop if no step-a label changed since the
	// previous step a, (c) fix empty clusters, (d) recompute centers.
	for it := 0; it < cfg.Iters; it++ {
		if assignAll(points, n, dim, centers, cfg, assign, scores) == 0 {
			break
		}
		copy(labels, assign) // the fix edits labels, not the step-a labels
		update(points, n, dim, centers, cfg, labels, scores)
	}
	return centers, nil
}

// initCenters picks K distinct training rows with NextBelow(n), drawing again on a repeat.
func initCenters(points []float32, n, dim int, cfg Config) []float32 {
	rng := splitmix.New(cfg.Seed)
	centers := make([]float32, cfg.K*dim)
	used := make(map[uint64]bool, cfg.K)
	for c := 0; c < cfg.K; c++ {
		r := rng.NextBelow(uint64(n))
		for used[r] {
			r = rng.NextBelow(uint64(n))
		}
		used[r] = true
		copy(centers[c*dim:(c+1)*dim], points[int(r)*dim:(int(r)+1)*dim])
	}
	return centers
}

// Score returns the assignment score of point p against center c: the dot
// product, or the negative squared distance in L2 mode. Higher is better.
func Score(p, c []float32, l2 bool) float32 {
	if !l2 {
		return distance.Dot(p, c)
	}
	c = c[:len(p)]
	var s float32
	for i := range p {
		d := p[i] - c[i]
		s += d * d
	}
	return -s
}

// Nearest returns the best center for p and its score.
func Nearest(p, centers []float32, k, dim int, l2 bool) (int, float32) {
	best, bestScore := 0, float32(math.Inf(-1))
	for c := 0; c < k; c++ {
		if s := Score(p, centers[c*dim:(c+1)*dim], l2); s > bestScore {
			best, bestScore = c, s
		}
	}
	return best, bestScore
}

// assignAll assigns every point to its best center in parallel and returns
// how many assignments changed.
func assignAll(points []float32, n, dim int, centers []float32, cfg Config, assign []int32, scores []float32) int {
	changed := make([]int, cfg.Threads)
	chunk := (n + cfg.Threads - 1) / cfg.Threads
	var wg sync.WaitGroup
	for t := 0; t < cfg.Threads; t++ {
		lo, hi := t*chunk, min((t+1)*chunk, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(t, lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				c, s := Nearest(points[i*dim:(i+1)*dim], centers, cfg.K, dim, cfg.L2)
				if int32(c) != assign[i] {
					changed[t]++
					assign[i] = int32(c)
				}
				scores[i] = s
			}
		}(t, lo, hi)
	}
	wg.Wait()
	total := 0
	for _, c := range changed {
		total += c
	}
	return total
}

// update fixes empty clusters, then sets each center to the mean of its
// points and normalizes it unless NoNormalize is set. It edits labels.
func update(points []float32, n, dim int, centers []float32, cfg Config, labels []int32, scores []float32) {
	counts := make([]int, cfg.K)
	for _, c := range labels {
		counts[c]++
	}
	fixEmpty(labels, scores, counts)
	sums := make([]float64, cfg.K*dim)
	for i := 0; i < n; i++ {
		c := int(labels[i])
		addRow(sums[c*dim:(c+1)*dim], points[i*dim:(i+1)*dim], 1)
	}
	for c := 0; c < cfg.K; c++ {
		setCenter(centers[c*dim:(c+1)*dim], sums[c*dim:(c+1)*dim], counts[c], !cfg.NoNormalize)
	}
}

// fixEmpty handles each empty cluster in index order: it moves the worst-fit
// point (lowest score to its own center) into it. Only points whose cluster
// has at least 2 members are candidates, so no cluster becomes empty.
func fixEmpty(labels []int32, scores []float32, counts []int) {
	for c := range counts {
		if counts[c] > 0 {
			continue
		}
		w, ws := -1, float32(math.Inf(1))
		for i, s := range scores {
			if counts[labels[i]] >= 2 && s < ws {
				w, ws = i, s
			}
		}
		counts[labels[w]]--
		labels[w] = int32(c)
		counts[c] = 1
	}
}

func addRow(sum []float64, row []float32, sign float64) {
	row = row[:len(sum)]
	for i := range sum {
		sum[i] += sign * float64(row[i])
	}
}

func setCenter(center []float32, sum []float64, count int, normalize bool) {
	var norm float64
	for i := range center {
		v := sum[i] / float64(count)
		center[i] = float32(v)
		norm += v * v
	}
	if !normalize || norm == 0 {
		return
	}
	inv := 1 / math.Sqrt(norm)
	for i := range center {
		center[i] = float32(sum[i] / float64(count) * inv)
	}
}
