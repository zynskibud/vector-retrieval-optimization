package kmeans

import (
	"math"
	"testing"

	"vro/indexes/go/splitmix"
)

// blobs returns n unit vectors around 4 well-separated directions.
func blobs(n, dim int) []float32 {
	rng := splitmix.New(1)
	pts := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		row := pts[i*dim : (i+1)*dim]
		row[i%4] = 1
		var norm float64
		for j := range row {
			row[j] += float32(rng.NextF64()-0.5) * 0.1
			norm += float64(row[j]) * float64(row[j])
		}
		for j := range row {
			row[j] /= float32(math.Sqrt(norm))
		}
	}
	return pts
}

func TestTrainNormalizedCenters(t *testing.T) {
	const n, dim, k = 400, 8, 4
	c, err := Train(blobs(n, dim), n, dim, Config{K: k, Iters: 20, Seed: 42, Threads: 3})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < k; i++ {
		var norm float64
		for _, v := range c[i*dim : (i+1)*dim] {
			norm += float64(v) * float64(v)
		}
		if math.Abs(norm-1) > 1e-4 {
			t.Errorf("center %d has squared norm %v", i, norm)
		}
	}
}

func TestThreadsDeterministic(t *testing.T) {
	const n, dim, k = 400, 8, 6
	pts := blobs(n, dim)
	a, _ := Train(pts, n, dim, Config{K: k, Iters: 20, Seed: 42, Threads: 1})
	b, _ := Train(pts, n, dim, Config{K: k, Iters: 20, Seed: 42, Threads: 4, L2: false})
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("centers differ between 1 and 4 threads")
		}
	}
}

func TestTrainSize(t *testing.T) {
	if got := TrainSize(100000, 1024); got != 100000 {
		t.Errorf("TrainSize = %d", got)
	}
	if got := TrainSize(1000000, 10); got != 2560 {
		t.Errorf("TrainSize = %d", got)
	}
}
