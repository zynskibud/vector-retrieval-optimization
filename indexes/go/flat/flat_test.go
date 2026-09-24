package flat

import (
	"path/filepath"
	"runtime"
	"testing"

	"vro/indexes/go/npy"
)

func devDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "data", "processed", "dev")
}

// TestRecallDev checks recall@10 = 1.0 and top-1 equality against ground truth.
func TestRecallDev(t *testing.T) {
	dir := devDir()
	vec, n, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	qs, q, _, err := npy.ReadFloat32(filepath.Join(dir, "queries.npy"))
	if err != nil {
		t.Fatal(err)
	}
	gt, _, gtCols, err := npy.ReadInt64(filepath.Join(dir, "ground_truth.npy"))
	if err != nil {
		t.Fatal(err)
	}
	ix, err := Build(vec, n, dim, nil, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	const k = 10
	hits := 0
	for i := 0; i < q; i++ {
		ids, _ := Search(ix, qs[i*dim:(i+1)*dim], k, nil)
		truth := gt[i*gtCols : i*gtCols+k]
		if ids[0] != truth[0] {
			t.Errorf("query %d: top-1 %d, ground truth %d", i, ids[0], truth[0])
		}
		set := make(map[int64]bool, k)
		for _, id := range truth {
			set[id] = true
		}
		for _, id := range ids {
			if set[id] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(q*k)
	t.Logf("recall@10 = %v", recall)
	if recall != 1.0 {
		t.Fatalf("recall@10 = %v, want 1.0", recall)
	}
	if got := ix.DistanceComputations(); got != int64(q*n) {
		t.Errorf("distance computations %d, want %d", got, q*n)
	}
}
