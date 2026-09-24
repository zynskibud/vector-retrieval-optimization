package ivf

import (
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"vro/indexes/go/npy"
)

func devDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "data", "processed", "dev")
}

type devData struct {
	vec, qs   []float32
	n, dim, q int
	gt        []int64
	gtCols    int
	ix4       *Index // default params, built with all CPUs
}

var (
	devOnce sync.Once
	dev     *devData
	devErr  error
)

// loadDev reads the dev set and builds the index once for all tests.
func loadDev(t *testing.T) *devData {
	t.Helper()
	devOnce.Do(func() {
		d := &devData{}
		dir := devDir()
		if d.vec, d.n, d.dim, devErr = npy.ReadFloat32(filepath.Join(dir, "vectors.npy")); devErr != nil {
			return
		}
		if d.qs, d.q, _, devErr = npy.ReadFloat32(filepath.Join(dir, "queries.npy")); devErr != nil {
			return
		}
		if d.gt, _, d.gtCols, devErr = npy.ReadInt64(filepath.Join(dir, "ground_truth.npy")); devErr != nil {
			return
		}
		if d.ix4, devErr = Build(d.vec, d.n, d.dim, nil, runtime.NumCPU(), 42); devErr != nil {
			return
		}
		dev = d
	})
	if devErr != nil {
		t.Fatal(devErr)
	}
	return dev
}

func recallAt10(t *testing.T, d *devData, ix *Index, nprobe int) float64 {
	const k = 10
	hits := 0
	p := map[string]any{"nprobe": nprobe}
	for i := 0; i < d.q; i++ {
		ids, _ := Search(ix, d.qs[i*d.dim:(i+1)*d.dim], k, p)
		set := make(map[int64]bool, k)
		for _, id := range d.gt[i*d.gtCols : i*d.gtCols+k] {
			set[id] = true
		}
		for _, id := range ids {
			if id < -1 || id >= int64(d.n) {
				t.Fatalf("query %d: invalid id %d", i, id)
			}
			if set[id] {
				hits++
			}
		}
	}
	return float64(hits) / float64(d.q*k)
}

func TestRecallDev(t *testing.T) {
	d := loadDev(t)
	r8 := recallAt10(t, d, d.ix4, 8)
	r64 := recallAt10(t, d, d.ix4, 64)
	t.Logf("recall@10 nprobe=8: %v, nprobe=64: %v", r8, r64)
	if r8 < 0.75 {
		t.Errorf("recall@10 nprobe=8 = %v, want >= 0.75", r8)
	}
	if r64 < r8 {
		t.Errorf("recall@10 nprobe=64 = %v < nprobe=8 = %v", r64, r8)
	}
}

func TestCSR(t *testing.T) {
	d := loadDev(t)
	ix := d.ix4
	if len(ix.offsets) != ix.nlist+1 || ix.offsets[0] != 0 || int(ix.offsets[ix.nlist]) != d.n {
		t.Fatalf("offsets: len %d, first %d, last %d; want len %d, 0, %d",
			len(ix.offsets), ix.offsets[0], ix.offsets[len(ix.offsets)-1], ix.nlist+1, d.n)
	}
	for c := 0; c < ix.nlist; c++ {
		if ix.offsets[c+1] < ix.offsets[c] {
			t.Fatalf("offsets decrease at list %d", c)
		}
	}
	seen := make([]int, d.n)
	for _, id := range ix.ids {
		if id < 0 || int(id) >= d.n {
			t.Fatalf("invalid id %d in lists", id)
		}
		seen[id]++
	}
	for i, s := range seen {
		if s != 1 {
			t.Fatalf("row %d appears %d times", i, s)
		}
	}
}

func TestDistanceCount(t *testing.T) {
	d := loadDev(t)
	ix := d.ix4
	for i := 0; i < 50; i++ {
		before := ix.DistanceComputations()
		Search(ix, d.qs[i*d.dim:(i+1)*d.dim], 10, nil)
		got := ix.DistanceComputations() - before
		if got < int64(ix.nlist) || got > int64(ix.nlist+d.n) {
			t.Fatalf("query %d: %d distance computations, want in [%d, %d]", i, got, ix.nlist, ix.nlist+d.n)
		}
	}
}

func TestThreadsIdentical(t *testing.T) {
	d := loadDev(t)
	// A 20k-row subset with nlist=128 keeps two builds inside the time budget.
	const rows = 20000
	p := map[string]any{"nlist": 128}
	ix1, err := Build(d.vec[:rows*d.dim], rows, d.dim, p, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	ix4, err := Build(d.vec[:rows*d.dim], rows, d.dim, p, 4, 42)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < d.q; i++ {
		a, _ := Search(ix1, d.qs[i*d.dim:(i+1)*d.dim], 10, nil)
		b, _ := Search(ix4, d.qs[i*d.dim:(i+1)*d.dim], 10, nil)
		for j := range a {
			if a[j] != b[j] {
				t.Fatalf("query %d: result %d differs: %d vs %d", i, j, a[j], b[j])
			}
		}
	}
	if len(ix1.ids) != len(ix4.ids) {
		t.Fatal("id array lengths differ")
	}
	for i := range ix1.ids {
		if ix1.ids[i] != ix4.ids[i] {
			t.Fatalf("ids differ at %d", i)
		}
	}
	for i := range ix1.offsets {
		if ix1.offsets[i] != ix4.offsets[i] {
			t.Fatalf("offsets differ at %d", i)
		}
	}
}
