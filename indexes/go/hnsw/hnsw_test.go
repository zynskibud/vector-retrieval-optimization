package hnsw

import (
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"testing"

	"vro/indexes/go/distance"
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
	ix        *Index
	err       error
}

var (
	devOnce sync.Once
	dev     devData
)

// loadDev reads the dev set and builds one index on all 100k rows with
// default params and all CPU cores. The tests share it.
func loadDev(t *testing.T) *devData {
	t.Helper()
	devOnce.Do(func() {
		d := &dev
		dir := devDir()
		if d.vec, d.n, d.dim, d.err = npy.ReadFloat32(filepath.Join(dir, "vectors.npy")); d.err != nil {
			return
		}
		if d.qs, d.q, _, d.err = npy.ReadFloat32(filepath.Join(dir, "queries.npy")); d.err != nil {
			return
		}
		if d.gt, _, d.gtCols, d.err = npy.ReadInt64(filepath.Join(dir, "ground_truth.npy")); d.err != nil {
			return
		}
		d.ix, d.err = Build(d.vec, d.n, d.dim, map[string]any{}, runtime.NumCPU(), 42)
		if d.err == nil {
			t.Logf("build on %d rows: %.1f s, top layer %d, nodes per layer %v",
				d.n, d.ix.AddSeconds(), d.ix.TopLayer(), d.ix.NodesPerLayer())
		}
	})
	if dev.err != nil {
		t.Fatal(dev.err)
	}
	return &dev
}

func recallAt(d *devData, ef int) float64 {
	const k = 10
	hits := 0
	p := map[string]any{"ef": int64(ef)}
	for i := 0; i < d.q; i++ {
		ids, _ := Search(d.ix, d.qs[i*d.dim:(i+1)*d.dim], k, p)
		truth := d.gt[i*d.gtCols : i*d.gtCols+k]
		for _, id := range ids {
			if slices.Contains(truth, id) {
				hits++
			}
		}
	}
	return float64(hits) / float64(d.q*k)
}

func TestRecallDev(t *testing.T) {
	d := loadDev(t)
	r16, r64, r128 := recallAt(d, 16), recallAt(d, 64), recallAt(d, 128)
	t.Logf("recall@10: ef=16 %.4f, ef=64 %.4f, ef=128 %.4f", r16, r64, r128)
	if r64 < 0.95 {
		t.Errorf("recall@10 at ef=64 = %.4f, want >= 0.95", r64)
	}
	if !(r128 >= r64 && r64 >= r16) {
		t.Errorf("recall not monotonic in ef: %.4f %.4f %.4f", r16, r64, r128)
	}
}

func TestLevels(t *testing.T) {
	a, b := Levels(100000, 16, 42), Levels(100000, 16, 42)
	if !slices.Equal(a, b) {
		t.Fatal("levels differ for the same seed")
	}
	up := 0
	for _, l := range a {
		if l >= 1 {
			up++
		}
	}
	frac := float64(up) / float64(len(a))
	t.Logf("fraction with level >= 1: %.4f (expected 1/m = 0.0625)", frac)
	if frac < 0.04 || frac > 0.09 {
		t.Errorf("fraction %.4f outside [0.04, 0.09]", frac)
	}
	if slices.Equal(a, Levels(100000, 16, 43)) {
		t.Error("levels equal for different seeds")
	}
}

func TestSlotLimits(t *testing.T) {
	d := loadDev(t)
	ix := d.ix
	for i, c := range ix.cnt0 {
		if c < 0 || int(c) > ix.m0 {
			t.Fatalf("node %d layer 0 count %d > %d", i, c, ix.m0)
		}
	}
	for b, c := range ix.cntUpper {
		if c < 0 || int(c) > ix.m {
			t.Fatalf("upper block %d count %d > %d", b, c, ix.m)
		}
	}
	// Every edge on layer l points to a node that exists on layer l.
	for i := 0; i < ix.n; i++ {
		for l := 0; l <= int(ix.levels[i]); l++ {
			sl, c := ix.slots(int32(i), l)
			for _, e := range sl[:*c] {
				if int(ix.levels[e]) < l || int(e) == i {
					t.Fatalf("bad edge %d -> %d on layer %d", i, e, l)
				}
			}
		}
	}
}

// reachable0 counts the nodes reachable from the entry point on layer 0 by BFS.
func reachable0(ix *Index) int {
	seen := make([]bool, ix.n)
	queue := []int32{ix.entry}
	seen[ix.entry] = true
	count := 1
	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]
		sl, c := ix.slots(v, 0)
		for _, e := range sl[:*c] {
			if !seen[e] {
				seen[e] = true
				count++
				queue = append(queue, e)
			}
		}
	}
	return count
}

// bruteTop10 returns the exact top-10 IDs of each query over the first n rows.
func bruteTop10(d *devData, n int) [][]int64 {
	out := make([][]int64, d.q)
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w; i < d.q; i += runtime.NumCPU() {
				tk := distance.NewTopK(10)
				q := d.qs[i*d.dim : (i+1)*d.dim]
				for r := 0; r < n; r++ {
					tk.Push(int64(r), distance.Dot(q, d.vec[r*d.dim:(r+1)*d.dim]))
				}
				out[i], _ = tk.Results()
			}
		}(w)
	}
	wg.Wait()
	return out
}

func recallTruth(d *devData, ix *Index, truth [][]int64, ef int) float64 {
	hits := 0
	p := map[string]any{"ef": int64(ef)}
	for i := 0; i < d.q; i++ {
		ids, _ := Search(ix, d.qs[i*d.dim:(i+1)*d.dim], 10, p)
		for _, id := range ids {
			if slices.Contains(truth[i], id) {
				hits++
			}
		}
	}
	return float64(hits) / float64(d.q*10)
}

// TestLayer0Connected builds the first 20000 rows with threads = 1 and with
// all cores (with the repair pass). Both must reach every node on layer 0
// from the entry point by BFS, and the parallel recall@10 at ef=64 must be
// within 0.01 of the threads = 1 recall (truth by brute force on 20000 rows).
// The repair pass runs after both builds. The shared parallel 100k build
// must also be fully reachable.
func TestLayer0Connected(t *testing.T) {
	d := loadDev(t)
	n := 20000
	seq, err := Build(d.vec[:n*d.dim], n, d.dim, nil, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	par, err := Build(d.vec[:n*d.dim], n, d.dim, nil, runtime.NumCPU(), 42)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		ix   *Index
	}{{"threads=1", seq}, {"all cores", par}} {
		got := reachable0(c.ix)
		t.Logf("%s, %d rows: build %.1f s, %d of %d reachable, extra %v",
			c.name, n, c.ix.AddSeconds(), got, n, c.ix.Extra())
		if got != n {
			t.Errorf("%s: layer 0 not connected: %d of %d reachable", c.name, got, n)
		}
	}
	truth := bruteTop10(d, n)
	rs, rp := recallTruth(d, seq, truth, 64), recallTruth(d, par, truth, 64)
	t.Logf("20000 rows recall@10 at ef=64: threads=1 %.4f, all cores %.4f", rs, rp)
	if rp < rs-0.01 || rp > rs+0.01 {
		t.Errorf("parallel recall %.4f not within 0.01 of threads=1 recall %.4f", rp, rs)
	}
	got := reachable0(d.ix)
	t.Logf("parallel build, %d rows: %d of %d reachable, extra %v", d.n, got, d.n, d.ix.Extra())
	if got != d.n {
		t.Errorf("100k parallel build: %d of %d reachable", got, d.n)
	}
}

func TestDistanceCount(t *testing.T) {
	d := loadDev(t)
	ix := d.ix
	for i := 0; i < 50; i++ {
		before := ix.DistanceComputations()
		ids, scores := Search(ix, d.qs[i*d.dim:(i+1)*d.dim], 10, map[string]any{"ef": int64(64)})
		got := ix.DistanceComputations() - before
		if got <= 0 || got >= int64(ix.n) {
			t.Fatalf("query %d: %d distance computations, want in (0, %d)", i, got, ix.n)
		}
		for j := 1; j < len(ids); j++ {
			if scores[j] > scores[j-1] || (scores[j] == scores[j-1] && ids[j] < ids[j-1]) {
				t.Fatalf("query %d: results not ordered", i)
			}
		}
	}
}

// TestSequentialDeterministic builds twice with threads = 1 on 5000 rows and
// checks the graphs are identical, and that k > found pads with -1.
func TestSequentialDeterministic(t *testing.T) {
	d := loadDev(t)
	n := 5000
	v := d.vec[:n*d.dim]
	a, err := Build(v, n, d.dim, nil, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := Build(v, n, d.dim, nil, 1, 42)
	if !slices.Equal(a.links0, b.links0) || !slices.Equal(a.cnt0, b.cnt0) ||
		!slices.Equal(a.linksUpper, b.linksUpper) || a.entry != b.entry {
		t.Fatal("threads=1 builds differ")
	}
	// The entry is the highest level, ties to the lowest row.
	best := 0
	for i, l := range a.levels {
		if l > a.levels[best] {
			best = i
		}
	}
	if int(a.entry) != best {
		t.Errorf("entry %d, want %d", a.entry, best)
	}
	tiny, _ := Build(v[:3*d.dim], 3, d.dim, nil, 1, 42)
	ids, scores := Search(tiny, d.qs[:d.dim], 10, nil)
	if ids[2] < 0 || ids[3] != -1 || scores[3] > -1e30 {
		t.Errorf("padding wrong: %v %v", ids, scores)
	}
}
