package hnsw

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vro/indexes/go/npy"
)

// TestInsertConcurrent builds on the first 90% of n rows with room for n, then
// inserts the last 10% in batches of 100 while 4 goroutines search without
// pause. Every search result must be a valid, unique row. After the inserts
// and one Repair, every node is reachable on layer 0 and NodesPerLayer counts
// n rows. Run it with -race: Search must be lock-free and race-free.
func TestInsertConcurrent(t *testing.T) {
	n := 20000
	if raceEnabled {
		n = 6000
	}
	dir := devDir()
	vec, _, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	qs, q, _, err := npy.ReadFloat32(filepath.Join(dir, "queries.npy"))
	if err != nil {
		t.Fatal(err)
	}
	nb := n * 9 / 10
	ix, err := BuildCap(vec[:n*dim], nb, n, dim, nil, 4, 42)
	if err != nil {
		t.Fatal(err)
	}
	if ix.Len() != nb {
		t.Fatalf("Len = %d after build, want %d", ix.Len(), nb)
	}
	var stop atomic.Bool
	var bad, done atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			p := map[string]any{"ef": int64(64)}
			for i := w; !stop.Load(); i = (i + 1) % q {
				ids, _ := Search(ix, qs[i*dim:(i+1)*dim], 10, p)
				seen := map[int64]bool{}
				for _, id := range ids {
					if id < 0 || id >= int64(n) || seen[id] {
						if bad.Add(1) <= 3 {
							t.Logf("bad result: %v", ids)
						}
					}
					seen[id] = true
				}
				done.Add(1)
			}
		}(w)
	}
	t0 := time.Now()
	for lo := nb; lo < n; lo += 100 {
		hi := min(lo+100, n)
		ids := make([]int64, hi-lo)
		for i := range ids {
			ids[i] = int64(lo + i)
		}
		if err := Insert(ix, ids, vec[lo*dim:hi*dim]); err != nil {
			t.Fatal(err)
		}
	}
	insS := time.Since(t0).Seconds()
	stop.Store(true)
	wg.Wait()
	t.Logf("inserted %d rows in %.2f s", n-nb, insS)
	if err := Insert(ix, []int64{int64(n)}, vec[:dim]); err == nil {
		t.Error("Insert past capacity: no error")
	}
	u, a := Repair(ix)
	got := reachable0(ix)
	t.Logf("%d rows (%d built, %d inserted): %d searches during inserts, %d bad; repair: %d unreachable before, %d edges added; %d reachable",
		n, nb, n-nb, done.Load(), bad.Load(), u, a, got)
	if bad.Load() != 0 {
		t.Errorf("%d searches returned invalid IDs", bad.Load())
	}
	if ix.Len() != n || got != n {
		t.Errorf("Len %d, reachable %d, want %d", ix.Len(), got, n)
	}
	if ix.NodesPerLayer()[0] != n {
		t.Errorf("layer 0 has %d nodes, want %d", ix.NodesPerLayer()[0], n)
	}
}

// TestInsertLevels checks that BuildCap draws the levels of all capN rows,
// so an inserted row has the level of a static build on capN rows.
func TestInsertLevels(t *testing.T) {
	dim, capN := 4, 500
	v := make([]float32, capN*dim)
	for i := 0; i < capN; i++ {
		v[i*dim+i%dim] = 1
	}
	ix, err := BuildCap(v, 400, capN, dim, nil, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	want := Levels(capN, 16, 42)
	for i := range want {
		if ix.levels[i] != want[i] {
			t.Fatalf("row %d: level %d, want %d", i, ix.levels[i], want[i])
		}
	}
}
