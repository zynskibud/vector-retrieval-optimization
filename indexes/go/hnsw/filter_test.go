package hnsw

import (
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"vro/indexes/go/distance"
	"vro/indexes/go/npy"
)

// TestFilter checks filtered search (CONTRACT.md section 11.5) on the first
// 20,000 dev rows. The truth is computed here by brute force over the passing rows.
func TestFilter(t *testing.T) {
	const rows, k = 20000, 10
	dir := devDir()
	DataDir = dir
	vec, n, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	n = min(n, rows)
	vec = vec[:n*dim]
	qs, q, _, err := npy.ReadFloat32(filepath.Join(dir, "queries.npy"))
	if err != nil {
		t.Fatal(err)
	}
	ix, err := Build(vec, n, dim, map[string]any{}, runtime.NumCPU(), 42)
	if err != nil {
		t.Fatal(err)
	}

	// filter=none gives the same results as a search without the key.
	for i := 0; i < q; i++ {
		query := qs[i*dim : (i+1)*dim]
		a, as := Search(ix, query, k, map[string]any{})
		b, bs := Search(ix, query, k, map[string]any{"filter": "none"})
		if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(as, bs) {
			t.Fatalf("query %d: filter=none %v differs from no filter %v", i, b, a)
		}
	}

	for _, name := range []string{"top10", "top01"} {
		mask, err := npy.FilterMask(dir, name)
		if err != nil {
			t.Fatal(err)
		}
		hits, total := 0, 0
		p := map[string]any{"filter": name}
		for i := 0; i < q; i++ {
			query := qs[i*dim : (i+1)*dim]
			ids, _ := Search(ix, query, k, p)
			tk := distance.NewTopK(k)
			for r := 0; r < n; r++ {
				if mask[r] {
					tk.Push(int64(r), distance.Dot(query, vec[r*dim:(r+1)*dim]))
				}
			}
			truth, _ := tk.Results()
			set := map[int64]bool{}
			for _, id := range truth {
				if id >= 0 {
					set[id] = true
					total++
				}
			}
			for _, id := range ids {
				if id == -1 {
					continue
				}
				if id < 0 || id >= int64(n) || !mask[id] {
					t.Fatalf("filter %s query %d: id %d does not pass the filter", name, i, id)
				}
				if set[id] {
					hits++
				}
			}
		}
		recall := float64(hits) / float64(total)
		t.Logf("filter=%s recall@10 = %.4f, counters %v", name, recall, ix.SearchCounters())
		if name == "top10" && recall < 0.85 {
			t.Errorf("filter=top10 recall@10 = %.4f, want >= 0.85", recall)
		}
	}
}
