package flat

// Phase 5 tests (CONTRACT.md section 13.5) on the first 20,000 dev rows. The
// truth is computed here by brute force.

import (
	"path/filepath"
	"runtime"
	"testing"

	"vro/indexes/go/distance"
	"vro/indexes/go/npy"
)

const changeRows = 20000

type changeData struct {
	vec, qs   []float32 // vec is a private copy: Update writes into it
	n, dim, q int
}

func loadChangeData(t *testing.T) changeData {
	t.Helper()
	dir := devDir()
	vec, n, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	n = min(n, changeRows)
	qs, q, _, err := npy.ReadFloat32(filepath.Join(dir, "queries.npy"))
	if err != nil {
		t.Fatal(err)
	}
	return changeData{vec: append([]float32(nil), vec[:n*dim]...), qs: qs, n: n, dim: dim, q: q}
}

// changeTruth returns the exact top-k over the rows with live[i] true
// (nil = all rows) for every query.
func changeTruth(d changeData, vec []float32, live []bool, k int) [][]int64 {
	out := make([][]int64, d.q)
	for i := 0; i < d.q; i++ {
		query := d.qs[i*d.dim : (i+1)*d.dim]
		tk := distance.NewTopK(k)
		for r := 0; r < d.n; r++ {
			if live == nil || live[r] {
				tk.Push(int64(r), distance.Dot(query, vec[r*d.dim:(r+1)*d.dim]))
			}
		}
		out[i], _ = tk.Results()
	}
	return out
}

// changeRecall runs every query and returns recall@k against truth. It fails
// the test if a result is a deleted row (dead[id] true).
func changeRecall(t *testing.T, d changeData, search func([]float32, int) ([]int64, []float32), truth [][]int64, dead []bool, k int) float64 {
	t.Helper()
	hits := 0
	for i := 0; i < d.q; i++ {
		ids, _ := search(d.qs[i*d.dim:(i+1)*d.dim], k)
		set := map[int64]bool{}
		for _, id := range truth[i] {
			set[id] = true
		}
		for _, id := range ids {
			if id >= 0 && dead != nil && dead[id] {
				t.Fatalf("query %d returned deleted row %d", i, id)
			}
			if set[id] {
				hits++
			}
		}
	}
	return float64(hits) / float64(d.q*k)
}

func del30(t *testing.T, n int) []bool {
	t.Helper()
	m, err := npy.ReadBool(filepath.Join(devDir(), "delete_del30.npy"))
	if err != nil {
		t.Fatal(err)
	}
	return m[:n]
}

func notMask(m []bool) []bool {
	out := make([]bool, len(m))
	for i, b := range m {
		out[i] = !b
	}
	return out
}

// TestDeleteDel30 checks that no deleted row is returned after del30 and that
// recall against the remaining-rows truth is within 0.03 of the undeleted
// recall (1.0 for flat). Then Compact (rebuild): recall within 0.01 of a fresh
// build on the remaining rows.
func TestDeleteDel30(t *testing.T) {
	const k = 10
	d := loadChangeData(t)
	ix, err := Build(d.vec, d.n, d.dim, map[string]any{}, runtime.NumCPU(), 42)
	if err != nil {
		t.Fatal(err)
	}
	search := func(ix *Index) func([]float32, int) ([]int64, []float32) {
		return func(q []float32, k int) ([]int64, []float32) { return Search(ix, q, k, map[string]any{}) }
	}
	base := changeRecall(t, d, search(ix), changeTruth(d, d.vec, nil, k), nil, k)

	dead := del30(t, d.n)
	live := notMask(dead)
	rows, err := Delete(ix, dead)
	if err != nil {
		t.Fatal(err)
	}
	truth := changeTruth(d, d.vec, live, k)
	after := changeRecall(t, d, search(ix), truth, dead, k)
	t.Logf("recall@10: undeleted %.4f, del30 (%d rows) %.4f", base, rows, after)
	if after < base-0.03 || (1.0 > 0 && after < 1.0) {
		t.Errorf("del30 recall %.4f, undeleted %.4f: more than 0.03 lower", after, base)
	}

	before := IndexBytes(ix)
	cix, err := Compact(ix, "rebuild")
	if err != nil {
		t.Fatal(err)
	}
	compacted := changeRecall(t, d, search(cix), truth, dead, k)

	// Fresh build on the remaining rows; node j is row liveIDs[j].
	var liveIDs []int64
	var liveVec []float32
	for r := 0; r < d.n; r++ {
		if live[r] {
			liveIDs = append(liveIDs, int64(r))
			liveVec = append(liveVec, d.vec[r*d.dim:(r+1)*d.dim]...)
		}
	}
	fix, err := Build(liveVec, len(liveIDs), d.dim, map[string]any{}, runtime.NumCPU(), 42)
	if err != nil {
		t.Fatal(err)
	}
	fresh := changeRecall(t, d, func(q []float32, k int) ([]int64, []float32) {
		ids, sc := Search(fix, q, k, map[string]any{})
		for j, id := range ids {
			if id >= 0 {
				ids[j] = liveIDs[id]
			}
		}
		return ids, sc
	}, truth, dead, k)
	t.Logf("recall@10 after compact %.4f, fresh build %.4f; index bytes %d -> %d", compacted, fresh, before, IndexBytes(cix))
	if compacted < fresh-0.01 || compacted > fresh+0.01 {
		t.Errorf("compacted recall %.4f, fresh build %.4f: differ by more than 0.01", compacted, fresh)
	}
	if true && IndexBytes(cix) >= before {
		t.Errorf("index bytes after compact %d >= before %d", IndexBytes(cix), before)
	}
}

// TestUpdateUpd10 checks that, after upd10, a query equal to the new vector
// of each of 100 sampled updated rows returns that row as the top-1.
func TestUpdateUpd10(t *testing.T) {
	d := loadChangeData(t)
	ix, err := Build(d.vec, d.n, d.dim, map[string]any{}, runtime.NumCPU(), 42)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := npy.ReadInt64Vec(filepath.Join(devDir(), "update_upd10_ids.npy"))
	if err != nil {
		t.Fatal(err)
	}
	vecs, _, _, err := npy.ReadFloat32(filepath.Join(devDir(), "update_upd10_vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	var uids []int64
	var uvec []float32
	for j, id := range ids {
		if id < int64(d.n) {
			uids = append(uids, id)
			uvec = append(uvec, vecs[j*d.dim:(j+1)*d.dim]...)
		}
	}
	rows, err := Update(ix, uids, uvec)
	if err != nil {
		t.Fatal(err)
	}
	if rows != len(uids) {
		t.Fatalf("Update returned %d rows, want %d", rows, len(uids))
	}
	step := max(len(uids)/100, 1)
	checked := 0
	for j := 0; j < len(uids) && checked < 100; j += step {
		q := uvec[j*d.dim : (j+1)*d.dim]
		got, _ := Search(ix, q, 10, map[string]any{})
		if got[0] != uids[j] {
			t.Errorf("updated row %d: top-1 is %d", uids[j], got[0])
		}
		checked++
	}
	t.Logf("%d rows updated, %d checked", len(uids), checked)
	truth := changeTruth(d, d.vec, nil, 10)
	r := changeRecall(t, d, func(q []float32, k int) ([]int64, []float32) { return Search(ix, q, k, map[string]any{}) }, truth, nil, 10)
	t.Logf("recall@10 after upd10 against the updated truth: %.4f", r)
}
