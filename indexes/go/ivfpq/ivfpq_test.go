package ivfpq

import (
	"bytes"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"vro/indexes/go/npy"
)

type devData struct {
	vec, qs   []float32
	n, dim, q int
	gt        []int64
	gtCols    int
}

var dev *devData

func loadDev(t *testing.T) *devData {
	t.Helper()
	if dev != nil {
		return dev
	}
	_, file, _, _ := runtime.Caller(0)
	dir := filepath.Join(filepath.Dir(file), "..", "..", "..", "data", "processed", "dev")
	d := &devData{}
	var err error
	if d.vec, d.n, d.dim, err = npy.ReadFloat32(filepath.Join(dir, "vectors.npy")); err != nil {
		t.Fatal(err)
	}
	if d.qs, d.q, _, err = npy.ReadFloat32(filepath.Join(dir, "queries.npy")); err != nil {
		t.Fatal(err)
	}
	if d.gt, _, d.gtCols, err = npy.ReadInt64(filepath.Join(dir, "ground_truth.npy")); err != nil {
		t.Fatal(err)
	}
	dev = d
	return d
}

// recall returns recall@10 over all queries and checks the l2 score sign.
func recall(t *testing.T, d *devData, ix *Index, params map[string]any, l2 bool) float64 {
	const k = 10
	hits := 0
	for i := 0; i < d.q; i++ {
		ids, scores := Search(ix, d.qs[i*d.dim:(i+1)*d.dim], k, params)
		if l2 {
			for _, s := range scores {
				if s > 0 {
					t.Fatalf("query %d: l2 score %v > 0", i, s)
				}
			}
		}
		set := make(map[int64]bool, k)
		for _, id := range d.gt[i*d.gtCols : i*d.gtCols+k] {
			set[id] = true
		}
		for _, id := range ids {
			if set[id] {
				hits++
			}
		}
	}
	return float64(hits) / float64(d.q*k)
}

// checkCSR checks that offsets end at n and every row appears exactly once.
func checkCSR(t *testing.T, ix *Index, n int) {
	ids, off := ix.Lists()
	if len(off) != ix.nlist+1 || off[0] != 0 || int(off[len(off)-1]) != n {
		t.Fatalf("bad offsets: len %d, first %d, last %d, want last %d", len(off), off[0], off[len(off)-1], n)
	}
	for c := 0; c < ix.nlist; c++ {
		if off[c+1] < off[c] {
			t.Fatalf("offsets decrease at list %d", c)
		}
	}
	seen := make([]bool, n)
	for _, id := range ids {
		if id < 0 || int(id) >= n || seen[id] {
			t.Fatalf("row %d missing, repeated, or out of range", id)
		}
		seen[id] = true
	}
	if len(ids) != n || len(ix.Codes()) != n*ix.m {
		t.Fatalf("ids %d codes %d, want %d and %d", len(ids), len(ix.Codes()), n, n*ix.m)
	}
}

// TestRecallDev builds at defaults for each metric on the dev set.
func TestRecallDev(t *testing.T) {
	d := loadDev(t)
	threads := runtime.NumCPU()
	for _, metric := range []string{"ip", "l2"} {
		ix, err := Build(d.vec, d.n, d.dim, map[string]any{"metric": metric}, threads, 42)
		if err != nil {
			t.Fatal(err)
		}
		checkCSR(t, ix, d.n)
		l2 := metric == "l2"
		r8 := recall(t, d, ix, map[string]any{"nprobe": 8, "rerank": 0}, l2)
		r8r := recall(t, d, ix, map[string]any{"nprobe": 8, "rerank": 100}, l2)
		r32 := recall(t, d, ix, map[string]any{"nprobe": 32, "rerank": 0}, l2)
		t.Logf("metric=%s train_s=%.2f add_s=%.2f recall@10 nprobe=8: %.4f nprobe=8,rerank=100: %.4f nprobe=32: %.4f",
			metric, ix.TrainSeconds(), ix.AddSeconds(), r8, r8r, r32)
		if r8 < 0.45 {
			t.Errorf("%s: recall@10 = %v, want >= 0.45", metric, r8)
		}
		if r8r < r8 {
			t.Errorf("%s: rerank=100 recall %v < rerank=0 recall %v", metric, r8r, r8)
		}
		if r32 < r8 {
			t.Errorf("%s: nprobe=32 recall %v < nprobe=8 recall %v", metric, r32, r8)
		}
	}
}

// TestDeterminism: 1 vs 4 threads, same seed, give identical codes and lists (20k rows).
func TestDeterminism(t *testing.T) {
	d := loadDev(t)
	n := 20000
	vec := d.vec[:n*d.dim]
	for _, metric := range []string{"ip", "l2"} {
		p := map[string]any{"nlist": 64, "train_size": 20000, "iters": 3, "metric": metric}
		a, err := Build(vec, n, d.dim, p, 4, 7)
		if err != nil {
			t.Fatal(err)
		}
		c, err := Build(vec, n, d.dim, p, 1, 7)
		if err != nil {
			t.Fatal(err)
		}
		checkCSR(t, a, n)
		for name, o := range map[string]*Index{"1 thread": c} {
			ia, oa := a.Lists()
			io, oo := o.Lists()
			if !bytes.Equal(a.Codes(), o.Codes()) || !slices.Equal(ia, io) || !slices.Equal(oa, oo) {
				t.Errorf("%s: %s gives different codes or lists", metric, name)
			}
		}
		ids, scores := Search(a, d.qs[:d.dim], 10, map[string]any{"nprobe": 2})
		ids2, scores2 := Search(c, d.qs[:d.dim], 10, map[string]any{"nprobe": 2})
		if !slices.Equal(ids, ids2) || !slices.Equal(scores, scores2) {
			t.Errorf("%s: search results differ between 4 and 1 threads", metric)
		}
	}
}

// TestPadding: a probe with fewer than k rows pads with -1 and -inf.
func TestPadding(t *testing.T) {
	d := loadDev(t)
	n := 2000
	ix, err := Build(d.vec[:n*d.dim], n, d.dim, map[string]any{"nlist": 256, "train_size": 2000, "iters": 3}, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	ids, scores := Search(ix, d.qs[:d.dim], 1000, map[string]any{"nprobe": 1})
	last := len(ids) - 1
	if ids[last] != -1 || scores[last] > -1e30 {
		t.Errorf("last result %d %v, want -1 -inf", ids[last], scores[last])
	}
}

// TestBadParams checks that Build rejects nbits != 8 and an unknown metric.
func TestBadParams(t *testing.T) {
	vec := make([]float32, 300*48)
	if _, err := Build(vec, 300, 48, map[string]any{"nbits": 4, "nlist": 4}, 1, 1); err == nil {
		t.Error("nbits=4 accepted")
	}
	if _, err := Build(vec, 300, 48, map[string]any{"metric": "cos", "nlist": 4}, 1, 1); err == nil {
		t.Error("metric=cos accepted")
	}
}
