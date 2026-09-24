package pq

import (
	"bytes"
	"path/filepath"
	"runtime"
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

// TestRecallDev builds with defaults for each metric and checks the floor,
// the rerank gain, the code length, and the l2 score sign.
func TestRecallDev(t *testing.T) {
	d := loadDev(t)
	threads := runtime.NumCPU()
	for _, metric := range []string{"ip", "l2"} {
		ix, err := Build(d.vec, d.n, d.dim, map[string]any{"metric": metric}, threads, 42)
		if err != nil {
			t.Fatal(err)
		}
		if len(ix.Codes()) != d.n*48 {
			t.Errorf("%s: codes length %d, want %d", metric, len(ix.Codes()), d.n*48)
		}
		r0 := recall(t, d, ix, map[string]any{"rerank": 0}, metric == "l2")
		r100 := recall(t, d, ix, map[string]any{"rerank": 100}, metric == "l2")
		t.Logf("metric=%s train_s=%.2f add_s=%.2f recall@10 rerank=0: %.4f rerank=100: %.4f",
			metric, ix.TrainSeconds(), ix.AddSeconds(), r0, r100)
		if r0 < 0.50 {
			t.Errorf("%s: recall@10 = %v, want >= 0.50", metric, r0)
		}
		if r100 < r0 {
			t.Errorf("%s: rerank=100 recall %v < rerank=0 recall %v", metric, r100, r0)
		}
	}
}

// TestDeterminism checks that the same seed gives identical codes, and that
// 1 and 4 threads give identical codes. It uses 5k rows to stay fast.
func TestDeterminism(t *testing.T) {
	d := loadDev(t)
	n := 5000
	vec := d.vec[:n*d.dim]
	p := map[string]any{"train_size": 5000, "iters": 5}
	for _, metric := range []string{"ip", "l2"} {
		p["metric"] = metric
		a, err := Build(vec, n, d.dim, p, 4, 7)
		if err != nil {
			t.Fatal(err)
		}
		b, err := Build(vec, n, d.dim, p, 4, 7)
		if err != nil {
			t.Fatal(err)
		}
		c, err := Build(vec, n, d.dim, p, 1, 7)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a.Codes(), b.Codes()) {
			t.Errorf("%s: same seed gives different codes", metric)
		}
		if !bytes.Equal(a.Codes(), c.Codes()) {
			t.Errorf("%s: 1 and 4 threads give different codes", metric)
		}
	}
}

// TestBadParams checks that Build rejects nbits != 8 and an unknown metric.
func TestBadParams(t *testing.T) {
	vec := make([]float32, 300*48)
	if _, err := Build(vec, 300, 48, map[string]any{"nbits": 4}, 1, 1); err == nil {
		t.Error("nbits=4 accepted")
	}
	if _, err := Build(vec, 300, 48, map[string]any{"metric": "cos"}, 1, 1); err == nil {
		t.Error("metric=cos accepted")
	}
}
