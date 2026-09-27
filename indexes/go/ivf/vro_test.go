package ivf

// Phase 7 tests (CONTRACT.md section 15.4) on the first 20,000 dev rows: a
// saved and loaded index returns the same IDs and scores for all 1,000
// queries, at the default setting, after del30, after a compaction, and with
// filter=top10 on the loaded index. A file whose header dim was changed is
// refused.

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"vro/indexes/go/npy"
	"vro/indexes/go/vro"
)

const vroRows = 20000

var vroParams = map[string]any{"nlist": int64(256), "train_size": int64(20000), "iters": int64(20)}

// vroData reads the first 20,000 rows (a private copy) and the queries.
func vroData(t *testing.T) (vec, qs []float32, n, dim, q int) {
	t.Helper()
	dir := devDir()
	DataDir = dir
	all, n, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	n = min(n, vroRows)
	qs, q, _, err = npy.ReadFloat32(filepath.Join(dir, "queries.npy"))
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), all[:n*dim]...), qs, n, dim, q
}

// vroSame runs every query on a and b and fails on the first difference.
func vroSame(t *testing.T, what string, a, b *Index, qs []float32, dim, q int, p map[string]any) {
	t.Helper()
	for i := 0; i < q; i++ {
		query := qs[i*dim : (i+1)*dim]
		ai, as := Search(a, query, 10, p)
		bi, bs := Search(b, query, 10, p)
		if !reflect.DeepEqual(ai, bi) || !reflect.DeepEqual(as, bs) {
			t.Fatalf("%s: query %d: saved %v %v, loaded %v %v", what, i, ai, as, bi, bs)
		}
	}
}

// vroRoundTrip saves ix, loads the file back and compares the searches.
func vroRoundTrip(t *testing.T, what string, ix *Index, qs []float32, dim, q int) *Index {
	t.Helper()
	path := filepath.Join(t.TempDir(), "ivf.vro")
	size, err := Save(ix, path)
	if err != nil {
		t.Fatal(err)
	}
	lx, h, err := Load(path, dim, vroParams)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: file %d bytes, n=%d, sections %d", what, size, h.N, len(h.Sections))
	vroSame(t, what, ix, lx, qs, dim, q, map[string]any{})
	return lx
}

func TestVroRoundTrip(t *testing.T) {
	vec, qs, n, dim, q := vroData(t)
	ix, err := Build(vec, n, dim, vroParams, runtime.NumCPU(), 42)
	if err != nil {
		t.Fatal(err)
	}
	lx := vroRoundTrip(t, "default", ix, qs, dim, q)
	// Filter on the loaded index: the same as on the built index.
	vroSame(t, "filter=top10", ix, lx, qs, dim, q, map[string]any{"filter": "top10"})

	// del30, then save and load: the tombstones travel with the file.
	mask, err := npy.ReadBool(filepath.Join(devDir(), "delete_del30.npy"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Delete(ix, mask[:n]); err != nil {
		t.Fatal(err)
	}
	dx := vroRoundTrip(t, "del30", ix, qs, dim, q)
	vroSame(t, "del30 filter=top10", ix, dx, qs, dim, q, map[string]any{"filter": "top10"})

	// Phase 5 on the loaded index: a delete on the loaded index and on the
	// original gives the same results.
	if _, err := Delete(lx, mask[:n]); err != nil {
		t.Fatal(err)
	}
	vroSame(t, "delete after load", ix, lx, qs, dim, q, map[string]any{})

	// After Compact the file still holds all N rows.
	cx, err := Compact(ix, "rebuild")
	if err != nil {
		t.Fatal(err)
	}
	vroRoundTrip(t, "compact", cx, qs, dim, q)
}

func TestVroDimRefused(t *testing.T) {
	vec, _, n, dim, _ := vroData(t)
	n = 1000
	ix, err := Build(vec[:n*dim], n, dim, map[string]any{"nlist": 16}, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "ivf.vro")
	if _, err := Save(ix, path); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// The header parses and every section offset is a multiple of 64.
	f, err := vro.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range f.Header.Sections {
		if s.Offset%64 != 0 {
			t.Errorf("section %s at offset %d", s.Name, s.Offset)
		}
	}
	f.Close()
	// Change the dim in the header (same length), then load: refused.
	old := []byte(`"dim":384`)
	if !bytes.Contains(raw, old) {
		t.Fatalf("header has no %s", old)
	}
	bad := bytes.Replace(raw, old, []byte(`"dim":385`), 1)
	if err := os.WriteFile(path, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(path, dim, nil); err == nil {
		t.Fatal("a file with dim 385 loaded for dim 384")
	} else {
		t.Logf("refused: %v", err)
	}
	// A wrong index name is refused as well.
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := vro.Load(path, "flat", dim, nil); err == nil {
		t.Fatal("a ivf file loaded as flat")
	}
}
