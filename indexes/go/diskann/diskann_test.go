package diskann

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vro/indexes/go/distance"
	"vro/indexes/go/npy"
)

// testRows is the corpus size of the tests (CONTRACT.md section 9).
const testRows = 20000

type testData struct {
	qs     []float32
	q, dim int
	truth  [][]int64 // brute-force top 10 on the first testRows rows
	ix     *Index
	freed  atomic.Bool // set by the finalizer of the corpus copy
	err    error
}

var (
	once   sync.Once
	td     testData
	tmpDir string
)

func TestMain(m *testing.M) {
	var err error
	tmpDir, err = os.MkdirTemp("", "diskann-test")
	if err != nil {
		panic(err)
	}
	code := m.Run()
	if td.ix != nil {
		td.ix.Close()
	}
	os.RemoveAll(tmpDir)
	os.Exit(code)
}

func devDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "data", "processed", "dev")
}

// bruteTop10 returns the exact top 10 ids per query by dot product.
func bruteTop10(vec []float32, n, dim int, qs []float32, q int) [][]int64 {
	out := make([][]int64, q)
	parallelFor(q, runtime.NumCPU(), func(lo, hi int) {
		tk := distance.NewTopK(10)
		for i := lo; i < hi; i++ {
			tk.Reset()
			qv := qs[i*dim : (i+1)*dim]
			for r := 0; r < n; r++ {
				tk.Push(int64(r), distance.Dot(qv, vec[r*dim:(r+1)*dim]))
			}
			out[i], _ = tk.Results()
		}
	})
	return out
}

// buildIndex builds on a private copy of rows, so the test can check that
// the copy is garbage collected once Build returns.
func buildIndex(vec []float32, dim int, params map[string]any, name string, freed *atomic.Bool) (*Index, error) {
	corpus := make([]float32, testRows*dim)
	copy(corpus, vec[:testRows*dim])
	if freed != nil {
		runtime.SetFinalizer(&corpus[0], func(*float32) { freed.Store(true) })
	}
	OutPath = filepath.Join(tmpDir, name+".json")
	return Build(corpus, testRows, dim, params, runtime.NumCPU(), 42)
}

func load(t *testing.T) *testData {
	t.Helper()
	once.Do(func() {
		d := &td
		dir := devDir()
		vec, _, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
		if err != nil {
			d.err = err
			return
		}
		if d.qs, d.q, d.dim, d.err = npy.ReadFloat32(filepath.Join(dir, "queries.npy")); d.err != nil {
			return
		}
		d.truth = bruteTop10(vec, testRows, dim, d.qs, d.q)
		start := time.Now()
		d.ix, d.err = buildIndex(vec, dim, map[string]any{}, "ip", &d.freed)
		if d.err == nil {
			t.Logf("build ip on %d rows: train %.1f s, add %.1f s (wall %.1f s), mean out-degree %.2f, entry %d",
				testRows, d.ix.TrainSeconds(), d.ix.AddSeconds(), time.Since(start).Seconds(), d.ix.meanDeg, d.ix.entry)
		}
	})
	if td.err != nil {
		t.Fatal(td.err)
	}
	return &td
}

func searchParams(l int, io string) map[string]any {
	return map[string]any{"l": int64(l), "beam": int64(4), "rerank": int64(100), "io": io}
}

// runAll runs every query and returns ids and per-query latencies.
func runAll(ix *Index, d *testData, p map[string]any) ([][]int64, []float64) {
	ids := make([][]int64, d.q)
	lat := make([]float64, d.q)
	for i := 0; i < d.q; i++ {
		start := time.Now()
		ids[i], _ = Search(ix, d.qs[i*d.dim:(i+1)*d.dim], 10, p)
		lat[i] = float64(time.Since(start).Nanoseconds()) / 1e6
	}
	return ids, lat
}

func recall(truth, got [][]int64) float64 {
	hit := 0
	for i := range truth {
		set := map[int64]bool{}
		for _, id := range truth[i] {
			set[id] = true
		}
		for _, id := range got[i] {
			if set[id] {
				hit++
			}
		}
	}
	return float64(hit) / float64(10*len(truth))
}

func p50(x []float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func TestRecallDefaults(t *testing.T) {
	d := load(t)
	ids, _ := runAll(d.ix, d, searchParams(100, "mmap"))
	rc := recall(d.truth, ids)
	t.Logf("recall@10 ip l=100 io=mmap: %.4f", rc)
	if rc < 0.90 {
		t.Fatalf("recall@10 %.4f < 0.90", rc)
	}
}

func TestRecallGrowsWithL(t *testing.T) {
	d := load(t)
	var rc [3]float64
	for i, l := range []int{50, 100, 200} {
		ids, _ := runAll(d.ix, d, searchParams(l, "mmap"))
		rc[i] = recall(d.truth, ids)
	}
	t.Logf("recall@10 l=50 %.4f, l=100 %.4f, l=200 %.4f", rc[0], rc[1], rc[2])
	if !(rc[2] >= rc[1] && rc[1] >= rc[0]) {
		t.Fatalf("recall is not monotonic in l: %v", rc)
	}
}

func TestIOModes(t *testing.T) {
	d := load(t)
	pm, pn := searchParams(100, "mmap"), searchParams(100, "nocache")
	runAll(d.ix, d, pm) // warm the map
	idsM, latM := runAll(d.ix, d, pm)
	before := d.ix.SearchCounters()["disk_reads"]
	idsN, latN := runAll(d.ix, d, pn)
	reads := (d.ix.SearchCounters()["disk_reads"] - before) / float64(d.q)
	for i := range idsM {
		for j := range idsM[i] {
			if idsM[i][j] != idsN[i][j] {
				t.Fatalf("query %d: mmap ids %v, nocache ids %v", i, idsM[i], idsN[i])
			}
		}
	}
	t.Logf("p50 mmap %.3f ms, nocache %.3f ms, disk_reads/query %.1f", p50(latM), p50(latN), reads)
	if reads <= 0 {
		t.Fatalf("disk_reads = %v", reads)
	}
	if p50(latN) < 2*p50(latM) {
		t.Fatalf("nocache p50 %.3f ms is not at least 2x mmap p50 %.3f ms", p50(latN), p50(latM))
	}
}

func TestGraphAndFile(t *testing.T) {
	d := load(t)
	deg, err := d.ix.OutDegrees()
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range deg {
		if c < 1 || c > d.ix.r {
			t.Fatalf("node %d has %d out-edges", i, c)
		}
	}
	st, err := os.Stat(d.ix.Path())
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(testRows) * 4096; st.Size() != want {
		t.Fatalf("file size %d, want %d", st.Size(), want)
	}
}

func TestCorpusReleased(t *testing.T) {
	d := load(t)
	for i := 0; i < 20 && !d.freed.Load(); i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	if !d.freed.Load() {
		t.Fatal("the corpus array was not freed after Build: the index still references it")
	}
	runtime.KeepAlive(d.ix)
}

func TestRecallL2(t *testing.T) {
	d := load(t)
	vec, _, dim, err := npy.ReadFloat32(filepath.Join(devDir(), "vectors.npy"))
	if err != nil {
		t.Fatal(err)
	}
	ix, err := buildIndex(vec, dim, map[string]any{"metric": "l2"}, "l2", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	ids, _ := runAll(ix, d, searchParams(100, "mmap"))
	rc := recall(d.truth, ids)
	t.Logf("recall@10 l2 l=100 io=mmap: %.4f", rc)
	if rc < 0.90 {
		t.Fatalf("recall@10 %.4f < 0.90", rc)
	}
}
