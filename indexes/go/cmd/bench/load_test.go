package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"testing"

	"vro/indexes/go/distance"
	"vro/indexes/go/npy"
)

// Load-run tests (CONTRACT.md section 12.5) on the first 20,000 dev rows.
// Under -race the loops are shorter (raceEnabled), because the race detector
// slows every memory access.

const loadRows = 20000

type loadDoc struct {
	Extra    map[string]any `json:"extra"`
	Searches []struct {
		SearchParams map[string]any `json:"search_params"`
		IDs          [][]int64      `json:"ids"`
		LatencyMS    []float64      `json:"latency_ms"`
		QPS          float64        `json:"qps"`
		Extra        map[string]any `json:"extra"`
	} `json:"searches"`
}

func runBench(t *testing.T, args ...string) loadDoc {
	t.Helper()
	out := filepath.Join(t.TempDir(), "r.json")
	data := filepath.Join(repoRoot(), "data", "processed", "dev")
	base := []string{"--index", "hnsw", "--data", data, "--out", out, "--limit", "20000", "--warmup", "10"}
	if err := run(append(base, args...)); err != nil {
		t.Fatalf("bench %v: %v", args, err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc loadDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

var (
	truthOnce sync.Once
	truth20k  [][]int64
)

// bruteTruth returns the exact top-10 of each query over the first 20,000 rows.
func bruteTruth(t *testing.T) [][]int64 {
	truthOnce.Do(func() {
		dir := filepath.Join(repoRoot(), "data", "processed", "dev")
		vec, _, dim, err := npy.ReadFloat32(filepath.Join(dir, "vectors.npy"))
		if err != nil {
			t.Fatal(err)
		}
		qs, q, _, err := npy.ReadFloat32(filepath.Join(dir, "queries.npy"))
		if err != nil {
			t.Fatal(err)
		}
		truth20k = make([][]int64, q)
		var wg sync.WaitGroup
		workers := runtime.NumCPU()
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				for i := w; i < q; i += workers {
					tk := distance.NewTopK(10)
					qv := qs[i*dim : (i+1)*dim]
					for r := 0; r < loadRows; r++ {
						tk.Push(int64(r), distance.Dot(qv, vec[r*dim:(r+1)*dim]))
					}
					truth20k[i], _ = tk.Results()
				}
			}(w)
		}
		wg.Wait()
	})
	if truth20k == nil {
		t.Fatal("no ground truth")
	}
	return truth20k
}

func recallOf(ids [][]int64, truth [][]int64) float64 {
	hits := 0
	for i, row := range ids {
		for _, id := range row[:10] {
			if slices.Contains(truth[i], id) {
				hits++
			}
		}
	}
	return float64(hits) / float64(len(ids)*10)
}

func num(t *testing.T, m map[string]any, key string) float64 {
	t.Helper()
	v, ok := m[key].(float64)
	if !ok {
		t.Fatalf("extra has no numeric key %q: %v", key, m)
	}
	return v
}

// TestLoadClients: --clients 8 --duration 5 gives more qps than 1 client for
// 5 s, zero errors, and first-pass recall@10 >= 0.95 at ef=64.
func TestLoadClients(t *testing.T) {
	dur, clients := "5", "8"
	if raceEnabled {
		dur, clients = "2", "4"
	}
	truth := bruteTruth(t)
	one := runBench(t, "--clients", "1", "--duration", dur, "--search", "ef=64")
	many := runBench(t, "--clients", clients, "--duration", dur, "--search", "ef=64")
	for name, doc := range map[string]loadDoc{"1 client": one, clients + " clients": many} {
		if len(doc.Searches) != 1 {
			t.Fatalf("%s: %d searches, want 1", name, len(doc.Searches))
		}
		s := doc.Searches[0]
		for _, key := range []string{"errors", "cpu_pct", "clients", "duration_s", "queries_done"} {
			num(t, s.Extra, key)
		}
		if e := num(t, s.Extra, "errors"); e != 0 {
			t.Errorf("%s: %v errors", name, e)
		}
		if got, want := len(s.LatencyMS), int(num(t, s.Extra, "queries_done")); got != want {
			t.Errorf("%s: %d latencies, %d queries done", name, got, want)
		}
		if len(s.IDs) != 1000 {
			t.Fatalf("%s: %d id rows", name, len(s.IDs))
		}
		r := recallOf(s.IDs, truth)
		t.Logf("%s: qps %.0f, cpu_pct %.0f, queries %v, first-pass recall@10 %.4f",
			name, s.QPS, num(t, s.Extra, "cpu_pct"), s.Extra["queries_done"], r)
		if r < 0.95 {
			t.Errorf("%s: first-pass recall %.4f < 0.95", name, r)
		}
	}
	if raceEnabled {
		t.Log("-race: qps comparison skipped (the race detector serializes the workers)")
		return
	}
	if many.Searches[0].QPS <= one.Searches[0].QPS {
		t.Errorf("%s clients qps %.0f <= 1 client qps %.0f", clients, many.Searches[0].QPS, one.Searches[0].QPS)
	}
}

// TestLoadInsert: --clients 4 --duration 10 --insert-rate 2000 with the build
// on 18,000 rows: zero errors, 2,000 inserted rows, and after-inserts recall
// within 0.01 of a static build on 20,000 rows.
func TestLoadInsert(t *testing.T) {
	dur, clients := "10", "4"
	if raceEnabled {
		dur, clients = "5", "2"
	}
	truth := bruteTruth(t)
	static := runBench(t, "--search", "ef=64")
	doc := runBench(t, "--clients", clients, "--duration", dur, "--insert-rate", "2000", "--search", "ef=64")
	if len(doc.Searches) != 2 {
		t.Fatalf("%d searches, want 2 (load run, after_inserts)", len(doc.Searches))
	}
	load, after := doc.Searches[0], doc.Searches[1]
	if e := num(t, load.Extra, "errors"); e != 0 {
		t.Errorf("load run: %v errors", e)
	}
	if after.SearchParams["phase"] != "after_inserts" {
		t.Errorf("last search phase = %v", after.SearchParams["phase"])
	}
	for _, m := range []map[string]any{doc.Extra, after.Extra} {
		if got := num(t, m, "inserted_rows"); got != 2000 {
			t.Errorf("inserted_rows = %v, want 2000 (the insert tail adds what the loop left)", got)
		}
		num(t, m, "insert_p50_ms")
		num(t, m, "insert_tail_s")
		if d := num(t, m, "inserted_during_loop"); d < 0 || d > 2000 {
			t.Errorf("inserted_during_loop = %v", d)
		}
	}
	t.Logf("inserted during loop %v, tail %v s", doc.Extra["inserted_during_loop"], doc.Extra["insert_tail_s"])
	if got := num(t, doc.Extra, "build_rows"); got != 18000 {
		t.Errorf("build_rows = %v, want 18000", got)
	}
	if raceEnabled {
		t.Log("-race: recall comparison skipped")
		return
	}
	rs, ra, rl := recallOf(static.Searches[0].IDs, truth), recallOf(after.IDs, truth), recallOf(load.IDs, truth)
	t.Logf("static 20k recall %.4f, after-inserts %.4f, load first pass %.4f; load qps %.0f, insert_p50_ms %.2f, extra %v",
		rs, ra, rl, load.QPS, num(t, doc.Extra, "insert_p50_ms"), doc.Extra)
	if ra < rs-0.01 || ra > rs+0.01 {
		t.Errorf("after-inserts recall %.4f not within 0.01 of static %.4f", ra, rs)
	}
}

// TestLoadOtherIndexRejected: --clients > 1 on flat is a usage error (exit 2).
func TestLoadOtherIndexRejected(t *testing.T) {
	_, err := parseFlags([]string{"--index", "flat", "--data", "d", "--out", "o", "--clients", "4"})
	if _, ok := err.(usageError); !ok {
		t.Fatalf("err = %v, want usageError", err)
	}
}
