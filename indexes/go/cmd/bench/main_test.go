package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func repoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "..")
}

// TestFlatJSON runs a flat benchmark with --limit 20000 and checks the output
// against CONTRACT.md section 3.
func TestFlatJSON(t *testing.T) {
	out := filepath.Join(t.TempDir(), "flat.json")
	data := filepath.Join(repoRoot(), "data", "processed", "dev")
	err := run([]string{"--index", "flat", "--data", data, "--out", out, "--limit", "20000", "--warmup", "10"})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, key := range []string{"contract_version", "language", "index", "data_dir", "n", "dim", "q", "k",
		"threads", "seed", "build_params", "build", "searches", "machine", "extra"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("missing key %q", key)
		}
	}
	if doc["n"].(float64) != 20000 || doc["language"] != "go" || doc["contract_version"].(float64) != 1 {
		t.Errorf("bad header values: n=%v language=%v", doc["n"], doc["language"])
	}
	build := doc["build"].(map[string]any)
	for _, key := range []string{"train_s", "add_s", "total_s", "peak_rss_mb", "index_bytes"} {
		if _, ok := build[key]; !ok {
			t.Errorf("missing build key %q", key)
		}
	}
	for _, key := range []string{"os", "arch", "cpu", "cores"} {
		if _, ok := doc["machine"].(map[string]any)[key]; !ok {
			t.Errorf("missing machine key %q", key)
		}
	}
	searches := doc["searches"].([]any)
	if len(searches) != 1 {
		t.Fatalf("%d searches, want 1", len(searches))
	}
	s := searches[0].(map[string]any)
	for _, key := range []string{"search_params", "ids", "scores", "latency_ms", "total_s", "qps", "distance_computations", "extra"} {
		if _, ok := s[key]; !ok {
			t.Errorf("missing search key %q", key)
		}
	}
	ids, scores, lat := s["ids"].([]any), s["scores"].([]any), s["latency_ms"].([]any)
	if len(ids) != 1000 || len(scores) != 1000 || len(lat) != 1000 {
		t.Fatalf("rows: ids %d scores %d latency %d, want 1000", len(ids), len(scores), len(lat))
	}
	for i := range ids {
		if len(ids[i].([]any)) != 10 || len(scores[i].([]any)) != 10 {
			t.Fatalf("row %d does not have 10 results", i)
		}
	}
	if s["distance_computations"].(float64) != 20000 {
		t.Errorf("distance_computations = %v, want 20000", s["distance_computations"])
	}
}

func TestUnknownParamIsUsageError(t *testing.T) {
	_, err := resolveParams(map[string]any{"ef": int64(64)}, []string{"bogus=1"}, "search")
	if _, ok := err.(usageError); !ok {
		t.Fatalf("err = %v, want usageError", err)
	}
}

func TestScoreNull(t *testing.T) {
	b, _ := json.Marshal([]score{0.5, score(math.Inf(-1))})
	if string(b) != "[0.5,null]" {
		t.Fatalf("got %s", b)
	}
}
