package main

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestChangeDeleteJSON runs bench as a subprocess with --delete del30 on hnsw
// and checks search_params.deleted / compacted and extra.deleted_rows
// (CONTRACT.md section 13.2).
func TestChangeDeleteJSON(t *testing.T) {
	out := filepath.Join(t.TempDir(), "hnsw-del30.json")
	data := filepath.Join(repoRoot(), "data", "processed", "dev")
	cmd := exec.Command(os.Args[0], "--index", "hnsw", "--data", data, "--out", out, "--limit", "20000",
		"--warmup", "10", "--delete", "del30", "--search", "ef=64")
	cmd.Env = append(os.Environ(), "BENCH_SUBPROCESS=1")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("bench failed: %v\n%s", err, b)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Searches []struct {
			SearchParams map[string]any `json:"search_params"`
			IDs          [][]int64      `json:"ids"`
		} `json:"searches"`
		Extra map[string]any `json:"extra"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if len(doc.Searches) != 1 {
		t.Fatalf("%d searches, want 1", len(doc.Searches))
	}
	sp := doc.Searches[0].SearchParams
	if sp["deleted"] != "del30" || sp["compacted"] != float64(0) {
		t.Errorf("search_params = %v, want deleted=del30 compacted=0", sp)
	}
	if _, ok := sp["updated"]; ok {
		t.Errorf("search_params has updated: %v", sp)
	}
	rows, ok := doc.Extra["deleted_rows"].(float64)
	if !ok || rows <= 0 {
		t.Errorf("extra.deleted_rows = %v", doc.Extra["deleted_rows"])
	}
	if _, ok := doc.Extra["delete_s"]; !ok {
		t.Errorf("no extra.delete_s")
	}
	t.Logf("deleted_rows=%v delete_s=%v", rows, doc.Extra["delete_s"])
}

// TestChangeFlagsUsage checks the flag rules: other indexes, bad names and
// combinations exit 2 (usage error).
func TestChangeFlagsUsage(t *testing.T) {
	base := []string{"--data", "d", "--out", "o.json"}
	cases := [][]string{
		{"--index", "pq", "--delete", "del10"},
		{"--index", "diskann", "--update", "upd10"},
		{"--index", "flat", "--delete", "del20"},
		{"--index", "flat", "--update", "upd20"},
		{"--index", "flat", "--delete", "del10", "--update", "upd10"},
		{"--index", "flat", "--compact"},
		{"--index", "hnsw", "--delete", "del10", "--compact", "--compact-mode", "vacuum"},
		{"--index", "hnsw", "--delete", "del10", "--clients", "4"},
	}
	for _, c := range cases {
		_, err := parseFlags(append(append([]string{}, base...), c...))
		var ue usageError
		if !errors.As(err, &ue) {
			t.Errorf("%v: err = %v, want usageError", c, err)
		}
	}
	if _, err := parseFlags(append(base, "--index", "ivf", "--delete", "del50", "--compact")); err != nil {
		t.Errorf("valid flags rejected: %v", err)
	}
}
