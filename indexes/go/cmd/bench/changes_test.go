package main

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
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

// benchJSON runs bench as a subprocess and returns the output document.
func benchJSON(t *testing.T, args ...string) (map[string]any, [][]int64) {
	t.Helper()
	out := filepath.Join(t.TempDir(), "out.json")
	data := filepath.Join(repoRoot(), "data", "processed", "dev")
	cmd := exec.Command(os.Args[0], append([]string{"--data", data, "--out", out, "--warmup", "10"}, args...)...)
	cmd.Env = append(os.Environ(), "BENCH_SUBPROCESS=1")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("bench %v failed: %v\n%s", args, err, b)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	var ids struct {
		Searches []struct {
			IDs [][]int64 `json:"ids"`
		} `json:"searches"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	json.Unmarshal(raw, &ids)
	return doc, ids.Searches[0].IDs
}

// TestSaveLoadBench runs a build with --save, then a separate process with
// --load, for flat, ivf and hnsw on 20,000 rows: the IDs are identical
// (CONTRACT.md section 15.4). A conflicting --build value exits 2.
func TestSaveLoadBench(t *testing.T) {
	dir := t.TempDir()
	for _, c := range []struct{ index, build string }{{"flat", ""}, {"ivf", "nlist=256"}, {"hnsw", ""}} {
		path := filepath.Join(dir, c.index+".vro")
		args := []string{"--index", c.index, "--limit", "20000", "--save", path}
		if c.build != "" {
			args = append(args, "--build", c.build)
		}
		wdoc, wids := benchJSON(t, args...)
		ldoc, lids := benchJSON(t, "--index", c.index, "--load", path)
		if !reflect.DeepEqual(wids, lids) {
			t.Errorf("%s: ids differ between the write run and the load run", c.index)
		}
		we, le := wdoc["extra"].(map[string]any), ldoc["extra"].(map[string]any)
		lb := ldoc["build"].(map[string]any)
		if lb["train_s"] != 0.0 || lb["add_s"] != 0.0 || le["loaded_from"] != path || le["load_s"] == nil {
			t.Errorf("%s: load run build %v extra %v", c.index, lb, le)
		}
		if we["save_s"] == nil || we["file_bytes"] == nil {
			t.Errorf("%s: write run extra %v", c.index, we)
		}
		if !reflect.DeepEqual(wdoc["build_params"], ldoc["build_params"]) || ldoc["n"] != 20000.0 {
			t.Errorf("%s: build_params %v vs %v, n %v", c.index, wdoc["build_params"], ldoc["build_params"], ldoc["n"])
		}
		t.Logf("%s: save_s=%v file_bytes=%v load_s=%v", c.index, we["save_s"], we["file_bytes"], le["load_s"])
	}
	// A --build value that conflicts with the header exits 2.
	cmd := exec.Command(os.Args[0], "--index", "ivf", "--data", filepath.Join(repoRoot(), "data", "processed", "dev"),
		"--out", filepath.Join(dir, "x.json"), "--load", filepath.Join(dir, "ivf.vro"), "--build", "nlist=512")
	cmd.Env = append(os.Environ(), "BENCH_SUBPROCESS=1")
	err := cmd.Run()
	var ee *exec.ExitError
	if !errors.As(err, &ee) || ee.ExitCode() != 2 {
		t.Errorf("conflicting --build: err %v, want exit 2", err)
	}
}
