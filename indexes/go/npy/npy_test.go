package npy

import (
	"path/filepath"
	"runtime"
	"testing"
)

// devDir returns data/processed/dev relative to this test file.
func devDir(t *testing.T) string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "data", "processed", "dev")
}

func TestQueries(t *testing.T) {
	data, rows, dim, err := ReadFloat32(filepath.Join(devDir(t), "queries.npy"))
	if err != nil {
		t.Fatal(err)
	}
	if rows != 1000 || dim != 384 {
		t.Fatalf("shape (%d, %d), want (1000, 384)", rows, dim)
	}
	want := []float32{-0.05522317439317703, -0.03818117082118988, 0.01416025310754776}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("row 0 col %d = %v, want %v", i, data[i], w)
		}
	}
}

func TestGroundTruthInt64(t *testing.T) {
	_, rows, cols, err := ReadInt64(filepath.Join(devDir(t), "ground_truth.npy"))
	if err != nil {
		t.Fatal(err)
	}
	if rows != 1000 || cols != 100 {
		t.Fatalf("shape (%d, %d), want (1000, 100)", rows, cols)
	}
}

func TestWrongDescr(t *testing.T) {
	if _, _, _, err := ReadInt64(filepath.Join(devDir(t), "queries.npy")); err == nil {
		t.Fatal("reading <f4 as <i8 did not fail")
	}
}
