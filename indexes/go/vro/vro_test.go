package vro

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestVroWriteRead writes three sections and reads them back: the header
// parses, every offset is a multiple of 64, and the data round-trips.
func TestVroWriteRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.vro")
	f32 := []float32{1, -2.5, 3.25, 0, 7, 8}
	i32 := []int32{-1, 0, 5, 2147483647, -2147483648}
	u8 := []byte{1, 2, 3}
	h := Header{Index: "flat", N: 2, Dim: 3, BuildParams: map[string]any{"m": int64(16)}, Seed: 42, ContractVersion: 1, Language: "go"}
	size, err := Write(path, h, []Array{
		{Name: "vectors", Shape: []int64{2, 3}, F32: f32},
		{Name: "ids", Shape: []int64{5}, I32: i32},
		{Name: "tombstones", Shape: []int64{3}, U8: u8},
	})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	if int64(len(raw)) != size || string(raw[:8]) != Magic {
		t.Fatalf("size %d / %d, magic %q", len(raw), size, raw[:8])
	}
	hl := binary.LittleEndian.Uint32(raw[8:12])
	for _, c := range raw[12 : 12+hl] {
		if c > 0x7f {
			t.Fatal("header is not ASCII")
		}
	}
	v, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	if len(v.Header.Sections) != 3 {
		t.Fatalf("%d sections", len(v.Header.Sections))
	}
	first := v.Header.Sections[0].Offset
	if first != (12+int64(hl)+63)/64*64 {
		t.Errorf("first section at %d, header ends at %d", first, 12+hl)
	}
	for _, s := range v.Header.Sections {
		if s.Offset%64 != 0 {
			t.Errorf("section %s at offset %d", s.Name, s.Offset)
		}
	}
	a, sh, err := v.F32("vectors")
	if err != nil || !reflect.DeepEqual(a, f32) || !reflect.DeepEqual(sh, []int64{2, 3}) {
		t.Errorf("vectors %v %v %v", a, sh, err)
	}
	b, _, err := v.I32("ids")
	if err != nil || !reflect.DeepEqual(b, i32) {
		t.Errorf("ids %v %v", b, err)
	}
	c, _, err := v.U8("tombstones")
	if err != nil || !reflect.DeepEqual(c, u8) {
		t.Errorf("tombstones %v %v", c, err)
	}
	if _, _, err := v.I32("vectors"); err == nil {
		t.Error("vectors read as int32")
	}
	// Refusals: index, dim, build_params.
	if err := v.Check("flat", 3, map[string]any{"m": 16}); err != nil {
		t.Errorf("matching check failed: %v", err)
	}
	for _, c := range []struct {
		index string
		dim   int
		p     map[string]any
	}{{"hnsw", 3, nil}, {"flat", 4, nil}, {"flat", 3, map[string]any{"m": 32}}, {"flat", 3, map[string]any{}}, {"flat", 3, map[string]any{"m": 16, "x": 1}}} {
		if err := v.Check(c.index, c.dim, c.p); err == nil {
			t.Errorf("check %v passed", c)
		}
	}
}

func TestVroBadMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.vro")
	if _, err := Write(path, Header{Index: "flat", N: 0, Dim: 1}, nil); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	raw[7] = '2'
	os.WriteFile(path, raw, 0o644)
	if _, err := Open(path); err == nil {
		t.Fatal("bad magic accepted")
	}
}
