// Package vro reads and writes the .vro index file (CONTRACT.md section 15.1).
//
// Layout, little-endian:
//
//	magic      8 bytes   "VROIDX01"
//	header_len uint32
//	header     ASCII JSON, header_len bytes
//	padding    zero bytes to the next multiple of 64
//	sections   raw arrays in the order of the header's "sections" list, each
//	           at its "offset" (a multiple of 64 from the file start)
//
// Dtypes: "f32" (float32), "int32", "u8". A reader takes every offset from the
// section table and never assumes one.
package vro

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// Magic is the first 8 bytes of every .vro file.
const Magic = "VROIDX01"

// Align is the alignment of the first section and of every section offset.
const Align = 64

// Section is one entry of the header's section table.
type Section struct {
	Name   string  `json:"name"`
	Dtype  string  `json:"dtype"`
	Shape  []int64 `json:"shape"`
	Offset int64   `json:"offset"`
	Bytes  int64   `json:"bytes"`
}

// Header is the JSON header. The field order is the order in CONTRACT 15.1.
type Header struct {
	Index           string         `json:"index"`
	N               int            `json:"n"`
	Dim             int            `json:"dim"`
	BuildParams     map[string]any `json:"build_params"`
	Seed            uint64         `json:"seed"`
	ContractVersion int            `json:"contract_version"`
	Language        string         `json:"language"`
	Sections        []Section      `json:"sections"`
}

// Array is one section to write. Exactly one of F32, I32, U8 is set.
type Array struct {
	Name  string
	Shape []int64
	F32   []float32
	I32   []int32
	U8    []byte
}

func (a Array) dtype() (string, int64) {
	switch {
	case a.F32 != nil:
		return "f32", int64(len(a.F32)) * 4
	case a.I32 != nil:
		return "int32", int64(len(a.I32)) * 4
	default:
		return "u8", int64(len(a.U8))
	}
}

func alignUp(x int64) int64 { return (x + Align - 1) / Align * Align }

// Write writes a .vro file. h.Sections is filled from arrays. It returns the
// file size in bytes.
func Write(path string, h Header, arrays []Array) (int64, error) {
	h.Sections = make([]Section, len(arrays))
	for i, a := range arrays {
		dt, nb := a.dtype()
		var elems int64 = 1
		for _, s := range a.Shape {
			elems *= s
		}
		size := int64(4)
		if dt == "u8" {
			size = 1
		}
		if elems*size != nb {
			return 0, fmt.Errorf("vro: section %s: shape %v does not match %d bytes", a.Name, a.Shape, nb)
		}
		h.Sections[i] = Section{Name: a.Name, Dtype: dt, Shape: a.Shape, Bytes: nb}
	}
	// The offsets depend on the header length, and the header holds the
	// offsets: iterate until the data start does not change.
	var hdr []byte
	start := int64(-1)
	for iter := 0; ; iter++ {
		guess := start
		if guess < 0 {
			guess = 0
		}
		off := guess
		for i := range h.Sections {
			h.Sections[i].Offset = off
			off = alignUp(off + h.Sections[i].Bytes)
		}
		var err error
		hdr, err = json.Marshal(h)
		if err != nil {
			return 0, err
		}
		need := alignUp(int64(len(Magic)) + 4 + int64(len(hdr)))
		if need == start {
			break
		}
		if iter > 10 {
			return 0, fmt.Errorf("vro: header length does not converge")
		}
		start = need
	}
	for _, c := range hdr {
		if c > 0x7f {
			return 0, fmt.Errorf("vro: header is not ASCII")
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	pos := int64(0)
	put := func(b []byte) error {
		n, err := w.Write(b)
		pos += int64(n)
		return err
	}
	pad := func(to int64) error {
		if to < pos {
			return fmt.Errorf("vro: offset %d behind position %d", to, pos)
		}
		return put(make([]byte, to-pos))
	}
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(hdr)))
	err = put([]byte(Magic))
	if err == nil {
		err = put(lenBuf[:])
	}
	if err == nil {
		err = put(hdr)
	}
	buf := make([]byte, 1<<16)
	for i, a := range arrays {
		if err != nil {
			break
		}
		if err = pad(h.Sections[i].Offset); err != nil {
			break
		}
		switch {
		case a.F32 != nil:
			for j := 0; j < len(a.F32) && err == nil; {
				k := 0
				for ; k+4 <= len(buf) && j < len(a.F32); j, k = j+1, k+4 {
					binary.LittleEndian.PutUint32(buf[k:], math.Float32bits(a.F32[j]))
				}
				err = put(buf[:k])
			}
		case a.I32 != nil:
			for j := 0; j < len(a.I32) && err == nil; {
				k := 0
				for ; k+4 <= len(buf) && j < len(a.I32); j, k = j+1, k+4 {
					binary.LittleEndian.PutUint32(buf[k:], uint32(a.I32[j]))
				}
				err = put(buf[:k])
			}
		default:
			err = put(a.U8)
		}
	}
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return 0, err
	}
	return pos, nil
}

// File is an open .vro file with its parsed header.
type File struct {
	Header Header
	f      *os.File
	size   int64
}

// Open reads the magic and the header and checks the section table: every
// offset is a multiple of 64, past the header, and inside the file.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	var pre [12]byte
	if _, err := io.ReadFull(f, pre[:]); err != nil {
		f.Close()
		return nil, fmt.Errorf("vro: %s: short file: %v", path, err)
	}
	if string(pre[:8]) != Magic {
		f.Close()
		return nil, fmt.Errorf("vro: %s: bad magic %q, want %q", path, pre[:8], Magic)
	}
	hl := int64(binary.LittleEndian.Uint32(pre[8:]))
	if 12+hl > st.Size() {
		f.Close()
		return nil, fmt.Errorf("vro: %s: header length %d past the end of the file", path, hl)
	}
	raw := make([]byte, hl)
	if _, err := io.ReadFull(f, raw); err != nil {
		f.Close()
		return nil, err
	}
	var h Header
	if err := json.Unmarshal(raw, &h); err != nil {
		f.Close()
		return nil, fmt.Errorf("vro: %s: bad header: %v", path, err)
	}
	dataStart := 12 + hl
	for _, s := range h.Sections {
		if s.Offset%Align != 0 || s.Offset < dataStart || s.Offset+s.Bytes > st.Size() {
			f.Close()
			return nil, fmt.Errorf("vro: %s: section %s at offset %d (%d bytes) is not aligned or not inside the file", path, s.Name, s.Offset, s.Bytes)
		}
	}
	return &File{Header: h, f: f, size: st.Size()}, nil
}

// Close closes the file.
func (v *File) Close() error { return v.f.Close() }

// Size returns the file size in bytes.
func (v *File) Size() int64 { return v.size }

// Section returns the table entry of name.
func (v *File) Section(name string) (Section, error) {
	for _, s := range v.Header.Sections {
		if s.Name == name {
			return s, nil
		}
	}
	return Section{}, fmt.Errorf("vro: no section %q", name)
}

func (v *File) find(name, dtype string, size int64) (Section, error) {
	s, err := v.Section(name)
	if err != nil {
		return s, err
	}
	if s.Dtype != dtype {
		return s, fmt.Errorf("vro: section %s has dtype %s, want %s", name, s.Dtype, dtype)
	}
	elems := int64(1)
	for _, d := range s.Shape {
		elems *= d
	}
	if elems*size != s.Bytes {
		return s, fmt.Errorf("vro: section %s: shape %v does not match %d bytes", name, s.Shape, s.Bytes)
	}
	return s, nil
}

// each streams the bytes of section s in chunks to fn.
func (v *File) each(s Section, unit int, fn func(b []byte)) error {
	buf := make([]byte, 1<<20/unit*unit)
	r := io.NewSectionReader(v.f, s.Offset, s.Bytes)
	for left := s.Bytes; left > 0; {
		k := int64(len(buf))
		if left < k {
			k = left
		}
		if _, err := io.ReadFull(r, buf[:k]); err != nil {
			return err
		}
		fn(buf[:k])
		left -= k
	}
	return nil
}

// F32 reads a float32 section. It returns the data and the shape.
func (v *File) F32(name string) ([]float32, []int64, error) {
	s, err := v.find(name, "f32", 4)
	if err != nil {
		return nil, nil, err
	}
	out := make([]float32, 0, s.Bytes/4)
	err = v.each(s, 4, func(b []byte) {
		for i := 0; i < len(b); i += 4 {
			out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
		}
	})
	return out, s.Shape, err
}

// I32 reads an int32 section.
func (v *File) I32(name string) ([]int32, []int64, error) {
	s, err := v.find(name, "int32", 4)
	if err != nil {
		return nil, nil, err
	}
	out := make([]int32, 0, s.Bytes/4)
	err = v.each(s, 4, func(b []byte) {
		for i := 0; i < len(b); i += 4 {
			out = append(out, int32(binary.LittleEndian.Uint32(b[i:])))
		}
	})
	return out, s.Shape, err
}

// U8 reads a u8 section.
func (v *File) U8(name string) ([]byte, []int64, error) {
	s, err := v.find(name, "u8", 1)
	if err != nil {
		return nil, nil, err
	}
	out := make([]byte, s.Bytes)
	_, err = io.ReadFull(io.NewSectionReader(v.f, s.Offset, s.Bytes), out)
	return out, s.Shape, err
}

// MismatchError is a refusal: the file's index, dim or build_params differ
// from what the caller expects.
type MismatchError struct{ Msg string }

func (e MismatchError) Error() string { return e.Msg }

// Check refuses a file whose index is not index, whose dim is not dim (0 =
// any dim), or whose build_params differ from params (nil = accept the
// file's). Every key of params must be in the file with an equal value, and
// the file must hold no key that params lacks. Numbers compare by value.
func (v *File) Check(index string, dim int, params map[string]any) error {
	h := v.Header
	if h.Index != index {
		return MismatchError{fmt.Sprintf("vro: file holds index %q, want %q", h.Index, index)}
	}
	if dim > 0 && h.Dim != dim {
		return MismatchError{fmt.Sprintf("vro: file has dim %d, want %d", h.Dim, dim)}
	}
	if params == nil {
		return nil
	}
	keys := map[string]bool{}
	for k := range params {
		keys[k] = true
	}
	for k := range h.BuildParams {
		keys[k] = true
	}
	names := make([]string, 0, len(keys))
	for k := range keys {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		want, ok1 := params[k]
		got, ok2 := h.BuildParams[k]
		if !ok1 || !ok2 || !Equal(want, got) {
			return MismatchError{fmt.Sprintf("vro: build parameter %s: file has %v, want %v", k, got, want)}
		}
	}
	return nil
}

// Equal compares two parameter values; numbers compare as float64.
func Equal(a, b any) bool {
	fa, oka := num(a)
	fb, okb := num(b)
	if oka && okb {
		return fa == fb
	}
	return fmt.Sprint(a) == fmt.Sprint(b)
}

func num(x any) (float64, bool) {
	switch v := x.(type) {
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	}
	return 0, false
}

// IntParam reads an integer from a header's build_params (JSON numbers are
// float64).
func IntParam(p map[string]any, key string) (int, error) {
	f, ok := num(p[key])
	if !ok || f != math.Trunc(f) {
		return 0, fmt.Errorf("vro: build parameter %s = %v is not an integer", key, p[key])
	}
	return int(f), nil
}

// Load opens path and runs Check. The caller closes the file.
func Load(path, index string, dim int, params map[string]any) (*File, error) {
	v, err := Open(path)
	if err != nil {
		return nil, err
	}
	if err := v.Check(index, dim, params); err != nil {
		v.Close()
		return nil, err
	}
	if v.Header.N < 0 || v.Header.Dim <= 0 {
		v.Close()
		return nil, fmt.Errorf("vro: bad n=%d dim=%d", v.Header.N, v.Header.Dim)
	}
	return v, nil
}

// Vectors reads the "vectors" section and checks its shape (n, dim).
func (v *File) Vectors() ([]float32, error) {
	x, sh, err := v.F32("vectors")
	if err != nil {
		return nil, err
	}
	if len(sh) != 2 || sh[0] != int64(v.Header.N) || sh[1] != int64(v.Header.Dim) {
		return nil, fmt.Errorf("vro: vectors shape %v, want [%d %d]", sh, v.Header.N, v.Header.Dim)
	}
	return x, nil
}

// Tombstones reads the "tombstones" bit set (ceil(n/8) bytes; bit i = row i
// deleted, LSB first within a byte).
func (v *File) Tombstones() ([]byte, error) {
	b, _, err := v.U8("tombstones")
	if err != nil {
		return nil, err
	}
	if len(b) != (v.Header.N+7)/8 {
		return nil, fmt.Errorf("vro: tombstones has %d bytes, want %d", len(b), (v.Header.N+7)/8)
	}
	return b, nil
}
