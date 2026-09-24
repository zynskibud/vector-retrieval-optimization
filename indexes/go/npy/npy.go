// Package npy reads NumPy .npy files (format version 1.0) as described in
// CONTRACT.md section 1.1. The header is parsed by hand.
package npy

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unsafe"
)

// header is the parsed .npy header dict.
type header struct {
	descr   string
	fortran bool
	shape   []int
}

// ReadFloat32 reads a 2-D '<f4' array. It returns the row-major data and its shape.
func ReadFloat32(path string) (data []float32, rows, dim int, err error) {
	raw, h, off, err := load(path, "<f4")
	if err != nil {
		return nil, 0, 0, err
	}
	rows, dim = h.shape[0], h.shape[1]
	body := raw[off:]
	if len(body) != rows*dim*4 {
		return nil, 0, 0, fmt.Errorf("npy %s: data is %d bytes, shape needs %d", path, len(body), rows*dim*4)
	}
	if rows*dim == 0 {
		return []float32{}, rows, dim, nil
	}
	// The file is little-endian. On a little-endian host the bytes already have
	// the in-memory layout of []float32, so we reinterpret the buffer instead of
	// copying 1.8 GB. The data offset is a multiple of 64 and the buffer comes
	// from a large heap allocation, so the pointer is 4-byte aligned.
	return unsafe.Slice((*float32)(unsafe.Pointer(&body[0])), rows*dim), rows, dim, nil
}

// ReadInt64 reads a 2-D '<i8' array. It returns the row-major data and its shape.
func ReadInt64(path string) (data []int64, rows, cols int, err error) {
	raw, h, off, err := load(path, "<i8")
	if err != nil {
		return nil, 0, 0, err
	}
	rows, cols = h.shape[0], h.shape[1]
	body := raw[off:]
	if len(body) != rows*cols*8 {
		return nil, 0, 0, fmt.Errorf("npy %s: data is %d bytes, shape needs %d", path, len(body), rows*cols*8)
	}
	// This file is small (ground truth), so a portable decode is enough.
	data = make([]int64, rows*cols)
	for i := range data {
		data[i] = int64(binary.LittleEndian.Uint64(body[i*8:]))
	}
	return data, rows, cols, nil
}

// load reads the file, checks the magic, version and header, and returns the
// raw bytes plus the data offset.
func load(path, wantDescr string) ([]byte, header, int, error) {
	if !hostLittleEndian() {
		return nil, header{}, 0, fmt.Errorf("npy: big-endian hosts are not supported")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, header{}, 0, err
	}
	if len(raw) < 10 || !bytes.Equal(raw[:6], []byte("\x93NUMPY")) {
		return nil, header{}, 0, fmt.Errorf("npy %s: bad magic string", path)
	}
	if raw[6] != 1 || raw[7] != 0 {
		return nil, header{}, 0, fmt.Errorf("npy %s: version %d.%d, want 1.0", path, raw[6], raw[7])
	}
	hlen := int(binary.LittleEndian.Uint16(raw[8:10]))
	off := 10 + hlen
	if len(raw) < off {
		return nil, header{}, 0, fmt.Errorf("npy %s: truncated header", path)
	}
	h, err := parseHeader(string(raw[10:off]))
	if err != nil {
		return nil, header{}, 0, fmt.Errorf("npy %s: %w", path, err)
	}
	if h.descr != wantDescr {
		return nil, header{}, 0, fmt.Errorf("npy %s: descr %q, want %q", path, h.descr, wantDescr)
	}
	if h.fortran {
		return nil, header{}, 0, fmt.Errorf("npy %s: fortran_order is True, want False", path)
	}
	if len(h.shape) != 2 {
		return nil, header{}, 0, fmt.Errorf("npy %s: shape %v, want 2 dimensions", path, h.shape)
	}
	return raw, h, off, nil
}

// parseHeader parses a dict such as
// {'descr': '<f4', 'fortran_order': False, 'shape': (1000, 384), }
func parseHeader(s string) (header, error) {
	var h header
	descr, err := valueAfter(s, "'descr':")
	if err != nil {
		return h, err
	}
	h.descr = strings.Trim(descr, "' ")
	fo, err := valueAfter(s, "'fortran_order':")
	if err != nil {
		return h, err
	}
	switch strings.TrimSpace(fo) {
	case "False":
	case "True":
		h.fortran = true
	default:
		return h, fmt.Errorf("bad fortran_order %q", fo)
	}
	i := strings.Index(s, "'shape':")
	if i < 0 {
		return h, fmt.Errorf("header has no 'shape'")
	}
	rest := s[i+len("'shape':"):]
	open, end := strings.Index(rest, "("), strings.Index(rest, ")")
	if open < 0 || end < open {
		return h, fmt.Errorf("bad shape in header")
	}
	for _, part := range strings.Split(rest[open+1:end], ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil {
			return h, fmt.Errorf("bad shape value %q", part)
		}
		h.shape = append(h.shape, v)
	}
	return h, nil
}

// valueAfter returns the text between key and the next comma.
func valueAfter(s, key string) (string, error) {
	i := strings.Index(s, key)
	if i < 0 {
		return "", fmt.Errorf("header has no %s", key)
	}
	rest := s[i+len(key):]
	j := strings.Index(rest, ",")
	if j < 0 {
		return "", fmt.Errorf("bad value for %s", key)
	}
	return strings.TrimSpace(rest[:j]), nil
}

func hostLittleEndian() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}
