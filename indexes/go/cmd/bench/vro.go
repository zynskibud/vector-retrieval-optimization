package main

// Save and load of the .vro index file (CONTRACT.md section 15.2).
//
// --save PATH writes the index after the build and after any Phase 5 change,
// before the warm-up: extra.save_s, extra.file_bytes.
// --load PATH skips the build and reads the file instead: build.train_s =
// build.add_s = 0, extra.load_s, extra.loaded_from, and build_params from the
// file's header. A --build value that differs from the header is a usage
// error (exit 2). The corpus vectors come from the file, so vectors.npy is
// not read.

import (
	"fmt"
	"os"
	"strings"
	"time"

	"vro/indexes/go/flat"
	"vro/indexes/go/hnsw"
	"vro/indexes/go/ivf"
	"vro/indexes/go/vro"
)

// saveIndexes lists the indexes with Save and Load.
var saveIndexes = map[string]bool{"flat": true, "ivf": true, "hnsw": true}

// checkSaveFlags validates --save and --load. Every error exits 2.
func checkSaveFlags(o options) error {
	if (o.savePath != "" || o.loadPath != "") && !saveIndexes[o.index] {
		return usagef("--save and --load apply to flat, ivf and hnsw, not %q", o.index)
	}
	if o.loadPath != "" && o.insertRate > 0 {
		return usagef("--load is not combined with --insert-rate (the file holds no room for inserts)")
	}
	return nil
}

// fileParams reads the header of path, checks the index name, and returns
// the header with its build_params as int64 where they are integers. A
// --build value that differs from the header is a usage error.
func fileParams(o options, defaults map[string]any) (*vro.Header, map[string]any, error) {
	f, err := vro.Open(o.loadPath)
	if err != nil {
		return nil, nil, err
	}
	h := f.Header
	f.Close()
	if h.Index != o.index {
		return nil, nil, fmt.Errorf("--load %s holds index %q, not %q", o.loadPath, h.Index, o.index)
	}
	params := make(map[string]any, len(h.BuildParams))
	for k, v := range h.BuildParams {
		if x, ok := v.(float64); ok && x == float64(int64(x)) {
			params[k] = int64(x)
		} else {
			params[k] = v
		}
	}
	// Only the keys given on the command line are compared.
	for _, item := range o.builds {
		for _, pair := range strings.Split(item, ",") {
			key, raw, ok := strings.Cut(strings.TrimSpace(pair), "=")
			def, known := defaults[key]
			if !ok || !known {
				return nil, nil, usagef("bad or unknown build parameter %q", pair)
			}
			v, err := parseValue(key, raw, def)
			if err != nil {
				return nil, nil, err
			}
			if fv, ok := params[key]; !ok || !vro.Equal(fv, v) {
				return nil, nil, usagef("--build %s=%v conflicts with %s=%v in %s", key, v, key, params[key], o.loadPath)
			}
		}
	}
	if o.limit > 0 && o.limit != h.N {
		return nil, nil, usagef("--limit %d conflicts with n=%d in %s", o.limit, h.N, o.loadPath)
	}
	return &h, params, nil
}

// loadIndex reads the file at o.loadPath and wraps the index. params are the
// build parameters of the header; dim is the query dimension.
func loadIndex(o options, dim int, params map[string]any) (instance, float64, error) {
	t0 := time.Now()
	var inst instance
	switch o.index {
	case "flat":
		ix, _, err := flat.Load(o.loadPath, dim, params)
		if err != nil {
			return inst, 0, err
		}
		inst = instance{idx: ix, bytes: flat.IndexBytes(ix),
			search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return flat.Search(ix, q, k, sp) }}
	case "ivf":
		ix, _, err := ivf.Load(o.loadPath, dim, params)
		if err != nil {
			return inst, 0, err
		}
		inst = instance{idx: ix, bytes: ivf.IndexBytes(ix),
			search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return ivf.Search(ix, q, k, sp) }}
	case "hnsw":
		ix, _, err := hnsw.Load(o.loadPath, dim, params)
		if err != nil {
			return inst, 0, err
		}
		inst = instance{idx: ix, bytes: hnsw.IndexBytes(ix),
			search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return hnsw.Search(ix, q, k, sp) },
			repair: func() { hnsw.Repair(ix) },
			rows:   ix.Len}
	default:
		return inst, 0, usagef("--load does not apply to %q", o.index)
	}
	return inst, time.Since(t0).Seconds(), nil
}

// saveIndex writes the current index to o.savePath. It returns the keys for
// the top-level extra.
func saveIndex(o options, inst instance) (map[string]any, error) {
	t0 := time.Now()
	var size int64
	var err error
	switch ix := inst.idx.(type) {
	case *flat.Index:
		size, err = flat.Save(ix, o.savePath)
	case *ivf.Index:
		size, err = ivf.Save(ix, o.savePath)
	case *hnsw.Index:
		size, err = hnsw.Save(ix, o.savePath)
	default:
		err = usagef("--save does not apply to %T", inst.idx)
	}
	if err != nil {
		return nil, err
	}
	s := time.Since(t0).Seconds()
	fmt.Fprintf(os.Stderr, "bench: saved %s (%d bytes) in %.2f s\n", o.savePath, size, s)
	return map[string]any{"save_s": s, "file_bytes": size, "saved_to": o.savePath}, nil
}
