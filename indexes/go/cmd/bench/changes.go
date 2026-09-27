package main

// Updates, deletes and compaction (CONTRACT.md section 13.2). The change is
// applied after the build and before the warm-up; each step is timed and
// reported in the top-level "extra". The change runs with no concurrent
// searches.

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"vro/indexes/go/flat"
	"vro/indexes/go/hnsw"
	"vro/indexes/go/ivf"
	"vro/indexes/go/npy"
)

var (
	deleteNames  = map[string]bool{"del10": true, "del30": true, "del50": true}
	updateNames  = map[string]bool{"upd10": true}
	compactModes = map[string]bool{"rebuild": true, "repair": true}
	// changeIndexes lists the indexes that accept --delete, --update, --compact.
	changeIndexes = map[string]bool{"flat": true, "ivf": true, "hnsw": true}
)

// changeMode reports whether the flags ask for a Phase 5 change.
func (o options) changeMode() bool { return o.deleteName != "" || o.updateName != "" }

// checkChangeFlags validates the Phase 5 flags. Every error exits 2.
func checkChangeFlags(o options) error {
	if o.deleteName != "" && !deleteNames[o.deleteName] {
		return usagef("--delete must be del10, del30 or del50, got %q", o.deleteName)
	}
	if o.updateName != "" && !updateNames[o.updateName] {
		return usagef("--update must be upd10, got %q", o.updateName)
	}
	if o.deleteName != "" && o.updateName != "" {
		return usagef("--delete and --update are not combined in one run")
	}
	if !compactModes[o.compactMode] {
		return usagef("--compact-mode must be rebuild or repair, got %q", o.compactMode)
	}
	if o.compact && !o.changeMode() {
		return usagef("--compact needs --delete or --update")
	}
	if o.changeMode() && !changeIndexes[o.index] {
		return usagef("--delete, --update and --compact apply to flat, ivf and hnsw, not %q", o.index)
	}
	if o.changeMode() && o.loadMode() {
		return usagef("--delete and --update are not combined with a load run")
	}
	return nil
}

// changeOps are the Phase 5 functions of one built index.
type changeOps struct {
	del     func(mask []bool) (int, error)
	upd     func(ids []int64, vecs []float32) (int, error)
	compact func(mode string) (instance, error)
	extra   func() map[string]any
}

func opsFor(inst instance) (changeOps, error) {
	switch ix := inst.idx.(type) {
	case *flat.Index:
		return changeOps{
			del: func(m []bool) (int, error) { return flat.Delete(ix, m) },
			upd: func(ids []int64, v []float32) (int, error) { return flat.Update(ix, ids, v) },
			compact: func(mode string) (instance, error) {
				nix, err := flat.Compact(ix, mode)
				if err != nil {
					return instance{}, err
				}
				return instance{idx: nix, bytes: flat.IndexBytes(nix),
					search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return flat.Search(nix, q, k, sp) }}, nil
			},
			extra: func() map[string]any { return nil },
		}, nil
	case *ivf.Index:
		return changeOps{
			del: func(m []bool) (int, error) { return ivf.Delete(ix, m) },
			upd: func(ids []int64, v []float32) (int, error) { return ivf.Update(ix, ids, v) },
			compact: func(mode string) (instance, error) {
				nix, err := ivf.Compact(ix, mode)
				if err != nil {
					return instance{}, err
				}
				return instance{idx: nix, bytes: ivf.IndexBytes(nix),
					search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return ivf.Search(nix, q, k, sp) }}, nil
			},
			extra: func() map[string]any { return nil },
		}, nil
	case *hnsw.Index:
		return changeOps{
			del: func(m []bool) (int, error) { return hnsw.Delete(ix, m) },
			upd: func(ids []int64, v []float32) (int, error) { return hnsw.Update(ix, ids, v) },
			compact: func(mode string) (instance, error) {
				nix, err := hnsw.Compact(ix, mode)
				if err != nil {
					return instance{}, err
				}
				return instance{idx: nix, bytes: hnsw.IndexBytes(nix),
					search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return hnsw.Search(nix, q, k, sp) }}, nil
			},
			extra: func() map[string]any { return ix.ChangeExtra() },
		}, nil
	}
	return changeOps{}, usagef("index %T does not support changes", inst.idx)
}

// changeData holds the change set read from the data directory before the
// build: a delete mask, or the update IDs and vectors (rows < n only).
type changeData struct {
	mask    []bool
	ids     []int64
	vectors []float32
}

// readChanges reads the change set of o for the first n rows.
func readChanges(o options, n, dim int) (changeData, error) {
	var c changeData
	if o.deleteName != "" {
		m, err := npy.ReadBool(filepath.Join(o.data, "delete_"+o.deleteName+".npy"))
		if err != nil {
			return c, err
		}
		if len(m) < n {
			return c, fmt.Errorf("delete %s has %d rows, corpus has %d", o.deleteName, len(m), n)
		}
		c.mask = m[:n]
	}
	if o.updateName != "" {
		ids, err := npy.ReadInt64Vec(filepath.Join(o.data, "update_"+o.updateName+"_ids.npy"))
		if err != nil {
			return c, err
		}
		vecs, k, vdim, err := npy.ReadFloat32(filepath.Join(o.data, "update_"+o.updateName+"_vectors.npy"))
		if err != nil {
			return c, err
		}
		if k != len(ids) || vdim != dim {
			return c, fmt.Errorf("update %s: %d ids, vectors %d x %d, corpus dim %d", o.updateName, len(ids), k, vdim, dim)
		}
		for j, id := range ids {
			if id >= 0 && id < int64(n) { // --limit keeps only rows in the corpus
				c.ids = append(c.ids, id)
				c.vectors = append(c.vectors, vecs[j*dim:(j+1)*dim]...)
			}
		}
	}
	return c, nil
}

// applyChanges runs the delete or update, then the optional compaction. It
// returns the instance to search and the keys for the top-level extra.
func applyChanges(o options, inst instance, c changeData) (instance, map[string]any, error) {
	ops, err := opsFor(inst)
	if err != nil {
		return inst, nil, err
	}
	extra := map[string]any{}
	if c.mask != nil {
		fmt.Fprintf(os.Stderr, "bench: delete %s\n", o.deleteName)
		t0 := time.Now()
		rows, err := ops.del(c.mask)
		if err != nil {
			return inst, nil, err
		}
		extra["delete_s"] = time.Since(t0).Seconds()
		extra["deleted_rows"] = rows
	}
	if o.updateName != "" {
		fmt.Fprintf(os.Stderr, "bench: update %s (%d rows)\n", o.updateName, len(c.ids))
		t0 := time.Now()
		rows, err := ops.upd(c.ids, c.vectors)
		if err != nil {
			return inst, nil, err
		}
		extra["update_s"] = time.Since(t0).Seconds()
		extra["updated_rows"] = rows
	}
	// index_bytes after the change, before any compaction (tombstones counted).
	extra["index_bytes_before_compact"] = indexBytesOf(inst)
	for key, v := range ops.extra() {
		extra[key] = v
	}
	if o.compact {
		fmt.Fprintf(os.Stderr, "bench: compact (%s)\n", o.compactMode)
		t0 := time.Now()
		ninst, err := ops.compact(o.compactMode)
		if err != nil {
			return inst, nil, err
		}
		extra["compact_s"] = time.Since(t0).Seconds()
		extra["compact_mode"] = o.compactMode
		extra["index_bytes_after"] = ninst.bytes
		if nops, err := opsFor(ninst); err == nil {
			for key, v := range nops.extra() {
				extra[key] = v
			}
		}
		ninst.rows = inst.rows
		inst = ninst
	}
	return inst, extra, nil
}

// indexBytesOf computes IndexBytes of the current index.
func indexBytesOf(inst instance) int64 {
	switch ix := inst.idx.(type) {
	case *flat.Index:
		return flat.IndexBytes(ix)
	case *ivf.Index:
		return ivf.IndexBytes(ix)
	case *hnsw.Index:
		return hnsw.IndexBytes(ix)
	}
	return inst.bytes
}

// changeSearchParams adds the Phase 5 keys to a search parameter set.
func changeSearchParams(o options, sp map[string]any) {
	if !o.changeMode() {
		return
	}
	if o.deleteName != "" {
		sp["deleted"] = o.deleteName
	}
	if o.updateName != "" {
		sp["updated"] = o.updateName
	}
	c := 0
	if o.compact {
		c = 1
	}
	sp["compacted"] = c
}
