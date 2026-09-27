package ivf

// Save and load (CONTRACT.md section 15.1). An ivf file holds "vectors",
// "tombstones", "centers" (f32, nlist x dim), "list_ids" (int32) and
// "list_offsets" (int32, nlist + 1), the CSR lists of ivf.go as they are.
// list_ids has one entry per row in the lists: N after a build, fewer after
// Compact (the dropped rows are no longer in any list).

import (
	"fmt"

	"vro/indexes/go/tombstone"
	"vro/indexes/go/vro"
)

// Save writes ix to path. It returns the file size in bytes.
func Save(ix *Index, path string) (int64, error) {
	n, d := ix.n, ix.dim
	h := vro.Header{Index: "ivf", N: n, Dim: d, BuildParams: ix.buildParams(),
		Seed: ix.seed, ContractVersion: 1, Language: "go"}
	return vro.Write(path, h, []vro.Array{
		{Name: "vectors", Shape: []int64{int64(n), int64(d)}, F32: ix.vectors[:n*d]},
		{Name: "tombstones", Shape: []int64{int64((n + 7) / 8)}, U8: tombstone.Bits(ix.del, n)},
		{Name: "centers", Shape: []int64{int64(ix.nlist), int64(d)}, F32: ix.centers},
		{Name: "list_ids", Shape: []int64{int64(len(ix.ids))}, I32: ix.ids},
		{Name: "list_offsets", Shape: []int64{int64(ix.nlist + 1)}, I32: ix.offsets},
	})
}

// buildParams returns every build parameter, defaults resolved.
func (ix *Index) buildParams() map[string]any {
	return map[string]any{"nlist": int64(ix.nlist), "train_size": int64(ix.trainSize), "iters": int64(ix.iters)}
}

// Load reads an ivf file. dim is the expected dimension (0 = any); params are
// the expected build parameters (nil = accept the file's).
func Load(path string, dim int, params map[string]any) (*Index, *vro.Header, error) {
	f, err := vro.Load(path, "ivf", dim, params)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	h := f.Header
	ix := &Index{n: h.N, dim: h.Dim, seed: h.Seed}
	if ix.nlist, err = vro.IntParam(h.BuildParams, "nlist"); err != nil {
		return nil, nil, err
	}
	if ix.trainSize, err = vro.IntParam(h.BuildParams, "train_size"); err != nil {
		return nil, nil, err
	}
	if ix.iters, err = vro.IntParam(h.BuildParams, "iters"); err != nil {
		return nil, nil, err
	}
	if ix.vectors, err = f.Vectors(); err != nil {
		return nil, nil, err
	}
	bits, err := f.Tombstones()
	if err != nil {
		return nil, nil, err
	}
	ix.del = tombstone.FromBits(bits, h.N)
	if ix.centers, _, err = f.F32("centers"); err != nil {
		return nil, nil, err
	}
	if ix.ids, _, err = f.I32("list_ids"); err != nil {
		return nil, nil, err
	}
	if ix.offsets, _, err = f.I32("list_offsets"); err != nil {
		return nil, nil, err
	}
	if len(ix.centers) != ix.nlist*ix.dim || len(ix.offsets) != ix.nlist+1 {
		return nil, nil, fmt.Errorf("ivf: centers %d floats, offsets %d, want nlist=%d x dim=%d and nlist+1", len(ix.centers), len(ix.offsets), ix.nlist, ix.dim)
	}
	if ix.offsets[0] != 0 || int(ix.offsets[ix.nlist]) != len(ix.ids) {
		return nil, nil, fmt.Errorf("ivf: list_offsets do not cover list_ids (%d..%d, %d ids)", ix.offsets[0], ix.offsets[ix.nlist], len(ix.ids))
	}
	for c := 0; c < ix.nlist; c++ {
		if ix.offsets[c] > ix.offsets[c+1] {
			return nil, nil, fmt.Errorf("ivf: list_offsets decrease at list %d", c)
		}
	}
	for _, id := range ix.ids {
		if id < 0 || int(id) >= ix.n {
			return nil, nil, fmt.Errorf("ivf: list_ids holds row %d outside 0..%d", id, ix.n-1)
		}
	}
	return ix, &h, nil
}
