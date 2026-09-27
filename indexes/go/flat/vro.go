package flat

// Save and load (CONTRACT.md section 15.1). A flat file holds the sections
// "vectors" (f32, N x dim) and "tombstones" (u8, ceil(N/8)).
//
// After Compact the index holds only the live rows and a map to their corpus
// IDs. Save then writes all N corpus rows again: a live row gets its vector,
// a dropped row gets a zero vector and its tombstone bit. The loaded index
// skips the same rows, so it returns the same IDs and scores.

import (
	"vro/indexes/go/tombstone"
	"vro/indexes/go/vro"
)

// Save writes ix to path. It returns the file size in bytes.
func Save(ix *Index, path string) (int64, error) {
	d := ix.dim
	n := ix.n
	vecs := ix.vectors[:n*d]
	del := ix.del
	if ix.ids != nil {
		n = ix.total
		vecs = make([]float32, n*d)
		del = tombstone.New(n)
		live := make([]bool, n)
		for r, id := range ix.ids {
			copy(vecs[int(id)*d:int(id)*d+d], ix.vectors[r*d:r*d+d])
			live[id] = true
		}
		for i, ok := range live {
			if !ok {
				del.Add(i)
			}
		}
	}
	h := vro.Header{Index: "flat", N: n, Dim: d, BuildParams: map[string]any{},
		Seed: ix.seed, ContractVersion: 1, Language: "go"}
	return vro.Write(path, h, []vro.Array{
		{Name: "vectors", Shape: []int64{int64(n), int64(d)}, F32: vecs},
		{Name: "tombstones", Shape: []int64{int64((n + 7) / 8)}, U8: tombstone.Bits(del, n)},
	})
}

// Load reads a flat file. dim is the expected dimension (0 = any); params are
// the expected build parameters (nil = accept the file's). The loaded index
// owns its vectors and needs no rebuild.
func Load(path string, dim int, params map[string]any) (*Index, *vro.Header, error) {
	f, err := vro.Load(path, "flat", dim, params)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	h := f.Header
	vecs, err := f.Vectors()
	if err != nil {
		return nil, nil, err
	}
	bits, err := f.Tombstones()
	if err != nil {
		return nil, nil, err
	}
	ix := &Index{vectors: vecs, n: h.N, dim: h.Dim, seed: h.Seed}
	if ix.del = tombstone.FromBits(bits, h.N); ix.del != nil {
		ix.changed = true
	}
	return ix, &h, nil
}
