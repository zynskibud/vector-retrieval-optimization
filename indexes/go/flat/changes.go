package flat

// Phase 5: deletes, updates and compaction (CONTRACT.md section 13.3).
//
// Delete sets bits in a tombstone bit set; Search skips tombstoned rows.
// Update overwrites the vectors in place, in the array the index was built on
// (flat copies nothing, so the caller's corpus array changes). Compact copies
// the live rows into a new array owned by the index, with an int32 map from
// the new row to the corpus row ID, and drops the bit set.
//
// Memory (IndexBytes). A plain flat index reports 0: the corpus array is the
// index. After a change the index no longer serves the corpus array as it is
// (rows are dead, or after Compact the index owns a copy), so IndexBytes then
// counts, as CONTRACT 13.3 fixes, the live rows it serves (live rows x dim x
// 4) plus the ID map (4 bytes per live row) plus the bit set (N/8 bytes,
// before compaction only). So index_bytes_after < index_bytes_before_compact.
//
// Concurrency: the change functions run with no concurrent searches.

import (
	"fmt"

	"vro/indexes/go/distance"
	"vro/indexes/go/tombstone"
)

// Delete marks every row i with mask[i] true as deleted. It returns the rows
// newly deleted.
func Delete(ix *Index, mask []bool) (int, error) {
	if ix.ids != nil {
		return 0, fmt.Errorf("flat: Delete after Compact is not supported")
	}
	if ix.del == nil {
		ix.del = tombstone.New(ix.n)
	}
	ix.changed = true
	return tombstone.Apply(ix.del, mask), nil
}

// Update overwrites the vector of each row ids[j] with vectors[j*dim:(j+1)*dim].
func Update(ix *Index, ids []int64, vectors []float32) (int, error) {
	if ix.ids != nil {
		return 0, fmt.Errorf("flat: Update after Compact is not supported")
	}
	d := ix.dim
	if len(vectors) < len(ids)*d {
		return 0, fmt.Errorf("flat: Update got %d ids and %d floats", len(ids), len(vectors))
	}
	for j, id := range ids {
		if id < 0 || id >= int64(ix.n) {
			return 0, fmt.Errorf("flat: Update row %d outside 0..%d", id, ix.n-1)
		}
		copy(ix.vectors[int(id)*d:int(id)*d+d], vectors[j*d:j*d+d])
	}
	ix.changed = true
	return len(ids), nil
}

// Compact drops the tombstoned rows. mode is "rebuild" or "repair"; for flat
// both do the same thing. Without tombstones it changes nothing.
func Compact(ix *Index, mode string) (*Index, error) {
	if mode != "rebuild" && mode != "repair" {
		return nil, fmt.Errorf("flat: unknown compact mode %q", mode)
	}
	ix.changed = true
	if ix.del == nil {
		return ix, nil
	}
	d := ix.dim
	live := ix.n - ix.del.Count()
	vecs := make([]float32, 0, live*d)
	ids := make([]int32, 0, live)
	for i := 0; i < ix.n; i++ {
		if ix.del.Has(i) {
			continue
		}
		vecs = append(vecs, ix.vectors[i*d:i*d+d]...)
		ids = append(ids, int32(i))
	}
	return &Index{vectors: vecs, n: live, dim: d, trainS: ix.trainS, addS: ix.addS,
		dists: ix.dists, passed: ix.passed, ids: ids, changed: true, total: ix.n, seed: ix.seed}, nil
}

// searchChanged is Search with tombstones or an ID map. mask is indexed by
// corpus row ID.
func (ix *Index) searchChanged(query []float32, tk *distance.TopK, mask []bool) ([]int64, []float32) {
	d := ix.dim
	scored := 0
	for r := 0; r < ix.n; r++ {
		if ix.del.Has(r) {
			continue
		}
		id := r
		if ix.ids != nil {
			id = int(ix.ids[r])
		}
		if mask != nil && !mask[id] {
			continue
		}
		tk.Push(int64(id), distance.Dot(query, ix.vectors[r*d:(r+1)*d]))
		scored++
	}
	ix.dists += int64(scored)
	ix.passed += int64(scored)
	return tk.Results()
}
