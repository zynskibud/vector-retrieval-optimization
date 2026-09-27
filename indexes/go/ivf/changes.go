package ivf

// Phase 5: deletes, updates and compaction (CONTRACT.md section 13.3).
//
// Delete sets bits in a tombstone bit set; Search skips tombstoned rows in the
// scanned lists (they are not scored and not counted in distance_computations).
// Update overwrites the vector in place, in the corpus array the index was
// built on (ivf copies no vectors), assigns the row to its best center again,
// and moves the ID to that list. The CSR lists are rebuilt once per Update
// call with a counting sort, so IDs stay ascending inside each list.
// Compact removes the tombstoned IDs from the lists, rebuilds the offsets and
// drops the bit set; IndexBytes then shrinks by 4 bytes per deleted row plus
// the bit set. The centers are not retrained. mode "rebuild" and "repair" do
// the same thing for ivf.
//
// Concurrency: the change functions run with no concurrent searches.

import (
	"fmt"

	"vro/indexes/go/kmeans"
	"vro/indexes/go/tombstone"
)

// Delete marks every row i with mask[i] true as deleted. It returns the rows
// newly deleted.
func Delete(ix *Index, mask []bool) (int, error) {
	if ix.del == nil {
		ix.del = tombstone.New(ix.n)
	}
	return tombstone.Apply(ix.del, mask), nil
}

// Update overwrites the vector of each row ids[j] and moves the row to the
// list of its new best center.
func Update(ix *Index, ids []int64, vectors []float32) (int, error) {
	d := ix.dim
	if len(vectors) < len(ids)*d {
		return 0, fmt.Errorf("ivf: Update got %d ids and %d floats", len(ids), len(vectors))
	}
	// Current list of every row in the lists.
	label := make([]int32, ix.n)
	for i := range label {
		label[i] = -1
	}
	for c := 0; c < ix.nlist; c++ {
		for _, id := range ix.ids[ix.offsets[c]:ix.offsets[c+1]] {
			label[id] = int32(c)
		}
	}
	for j, id := range ids {
		if id < 0 || id >= int64(ix.n) || label[id] < 0 {
			return 0, fmt.Errorf("ivf: Update row %d is not in the index", id)
		}
		row := ix.vectors[int(id)*d : int(id)*d+d]
		copy(row, vectors[j*d:j*d+d])
		c, _ := kmeans.Nearest(row, ix.centers, ix.nlist, d, false)
		label[id] = int32(c)
	}
	ix.rebuildLists(label)
	return len(ids), nil
}

// Compact drops the tombstoned rows from the lists. Without tombstones it
// changes nothing.
func Compact(ix *Index, mode string) (*Index, error) {
	if mode != "rebuild" && mode != "repair" {
		return nil, fmt.Errorf("ivf: unknown compact mode %q", mode)
	}
	if ix.del == nil {
		return ix, nil
	}
	label := make([]int32, ix.n)
	for i := range label {
		label[i] = -1
	}
	for c := 0; c < ix.nlist; c++ {
		for _, id := range ix.ids[ix.offsets[c]:ix.offsets[c+1]] {
			if !ix.del.Has(int(id)) {
				label[id] = int32(c)
			}
		}
	}
	ix.rebuildLists(label)
	ix.del = nil
	return ix, nil
}

// rebuildLists rebuilds ids and offsets from one list label per row
// (-1 = not in any list), with a counting sort in row order.
func (ix *Index) rebuildLists(label []int32) {
	offsets := make([]int32, ix.nlist+1)
	total := 0
	for _, c := range label {
		if c >= 0 {
			offsets[c+1]++
			total++
		}
	}
	for c := 0; c < ix.nlist; c++ {
		offsets[c+1] += offsets[c]
	}
	ids := make([]int32, total)
	next := make([]int32, ix.nlist)
	copy(next, offsets[:ix.nlist])
	for i, c := range label {
		if c >= 0 {
			ids[next[c]] = int32(i)
			next[c]++
		}
	}
	ix.ids, ix.offsets = ids, offsets
}
