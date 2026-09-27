package hnsw

// Save and load (CONTRACT.md section 15.1). An hnsw file holds "vectors",
// "tombstones", "levels" (u8, N), "entry" (int32, [1]), "layer0_slots"
// (int32, N x 2m, -1 in an empty slot), "layer0_counts" (int32, N),
// "upper_slots" (int32, L x m), "upper_counts" (int32, L) and
// "upper_offsets" (int32, N + 1). L is the number of (node, layer) pairs
// above layer 0. Node i's blocks are upper_offsets[i] .. upper_offsets[i+1]-1,
// one per layer 1..level(i).
//
// The in-memory layout of hnsw.go is the same except for two points, and Save
// and Load convert them:
//   - upperOff[i] is the first block of node i, or -1 for a level-0 node. The
//     file holds prefix sums with N + 1 entries instead.
//   - Slots past a node's count may hold stale IDs in memory. The file holds -1.
//
// An index built with BuildCap for more rows than it holds is saved with its
// Len() rows only. After a Compact rebuild, node j is corpus row rowIDs[j].
// Save then writes N = the corpus rows before the rebuild, with every node
// ID mapped to its row ID: a dropped row gets a zero vector, level 0, no
// edges and its tombstone bit. No edge points to it, so no search reaches it,
// and the row order of the live nodes is unchanged, so ties break the same
// way. The loaded index returns the same IDs and scores.
//
// Load builds the arrays of hnsw.go from the sections, with capacity N. No
// rebuild and no repair runs. Slots and counts are plain arrays, read and
// written with sync/atomic as after a build, so the Phase 4 concurrent
// searches work on a loaded index. A tombstoned node with no edge on any
// layer is marked purged (as after Compact mode "repair"), so a later repair
// pass skips it.

import (
	"fmt"

	"vro/indexes/go/tombstone"
	"vro/indexes/go/vro"
)

// Save writes ix to path. It returns the file size in bytes.
func Save(ix *Index, path string) (int64, error) {
	nodes := ix.Len()
	d, m, m0 := ix.dim, ix.m, ix.m0
	n := nodes
	row := func(j int32) int32 { return j }
	if ix.rowIDs != nil {
		n = ix.total
		row = func(j int32) int32 { return ix.rowIDs[j] }
	}
	node := make([]int32, n) // node of row r, -1 for a dropped row
	for r := range node {
		node[r] = -1
	}
	for j := 0; j < nodes; j++ {
		node[row(int32(j))] = int32(j)
	}

	var vecs []float32
	del := ix.del
	if ix.rowIDs == nil {
		vecs = ix.vectors[:n*d]
	} else {
		vecs = make([]float32, n*d)
		del = tombstone.New(n)
		for r, j := range node {
			if j < 0 {
				del.Add(r)
				continue
			}
			copy(vecs[r*d:r*d+d], ix.vec(j))
		}
	}
	levels := make([]uint8, n)
	offsets := make([]int32, n+1)
	for r, j := range node {
		if j >= 0 {
			levels[r] = ix.levels[j]
		}
		offsets[r+1] = offsets[r] + int32(levels[r])
	}
	blocks := int(offsets[n])
	slots0 := make([]int32, n*m0)
	cnt0 := make([]int32, n)
	upper := make([]int32, blocks*m)
	cntU := make([]int32, blocks)
	for i := range slots0 {
		slots0[i] = -1
	}
	for i := range upper {
		upper[i] = -1
	}
	for r, j := range node {
		if j < 0 {
			continue
		}
		for l := 0; l <= int(levels[r]); l++ {
			sl, c := ix.slots(j, l)
			cnt := int(*c)
			var dst []int32
			if l == 0 {
				dst = slots0[r*m0 : r*m0+m0]
				cnt0[r] = int32(cnt)
			} else {
				b := int(offsets[r]) + l - 1
				dst = upper[b*m : b*m+m]
				cntU[b] = int32(cnt)
			}
			for k := 0; k < cnt; k++ {
				dst[k] = row(sl[k])
			}
		}
	}
	entry, _ := ix.entryTop()
	if entry >= 0 {
		entry = row(entry)
	}
	h := vro.Header{Index: "hnsw", N: n, Dim: d,
		BuildParams: map[string]any{"m": int64(m), "ef_construct": int64(ix.efC)},
		Seed:        ix.seed, ContractVersion: 1, Language: "go"}
	N := int64(n)
	return vro.Write(path, h, []vro.Array{
		{Name: "vectors", Shape: []int64{N, int64(d)}, F32: vecs},
		{Name: "tombstones", Shape: []int64{(N + 7) / 8}, U8: tombstone.Bits(del, n)},
		{Name: "levels", Shape: []int64{N}, U8: levels},
		{Name: "entry", Shape: []int64{1}, I32: []int32{entry}},
		{Name: "layer0_slots", Shape: []int64{N, int64(m0)}, I32: slots0},
		{Name: "layer0_counts", Shape: []int64{N}, I32: cnt0},
		{Name: "upper_slots", Shape: []int64{int64(blocks), int64(m)}, I32: upper},
		{Name: "upper_counts", Shape: []int64{int64(blocks)}, I32: cntU},
		{Name: "upper_offsets", Shape: []int64{N + 1}, I32: offsets},
	})
}

// Load reads an hnsw file. dim is the expected dimension (0 = any); params
// are the expected build parameters (nil = accept the file's).
func Load(path string, dim int, params map[string]any) (*Index, *vro.Header, error) {
	f, err := vro.Load(path, "hnsw", dim, params)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	h := f.Header
	n := h.N
	m, err := vro.IntParam(h.BuildParams, "m")
	if err != nil {
		return nil, nil, err
	}
	efC, err := vro.IntParam(h.BuildParams, "ef_construct")
	if err != nil {
		return nil, nil, err
	}
	if m < 2 || efC < 1 {
		return nil, nil, fmt.Errorf("hnsw: file has m=%d ef_construct=%d", m, efC)
	}
	ix := &Index{capN: n, dim: h.Dim, m: m, m0: 2 * m, efC: efC, seed: h.Seed, buildThreads: 1}
	ix.scPool.New = func() any { return newScratch(n) }
	if ix.vectors, err = f.Vectors(); err != nil {
		return nil, nil, err
	}
	bits, err := f.Tombstones()
	if err != nil {
		return nil, nil, err
	}
	ix.del = tombstone.FromBits(bits, n)
	var entry, offsets []int32
	if ix.levels, _, err = f.U8("levels"); err == nil {
		if entry, _, err = f.I32("entry"); err == nil {
			if ix.links0, _, err = f.I32("layer0_slots"); err == nil {
				if ix.cnt0, _, err = f.I32("layer0_counts"); err == nil {
					if ix.linksUpper, _, err = f.I32("upper_slots"); err == nil {
						if ix.cntUpper, _, err = f.I32("upper_counts"); err == nil {
							offsets, _, err = f.I32("upper_offsets")
						}
					}
				}
			}
		}
	}
	if err != nil {
		return nil, nil, err
	}
	if len(ix.levels) != n || len(entry) != 1 || len(ix.links0) != n*ix.m0 || len(ix.cnt0) != n || len(offsets) != n+1 {
		return nil, nil, fmt.Errorf("hnsw: section sizes do not match n=%d m=%d", n, m)
	}
	blocks := len(ix.cntUpper)
	if int(offsets[0]) != 0 || int(offsets[n]) != blocks || len(ix.linksUpper) != blocks*m {
		return nil, nil, fmt.Errorf("hnsw: upper_offsets end %d, upper_counts %d, upper_slots %d, m=%d", offsets[n], blocks, len(ix.linksUpper), m)
	}
	top := 0
	ix.upperOff = make([]int32, n)
	for i := 0; i < n; i++ {
		l := int(ix.levels[i])
		if int(offsets[i+1]-offsets[i]) != l {
			return nil, nil, fmt.Errorf("hnsw: node %d has level %d and %d upper blocks", i, l, offsets[i+1]-offsets[i])
		}
		ix.upperOff[i] = -1
		if l > 0 {
			ix.upperOff[i] = offsets[i]
		}
		top = max(top, l)
	}
	// Check every count and every edge, so that a search never reads out of range.
	checkList := func(sl []int32, c int32, limit int) error {
		if c < 0 || int(c) > limit {
			return fmt.Errorf("hnsw: count %d outside 0..%d", c, limit)
		}
		for _, e := range sl[:c] {
			if e < 0 || int(e) >= n {
				return fmt.Errorf("hnsw: edge to node %d outside 0..%d", e, n-1)
			}
		}
		return nil
	}
	for i := 0; i < n; i++ {
		if err := checkList(ix.links0[i*ix.m0:(i+1)*ix.m0], ix.cnt0[i], ix.m0); err != nil {
			return nil, nil, err
		}
	}
	for b := 0; b < blocks; b++ {
		if err := checkList(ix.linksUpper[b*m:(b+1)*m], ix.cntUpper[b], m); err != nil {
			return nil, nil, err
		}
	}
	e := entry[0]
	if n > 0 && (e < 0 || int(e) >= n || int(ix.levels[e]) != top) {
		return nil, nil, fmt.Errorf("hnsw: entry %d is not a node with the top level %d", e, top)
	}
	if n == 0 {
		e = -1
	}
	ix.setEntryTop(e, top)
	ix.count.Store(int64(n))
	// Purged: tombstoned nodes with no edge on any layer.
	if ix.del != nil {
		for i := 0; i < n; i++ {
			if !ix.del.Has(i) || ix.cnt0[i] != 0 {
				continue
			}
			empty := true
			for l := 1; l <= int(ix.levels[i]); l++ {
				if _, c := ix.slots(int32(i), l); *c != 0 {
					empty = false
				}
			}
			if empty {
				if ix.purged == nil {
					ix.purged = tombstone.New(n)
				}
				ix.purged.Add(i)
			}
		}
	}
	return ix, &h, nil
}
