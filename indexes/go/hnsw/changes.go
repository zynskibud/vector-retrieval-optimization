package hnsw

// Phase 5: deletes, updates and compaction (CONTRACT.md section 13.3).
//
// Delete sets bits in a tombstone bit set. A tombstoned node stays in the
// graph: Search still expands it and follows its edges, but the node never
// enters the result list (hnswlib markDelete). The upper-layer greedy descent
// ignores tombstones.
//
// Update(u) keeps the node ID and its level. For each updated node, in the
// order given: (1) on every layer 0..level(u), remove u from the lists of its
// current out-neighbors (a scan over each of those lists) and set u's own
// count to 0; (2) overwrite the vector; (3) run the insert procedure
// (Algorithm 1) for u from the entry point, with u excluded from every search
// (a node can still hold a stale edge to u, and u must not select itself).
// If u is the entry point, the descent starts from one of u's old neighbors on
// the highest layer where u had one. Edges from other nodes to u that were not
// in u's own lists stay; they now point to the new vector. After all updates,
// the repair pass (section 6.6) runs once.
//
// Compact, mode "rebuild" (the default): build a new graph from the live rows
// only, with the same m, ef_construct, seed and build threads. The live rows
// are copied to a new array in row order and the levels are drawn again for
// them in that order. Node j of the new graph is corpus row rowIDs[j]; Search
// maps node IDs back to row IDs.
//
// Compact, mode "repair" (in place, cheaper): for each live node x and each
// layer l <= level(x) whose list holds a tombstoned node, the new list is
// chosen with the heuristic (Algorithm 4, cap m or 2m) from x's live
// neighbors plus the live neighbors of each tombstoned neighbor on layer l.
// Then every tombstoned node loses all its out-edges and is marked purged;
// no live node points to it any more, so no search reaches it. If the entry
// point was tombstoned, the live node with the highest level (lowest row on
// ties) becomes the entry point. Last, the repair pass of section 6.6 runs
// and skips the purged nodes. The bit set stays (a purged row is still a
// tombstone), so IndexBytes counts it.
//
// Concurrency: the change functions run with no concurrent searches in
// Phase 5. They write slots and counts with the same atomic stores as Insert
// (slots before the count), but they take no locks and change the entry
// point without the entry lock.

import (
	"fmt"
	"sync/atomic"

	"vro/indexes/go/distance"
	"vro/indexes/go/tombstone"
)

// Delete marks every node i with mask[i] true as deleted. It returns the
// nodes newly deleted.
func Delete(ix *Index, mask []bool) (int, error) {
	if ix.rowIDs != nil {
		return 0, fmt.Errorf("hnsw: Delete after a Compact rebuild is not supported")
	}
	if ix.del == nil {
		ix.del = tombstone.New(ix.Len())
	}
	return tombstone.Apply(ix.del, mask), nil
}

// Update replaces the vectors of the nodes ids with vectors (row-major) and
// re-inserts each node, then runs the repair pass. The vectors are written
// into the array the index was built on.
func Update(ix *Index, ids []int64, vectors []float32) (int, error) {
	d := ix.dim
	if len(vectors) < len(ids)*d {
		return 0, fmt.Errorf("hnsw: Update got %d ids and %d floats", len(ids), len(vectors))
	}
	if ix.rowIDs != nil {
		return 0, fmt.Errorf("hnsw: Update after a Compact rebuild is not supported")
	}
	n := ix.Len()
	s := newScratch(ix.capN)
	for j, id := range ids {
		if id < 0 || id >= int64(n) {
			return 0, fmt.Errorf("hnsw: Update node %d outside 0..%d", id, n-1)
		}
		u := int32(id)
		entry, top := ix.entryTop()
		if entry == u {
			entry, top = ix.altEntry(u)
		}
		ix.detach(u)
		copy(ix.vectors[int(u)*d:int(u)*d+d], vectors[j*d:j*d+d])
		ix.reinsert(u, entry, top, s)
	}
	ix.updUnreach, ix.updAdded, ix.updAddedUnreach = ix.repair(s)
	return len(ids), nil
}

// Compact runs the rebuild or the in-place repair described above. It
// returns the index to search from now on: a new index for "rebuild", ix for
// "repair".
func Compact(ix *Index, mode string) (*Index, error) {
	switch mode {
	case "rebuild":
		return ix.rebuild()
	case "repair":
		ix.repairInPlace()
		return ix, nil
	}
	return nil, fmt.Errorf("hnsw: unknown compact mode %q", mode)
}

// ---------- update ----------

// altEntry returns a start node for the descent when u is the entry point:
// the first of u's neighbors on the highest layer where u has one, and that
// layer. It returns (-1, 0) if u has no neighbor.
func (ix *Index) altEntry(u int32) (int32, int) {
	for l := int(ix.levels[u]); l >= 0; l-- {
		sl, c := ix.slots(u, l)
		if *c > 0 {
			return sl[0], l
		}
	}
	return -1, 0
}

// detach removes u from the lists of its out-neighbors on every layer and
// empties u's own lists.
func (ix *Index) detach(u int32) {
	for l := 0; l <= int(ix.levels[u]); l++ {
		sl, c := ix.slots(u, l)
		for _, e := range sl[:*c] {
			ix.removeEdge(e, u, l)
		}
		atomic32Store(c, 0)
	}
}

// removeEdge deletes v from e's list on layer l, if present, and shifts the
// later slots down by one.
func (ix *Index) removeEdge(e, v int32, l int) {
	sl, c := ix.slots(e, l)
	n := int(*c)
	for j := 0; j < n; j++ {
		if sl[j] != v {
			continue
		}
		for k := j; k < n-1; k++ {
			atomic32Store(&sl[k], sl[k+1])
		}
		atomic32Store(c, int32(n-1))
		return
	}
}

// reinsert runs Algorithm 1 for the existing node u (level unchanged) from
// entry on layer top. u is excluded from every search. The entry point does
// not change: u keeps it if it had it.
func (ix *Index) reinsert(u, entry int32, top int, s *scratch) {
	if entry < 0 {
		return
	}
	s.skip = u
	defer func() { s.skip = -1 }()
	q := ix.vec(u)
	lvl := int(ix.levels[u])
	cur := item{dot(q, ix.vec(entry)), entry}
	for l := top; l > lvl; l-- {
		cur = ix.greedy(q, cur, l, s, false)
	}
	lmax := min(lvl, top)
	sels := make([][]item, lmax+1)
	for l := lmax; l >= 0; l-- {
		ix.searchLayer(q, cur, ix.efC, l, s, false)
		s.pool = append(s.pool[:0], s.res.a...)
		best := s.pool[0]
		for _, it := range s.pool[1:] {
			if better(it, best) {
				best = it
			}
		}
		s.sel = ix.selectHeuristic(s.pool, ix.m, s.sel)
		sels[l] = append([]item(nil), s.sel...)
		cur = best
	}
	for l := 0; l <= lmax; l++ {
		sl, c := ix.slots(u, l)
		for j, it := range sels[l] {
			atomic32Store(&sl[j], it.id)
		}
		atomic32Store(c, int32(len(sels[l])))
		for _, it := range sels[l] {
			ix.connect(it.id, u, it.s, l, ix.capOf(l), s, false)
		}
	}
}

// ---------- compact ----------

// rebuild builds a new index from the live rows (see the file comment).
func (ix *Index) rebuild() (*Index, error) {
	n, d := ix.Len(), ix.dim
	live := make([]int32, 0, n-ix.del.Count())
	for i := 0; i < n; i++ {
		if !ix.del.Has(i) {
			live = append(live, int32(i))
		}
	}
	vecs := make([]float32, 0, len(live)*d)
	for _, r := range live {
		vecs = append(vecs, ix.vec(r)...)
	}
	params := map[string]any{"m": int64(ix.m), "ef_construct": int64(ix.efC)}
	nix, err := Build(vecs, len(live), d, params, max(ix.buildThreads, 1), ix.seed)
	if err != nil {
		return nil, err
	}
	if len(live) < n {
		nix.rowIDs = live
	}
	return nix, nil
}

// repairInPlace is Compact mode "repair" (see the file comment).
func (ix *Index) repairInPlace() {
	if ix.del.Count() == 0 {
		ix.updUnreach, ix.updAdded, ix.updAddedUnreach = ix.repair(newScratch(ix.capN))
		return
	}
	n := ix.Len()
	seen := make([]uint32, n) // stamp per candidate, to drop duplicates
	stamp := uint32(0)
	var pool, out []item
	for x := int32(0); int(x) < n; x++ {
		if ix.del.Has(int(x)) {
			continue
		}
		vx := ix.vec(x)
		for l := 0; l <= int(ix.levels[x]); l++ {
			sl, c := ix.slots(x, l)
			cnt := int(*c)
			dirty := false
			for _, e := range sl[:cnt] {
				if ix.del.Has(int(e)) {
					dirty = true
					break
				}
			}
			if !dirty {
				continue
			}
			stamp++
			seen[x] = stamp
			pool = pool[:0]
			add := func(e int32) {
				if seen[e] == stamp || ix.del.Has(int(e)) {
					return
				}
				seen[e] = stamp
				pool = append(pool, item{dot(vx, ix.vec(e)), e})
			}
			for _, e := range sl[:cnt] {
				if !ix.del.Has(int(e)) {
					add(e)
					continue
				}
				dl, dc := ix.slots(e, l)
				for _, f := range dl[:*dc] {
					add(f)
				}
			}
			out = ix.selectHeuristic(pool, ix.capOf(l), out)
			for j, it := range out {
				atomic32Store(&sl[j], it.id)
			}
			atomic32Store(c, int32(len(out)))
		}
	}
	ix.purged = tombstone.New(n)
	for v := 0; v < n; v++ {
		if !ix.del.Has(v) {
			continue
		}
		ix.purged.Add(v)
		for l := 0; l <= int(ix.levels[v]); l++ {
			_, c := ix.slots(int32(v), l)
			atomic32Store(c, 0)
		}
	}
	if e, _ := ix.entryTop(); ix.del.Has(int(e)) {
		best, top := int32(-1), -1
		for v := 0; v < n; v++ {
			if !ix.del.Has(v) && int(ix.levels[v]) > top {
				best, top = int32(v), int(ix.levels[v])
			}
		}
		if best >= 0 {
			ix.setEntryTop(best, top)
		}
	}
	ix.updUnreach, ix.updAdded, ix.updAddedUnreach = ix.repair(newScratch(ix.capN))
}

// rowID returns the corpus row ID of node i.
func (ix *Index) rowID(i int32) int32 {
	if ix.rowIDs != nil {
		return ix.rowIDs[i]
	}
	return i
}

// admits reports whether node i may enter the result list of Search: it is
// not tombstoned and passes the filter mask (nil = no filter).
func (ix *Index) admits(i int32, mask []bool) bool {
	if ix.del.Has(int(i)) {
		return false
	}
	return mask == nil || mask[ix.rowID(i)]
}

// ChangeExtra returns the repair counts of the last Update or Compact repair.
func (ix *Index) ChangeExtra() map[string]any {
	return map[string]any{
		"change_repair_unreachable_before": ix.updUnreach,
		"change_repair_added":              ix.updAdded,
		"change_repair_added_unreachable":  ix.updAddedUnreach,
	}
}

func dot(a, b []float32) float32 { return distance.Dot(a, b) }

func atomic32Store(p *int32, v int32) { atomic.StoreInt32(p, v) }
