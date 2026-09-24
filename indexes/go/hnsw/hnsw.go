// Package hnsw is the HNSW index (CONTRACT.md section 6.6), after Malkov and
// Yashunin 2018: Algorithm 1 (insert), 2 (search-layer), 4 (heuristic
// neighbor selection) and 5 (search).
//
// Recall floor (CONTRACT.md section 9): recall@10 >= 0.95 with ef=64 on the dev set at default params.
//
// Storage. Layer 0 is one flat int32 array with 2m slots per node plus one
// int32 count per node. The upper layers (1..top) share one flat int32 array:
// a node with level L > 0 owns L consecutive blocks of m slots, one block per
// layer 1..L, starting at upperOff[node]. Each block has its own count. No
// per-node slice is allocated after the build.
//
// Threads. With threads == 1, rows are inserted strictly in row order. With
// threads > 1, workers take rows in row order from an atomic counter and insert
// in parallel. Each node's neighbor lists are guarded by one sync.Mutex per
// node. A node whose level is above the current top layer holds the global
// entry lock in write mode for its whole insertion. The parallel build is not
// deterministic, and it can leave about 1% of nodes with no layer-0 in-edge:
// two near-duplicate rows inserted at the same time do not see each other,
// and their shared neighbors keep only one of them after the heuristic shrink.
// A repair pass after every build fixes these nodes (see repair).
package hnsw

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"vro/indexes/go/distance"
	"vro/indexes/go/splitmix"
)

// BuildDefaults lists every build parameter with its default. The values are
// int64 so that bench parses command-line values as integers.
var BuildDefaults = map[string]any{"m": int64(16), "ef_construct": int64(100)}

// SearchDefaults lists every search parameter with its default.
var SearchDefaults = map[string]any{"ef": int64(64)}

// Index is the HNSW index.
type Index struct {
	trainS, addS float64
	dists        int64

	vectors []float32
	n, dim  int
	m, m0   int // max edges on layers >= 1, and on layer 0 (2m)
	efC     int

	levels []uint8 // level of each node

	links0 []int32 // n * m0 slots
	cnt0   []int32 // n counts

	upperOff   []int32 // first block of node i in links, or -1 when level 0
	linksUpper []int32 // blocks of m slots
	cntUpper   []int32 // one count per block

	entry int32
	top   int

	locks []sync.Mutex // one per node; used only by the parallel build

	unreachableBefore  int             // layer-0 BFS misses after a parallel build, before repair
	repairAdded        int             // edges added by the repair pass
	repairAddedUnreach int             // edges added by repair step B
	protected          map[uint64]bool // repair edges u->v (u<<32|v); never pruned
	buildThreads       int

	sc *scratch // search scratch (search runs on one goroutine)
}

// ---------- parameters ----------

func intParam(p map[string]any, key string, def int) (int, error) {
	v, ok := p[key]
	if !ok {
		return def, nil
	}
	switch x := v.(type) {
	case int:
		return x, nil
	case int64:
		return int(x), nil
	case float64:
		return int(x), nil
	}
	return 0, fmt.Errorf("hnsw: parameter %s has type %T", key, v)
}

// Levels returns the level of each of n rows: one SplitMix64 seeded seed,
// next_f64 drawn in row order, level = floor(-ln(u) * (1/ln(m))).
func Levels(n, m int, seed uint64) []uint8 {
	rng := splitmix.New(seed)
	mL := 1 / math.Log(float64(m))
	out := make([]uint8, n)
	for i := range out {
		u := rng.NextF64()
		l := math.Floor(-math.Log(u) * mL)
		if l > 255 || math.IsInf(l, 1) {
			l = 255
		}
		out[i] = uint8(l)
	}
	return out
}

// ---------- heaps ----------

type item struct {
	s  float32
	id int32
}

// better reports whether a ranks above b: higher score, then lower ID.
func better(a, b item) bool {
	if a.s != b.s {
		return a.s > b.s
	}
	return a.id < b.id
}

// heap is a binary heap. With max=true the root is the best item, otherwise
// the root is the worst item.
type heap struct {
	a   []item
	max bool
}

func (h *heap) less(i, j int) bool {
	if h.max {
		return better(h.a[i], h.a[j])
	}
	return better(h.a[j], h.a[i])
}

func (h *heap) push(x item) {
	h.a = append(h.a, x)
	i := len(h.a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !h.less(i, p) {
			break
		}
		h.a[i], h.a[p] = h.a[p], h.a[i]
		i = p
	}
}

func (h *heap) pop() item {
	top := h.a[0]
	last := len(h.a) - 1
	h.a[0] = h.a[last]
	h.a = h.a[:last]
	i, n := 0, last
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		c := l
		if r := l + 1; r < n && h.less(r, l) {
			c = r
		}
		if !h.less(c, i) {
			break
		}
		h.a[i], h.a[c] = h.a[c], h.a[i]
		i = c
	}
	return top
}

// ---------- scratch ----------

// scratch holds the per-goroutine buffers of one search-layer call.
type scratch struct {
	visited []uint32 // generation stamps
	gen     uint32
	cand    heap // max-heap: best candidate at root
	res     heap // min-heap: worst result at root
	nbuf    []int32
	sel     []item
	pool    []item
	dists   int64
}

func newScratch(n int) *scratch {
	return &scratch{
		visited: make([]uint32, n),
		cand:    heap{max: true},
		res:     heap{max: false},
	}
}

func (s *scratch) nextGen() {
	s.gen++
	if s.gen == 0 {
		for i := range s.visited {
			s.visited[i] = 0
		}
		s.gen = 1
	}
}

// ---------- graph access ----------

func (ix *Index) vec(i int32) []float32 {
	d := ix.dim
	return ix.vectors[int(i)*d : int(i)*d+d]
}

// slots returns the slot array and the count pointer of node i on layer l.
func (ix *Index) slots(i int32, l int) ([]int32, *int32) {
	if l == 0 {
		o := int(i) * ix.m0
		return ix.links0[o : o+ix.m0], &ix.cnt0[i]
	}
	b := int(ix.upperOff[i]) + l - 1
	o := b * ix.m
	return ix.linksUpper[o : o+ix.m], &ix.cntUpper[b]
}

// neighbors copies the neighbor list of node i on layer l into buf.
func (ix *Index) neighbors(i int32, l int, buf []int32, locked bool) []int32 {
	if locked {
		ix.locks[i].Lock()
	}
	sl, c := ix.slots(i, l)
	buf = append(buf[:0], sl[:*c]...)
	if locked {
		ix.locks[i].Unlock()
	}
	return buf
}

// ---------- search-layer (Algorithm 2) ----------

// greedy is search-layer with ef = 1: it moves to the best neighbor until no
// neighbor improves the score.
func (ix *Index) greedy(q []float32, cur item, l int, s *scratch, locked bool) item {
	for changed := true; changed; {
		changed = false
		s.nbuf = ix.neighbors(cur.id, l, s.nbuf, locked)
		for _, e := range s.nbuf {
			sc := distance.Dot(q, ix.vec(e))
			s.dists++
			if c := (item{sc, e}); better(c, cur) {
				cur, changed = c, true
			}
		}
	}
	return cur
}

// searchLayer runs Algorithm 2 from entry ep with list size ef on layer l.
// The result stays in s.res (min-heap, size <= ef).
func (ix *Index) searchLayer(q []float32, ep item, ef, l int, s *scratch, locked bool) {
	s.nextGen()
	s.cand.a, s.res.a = s.cand.a[:0], s.res.a[:0]
	s.visited[ep.id] = s.gen
	s.cand.push(ep)
	s.res.push(ep)
	for len(s.cand.a) > 0 {
		c := s.cand.pop()
		if len(s.res.a) >= ef && better(s.res.a[0], c) {
			break
		}
		s.nbuf = ix.neighbors(c.id, l, s.nbuf, locked)
		for _, e := range s.nbuf {
			if s.visited[e] == s.gen {
				continue
			}
			s.visited[e] = s.gen
			sc := distance.Dot(q, ix.vec(e))
			s.dists++
			it := item{sc, e}
			if len(s.res.a) < ef || better(it, s.res.a[0]) {
				s.cand.push(it)
				s.res.push(it)
				if len(s.res.a) > ef {
					s.res.pop()
				}
			}
		}
	}
}

// ---------- heuristic (Algorithm 4) ----------

// selectHeuristic keeps up to limit items of cands (scores relative to the
// base node), sorted best first. An item is kept only if it scores higher
// with the base than with every item already kept. extendCandidates = false,
// keepPrunedConnections = false. The result is written to out.
func (ix *Index) selectHeuristic(cands []item, limit int, out []item) []item {
	sort.Slice(cands, func(a, b int) bool { return better(cands[a], cands[b]) })
	out = out[:0]
	for _, c := range cands {
		if len(out) >= limit {
			break
		}
		vc := ix.vec(c.id)
		keep := true
		for _, r := range out {
			if distance.Dot(vc, ix.vec(r.id)) > c.s {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}

// ---------- insert (Algorithm 1) ----------

func (ix *Index) insert(i int32, s *scratch, locked bool, epLock *sync.RWMutex) {
	lvl := int(ix.levels[i])
	q := ix.vec(i)

	var entry int32
	var top int
	if locked {
		epLock.RLock()
		entry, top = ix.entry, ix.top
		epLock.RUnlock()
		if lvl > top {
			// Rare: hold the entry lock for the whole insertion, so the
			// entry point and top layer change atomically with it.
			epLock.Lock()
			defer epLock.Unlock()
			entry, top = ix.entry, ix.top
		}
	} else {
		entry, top = ix.entry, ix.top
	}
	if entry < 0 { // first node
		ix.entry, ix.top = i, lvl
		return
	}

	cur := item{distance.Dot(q, ix.vec(entry)), entry}
	for l := top; l > lvl; l-- {
		cur = ix.greedy(q, cur, l, s, locked)
	}
	for l := min(lvl, top); l >= 0; l-- {
		ix.searchLayer(q, cur, ix.efC, l, s, locked)
		s.pool = append(s.pool[:0], s.res.a...)
		// Next layer starts from the best result.
		best := s.pool[0]
		for _, it := range s.pool[1:] {
			if better(it, best) {
				best = it
			}
		}
		// The new node selects m neighbors on every layer (paper M). The cap
		// of a list (m, or 2m on layer 0; paper M_max0) applies in connect.
		s.sel = ix.selectHeuristic(s.pool, ix.m, s.sel)
		sel := append([]item(nil), s.sel...)

		if locked {
			ix.locks[i].Lock()
		}
		sl, c := ix.slots(i, l)
		for j, it := range sel {
			sl[j] = it.id
		}
		*c = int32(len(sel))
		if locked {
			ix.locks[i].Unlock()
		}
		for _, it := range sel {
			ix.connect(it.id, i, it.s, l, ix.capOf(l), s, locked)
		}
		cur = best
	}
	if lvl > top {
		ix.entry, ix.top = i, lvl
	}
}

func (ix *Index) capOf(l int) int {
	if l == 0 {
		return ix.m0
	}
	return ix.m
}

// connect adds edge e -> i on layer l. sim is q(i) . x(e). If e is full, its
// list is shrunk with the heuristic over its current neighbors plus i.
func (ix *Index) connect(e, i int32, sim float32, l, limit int, s *scratch, locked bool) {
	if locked {
		ix.locks[e].Lock()
		defer ix.locks[e].Unlock()
	}
	sl, c := ix.slots(e, l)
	n := int(*c)
	if n < limit {
		sl[n] = i
		*c++
		return
	}
	ve := ix.vec(e)
	pool := make([]item, 0, n+1)
	for _, nb := range sl[:n] {
		pool = append(pool, item{distance.Dot(ve, ix.vec(nb)), nb})
	}
	pool = append(pool, item{sim, i})
	out := ix.selectHeuristic(pool, limit, make([]item, 0, limit))
	for j, it := range out {
		sl[j] = it.id
	}
	*c = int32(len(out))
}

// ---------- repair pass (CONTRACT 6.6, parallel build) ----------

// reach0Set marks the nodes reachable from the entry point on layer 0 by BFS.
func (ix *Index) reach0Set() []bool {
	seen := make([]bool, ix.n)
	if ix.n == 0 {
		return seen
	}
	queue := []int32{ix.entry}
	seen[ix.entry] = true
	for h := 0; h < len(queue); h++ {
		sl, c := ix.slots(queue[h], 0)
		for _, e := range sl[:*c] {
			if !seen[e] {
				seen[e] = true
				queue = append(queue, e)
			}
		}
	}
	return seen
}

// reachable0 counts the nodes reachable from the entry point on layer 0.
func (ix *Index) reachable0() int {
	c := 0
	for _, b := range ix.reach0Set() {
		if b {
			c++
		}
	}
	return c
}

// repair runs after every build (CONTRACT 6.6, Parallel build). One pass:
//
//   - Step A, in-degree: for each node v with zero layer-0 in-degree, in row
//     order, skipping the entry point: u = the nearest node in v's own
//     layer-0 list, or, if v has no out-edges, the nearest result of a
//     search-layer from the entry point with ef_construct. Add u -> v.
//   - Step B, reachability: directed BFS from the entry point on layer 0. For
//     each node v that the BFS did not reach, in row order: search-layer from
//     the entry point with ef_construct (it visits reachable nodes only).
//     Among the results, nearest first, u = the first node with a free
//     layer-0 slot, else the nearest. Add u -> v.
//
// Every repair edge is protected: when u's list is over its cap, the
// heuristic shrink never prunes v or any earlier repair edge (addKeep).
// The pass repeats while step B found an unreachable node, at most 3 passes.
func (ix *Index) repair(s *scratch) {
	if ix.n <= 1 {
		return
	}
	ix.unreachableBefore = ix.n - ix.reachable0()
	ix.protected = make(map[uint64]bool)
	defer func() { ix.protected = nil }()
	indeg := make([]int32, ix.n)
	for pass := 0; pass < 3; pass++ {
		// Step A.
		clear(indeg)
		for i := 0; i < ix.n; i++ {
			sl, c := ix.slots(int32(i), 0)
			for _, e := range sl[:*c] {
				indeg[e]++
			}
		}
		for v := int32(0); int(v) < ix.n; v++ {
			if indeg[v] > 0 || v == ix.entry {
				continue
			}
			var u int32
			if _, c := ix.slots(v, 0); *c > 0 {
				u = ix.nearestInList(v)
			} else {
				u = ix.searchFrom(v, s, false)
			}
			if u >= 0 {
				ix.addKeep(u, v)
				ix.repairAdded++
			}
		}
		// Step B.
		seen := ix.reach0Set()
		foundB := false
		for v := int32(0); int(v) < ix.n; v++ {
			if seen[v] {
				continue
			}
			foundB = true
			if u := ix.searchFrom(v, s, true); u >= 0 {
				ix.addKeep(u, v)
				ix.repairAdded++
				ix.repairAddedUnreach++
			}
		}
		if !foundB {
			return
		}
	}
}

// nearestInList returns the best-scoring node in v's own layer-0 list.
func (ix *Index) nearestInList(v int32) int32 {
	q := ix.vec(v)
	best := item{id: -1}
	sl, c := ix.slots(v, 0)
	for _, e := range sl[:*c] {
		it := item{distance.Dot(q, ix.vec(e)), e}
		if best.id < 0 || better(it, best) {
			best = it
		}
	}
	return best.id
}

// searchFrom runs a search for v from the entry point (greedy descent, then
// search-layer on layer 0 with ef_construct) and returns the nearest result
// other than v. With preferFree, it returns the nearest result whose layer-0
// list has a free slot, and the nearest result only if none has one.
func (ix *Index) searchFrom(v int32, s *scratch, preferFree bool) int32 {
	q := ix.vec(v)
	ep := item{distance.Dot(q, ix.vec(ix.entry)), ix.entry}
	for l := ix.top; l >= 1; l-- {
		ep = ix.greedy(q, ep, l, s, false)
	}
	ix.searchLayer(q, ep, ix.efC, 0, s, false)
	res := append(s.pool[:0], s.res.a...)
	s.pool = res
	sort.Slice(res, func(a, b int) bool { return better(res[a], res[b]) })
	nearest := int32(-1)
	for _, it := range res {
		if it.id == v {
			continue
		}
		if nearest < 0 {
			nearest = it.id
			if !preferFree {
				return nearest
			}
		}
		if ix.cnt0[it.id] < int32(ix.m0) {
			return it.id
		}
	}
	return nearest
}

func edgeKey(u, v int32) uint64 { return uint64(uint32(u))<<32 | uint64(uint32(v)) }

// addKeep adds the protected edge u -> v on layer 0. If u's list is over its
// cap, the heuristic shrinks it, but v and every earlier repair edge of u
// are kept.
func (ix *Index) addKeep(u, v int32) {
	ix.protected[edgeKey(u, v)] = true
	sl, c := ix.slots(u, 0)
	n := int(*c)
	if n < ix.m0 {
		sl[n] = v
		*c++
		return
	}
	vu := ix.vec(u)
	keep := []int32{v}
	pool := make([]item, 0, n)
	for _, nb := range sl[:n] {
		if ix.protected[edgeKey(u, nb)] {
			keep = append(keep, nb)
		} else {
			pool = append(pool, item{distance.Dot(vu, ix.vec(nb)), nb})
		}
	}
	out := ix.selectHeuristic(pool, max(ix.m0-len(keep), 0), make([]item, 0, ix.m0))
	j := 0
	for _, it := range out {
		sl[j] = it.id
		j++
	}
	for _, id := range keep {
		if j < ix.m0 {
			sl[j] = id
			j++
		}
	}
	*c = int32(j)
}

// parallelChunk is the number of consecutive rows a worker claims at once
// (CONTRACT 6.6). Near-duplicate rows are often consecutive in this corpus.
// Inside one chunk one worker inserts them in row order, so the later row
// sees the earlier one. Measured on 20000 dev rows, 10 threads, with the
// full repair: one row per claim gave recall@10 (ef=64) 0.9612 against
// 0.9776 for threads=1; 64 rows per claim gave 0.9761.
const parallelChunk = 64

// ---------- public interface ----------

// Build runs train and add. vectors is row-major (n, dim).
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	m, err := intParam(params, "m", 16)
	if err != nil {
		return nil, err
	}
	efC, err := intParam(params, "ef_construct", 100)
	if err != nil {
		return nil, err
	}
	if m < 2 || efC < 1 {
		return nil, fmt.Errorf("hnsw: need m >= 2 and ef_construct >= 1, got m=%d ef_construct=%d", m, efC)
	}
	start := time.Now()
	ix := &Index{vectors: vectors, n: n, dim: dim, m: m, m0: 2 * m, efC: efC, entry: -1}
	ix.levels = Levels(n, m, seed)
	ix.links0 = make([]int32, n*ix.m0)
	ix.cnt0 = make([]int32, n)
	ix.upperOff = make([]int32, n)
	blocks := 0
	for i, l := range ix.levels {
		if l > 0 {
			ix.upperOff[i] = int32(blocks)
			blocks += int(l)
		} else {
			ix.upperOff[i] = -1
		}
	}
	ix.linksUpper = make([]int32, blocks*m)
	ix.cntUpper = make([]int32, blocks)

	if n > 0 {
		if threads <= 1 {
			s := newScratch(n)
			for i := 0; i < n; i++ {
				ix.insert(int32(i), s, false, nil)
			}
		} else {
			ix.locks = make([]sync.Mutex, n)
			var epLock sync.RWMutex
			// Row 0 first, so every other worker has an entry point.
			s0 := newScratch(n)
			ix.insert(0, s0, false, nil)
			// Workers claim chunks of consecutive rows; see parallelChunk.
			chunk := int64(parallelChunk)
			var next atomic.Int64
			next.Store(0)
			var wg sync.WaitGroup
			for w := 0; w < threads; w++ {
				wg.Add(1)
				go func(s *scratch) {
					defer wg.Done()
					for {
						lo := (next.Add(1) - 1) * chunk
						if lo >= int64(n) {
							return
						}
						for i := max(lo, 1); i < min(lo+chunk, int64(n)); i++ {
							ix.insert(int32(i), s, true, &epLock)
						}
					}
				}(newScratch(n))
			}
			wg.Wait()
			ix.locks = nil
		}
		ix.repair(newScratch(n))
	}
	ix.addS = time.Since(start).Seconds() // includes the repair pass
	ix.buildThreads = threads
	ix.sc = newScratch(n)
	return ix, nil
}

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	ids := make([]int64, k)
	scores := make([]float32, k)
	for i := range ids {
		ids[i], scores[i] = -1, float32(math.Inf(-1))
	}
	if ix.n == 0 || ix.entry < 0 {
		return ids, scores
	}
	ef, _ := intParam(params, "ef", 64)
	ef = max(ef, k)
	s := ix.sc
	s.dists = 0
	cur := item{distance.Dot(query, ix.vec(ix.entry)), ix.entry}
	s.dists++
	for l := ix.top; l >= 1; l-- {
		cur = ix.greedy(query, cur, l, s, false)
	}
	ix.searchLayer(query, cur, ef, 0, s, false)
	res := append(s.pool[:0], s.res.a...)
	s.pool = res
	sort.Slice(res, func(a, b int) bool { return better(res[a], res[b]) })
	for i := 0; i < k && i < len(res); i++ {
		ids[i], scores[i] = int64(res[i].id), res[i].s
	}
	ix.dists += s.dists
	return ids, scores
}

// IndexBytes returns the computed memory of the index structure (section 4):
// number of edges x 4 bytes over all layers, plus 1 byte per node for its level.
func IndexBytes(ix *Index) int64 {
	var edges int64
	for _, c := range ix.cnt0 {
		edges += int64(c)
	}
	for _, c := range ix.cntUpper {
		edges += int64(c)
	}
	return edges*4 + int64(ix.n)
}

// TopLayer returns the highest layer of the graph.
func (ix *Index) TopLayer() int { return ix.top }

// Entry returns the entry point node.
func (ix *Index) Entry() int32 { return ix.entry }

// NodesPerLayer returns the number of nodes on each layer 0..top.
func (ix *Index) NodesPerLayer() []int {
	out := make([]int, ix.top+1)
	for _, l := range ix.levels {
		for j := 0; j <= int(l) && j <= ix.top; j++ {
			out[j]++
		}
	}
	return out
}

// TrainSeconds returns the train time of the build.
func (ix *Index) TrainSeconds() float64 { return ix.trainS }

// AddSeconds returns the add time of the build.
func (ix *Index) AddSeconds() float64 { return ix.addS }

// DistanceComputations returns the total dot products computed by Search so far.
func (ix *Index) DistanceComputations() int64 { return ix.dists }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any {
	return map[string]any{
		"top_layer":                 ix.top,
		"entry_point":               ix.entry,
		"nodes_per_layer":           ix.NodesPerLayer(),
		"unreachable_before_repair": ix.unreachableBefore,
		"repair_added":              ix.repairAdded,
		"repair_added_unreachable":  ix.repairAddedUnreach,
		"build_threads":             ix.buildThreads,
	}
}

// SearchCounters returns cumulative per-search counters. HNSW has none beyond
// distance computations.
func (ix *Index) SearchCounters() map[string]float64 { return map[string]float64{} }
