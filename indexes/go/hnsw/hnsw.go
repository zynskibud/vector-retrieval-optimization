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
// Concurrency (CONTRACT.md section 12). Search takes no lock. Every slot and
// every count is read and written with sync/atomic, and a writer stores the
// slots before the count. The entry point and top layer are one atomic word,
// changed only under the entry lock. Each Search takes its scratch buffers
// from a sync.Pool. Insert adds rows after the build (BuildCap reserves the
// room) while searches run.
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
	"vro/indexes/go/npy"
	"vro/indexes/go/splitmix"
)

// BuildDefaults lists every build parameter with its default. The values are
// int64 so that bench parses command-line values as integers.
var BuildDefaults = map[string]any{"m": int64(16), "ef_construct": int64(100)}

// SearchDefaults lists every search parameter with its default.
// filter names a mask filter_<name>.npy in DataDir (section 11); "none" = no filter.
var SearchDefaults = map[string]any{"ef": int64(64), "filter": "none"}

// DataDir is the data directory that holds filter_<name>.npy. bench sets it.
var DataDir string

// Index is the HNSW index.
type Index struct {
	trainS, addS float64
	// Search counters are atomic: several goroutines may call Search at once.
	dists      atomic.Int64
	filterRows atomic.Int64 // corpus rows passing the filter, summed over searches
	passCache  sync.Map     // *bool (first element of a mask) -> int64 pass count
	visitedSum atomic.Int64 // layer-0 nodes expanded by Search, summed over searches
	passedSum  atomic.Int64 // layer-0 nodes scored by Search that passed the filter

	// vectors holds capN rows. Rows 0..count-1 are in the graph. Rows at and
	// above count are never read until Insert links them.
	vectors []float32
	capN    int          // rows allocated: levels, slots and counts exist for capN rows
	count   atomic.Int64 // rows in the graph, published after each insert
	dim     int
	m, m0   int // max edges on layers >= 1, and on layer 0 (2m)
	efC     int

	levels []uint8 // level of each node

	links0 []int32 // n * m0 slots
	cnt0   []int32 // n counts

	upperOff   []int32 // first block of node i in links, or -1 when level 0
	linksUpper []int32 // blocks of m slots
	cntUpper   []int32 // one count per block

	// epTop packs the entry point (high 32 bits) and the top layer (low 32
	// bits), so Search reads both with one atomic load. It changes only under
	// epLock (write mode).
	epTop  atomic.Uint64
	epLock sync.RWMutex

	locks []sync.Mutex // one per node; used by the parallel build and by Insert

	unreachableBefore  int             // layer-0 BFS misses after a parallel build, before repair
	repairAdded        int             // edges added by the repair pass
	repairAddedUnreach int             // edges added by repair step B
	protected          map[uint64]bool // repair edges u->v (u<<32|v); never pruned
	buildThreads       int

	scPool sync.Pool // *scratch, one per concurrent Search

	insUnreach, insAdded, insAddedUnreach int // repair counts of the last Repair call
}

// entryTop returns the entry point and the top layer (one atomic load).
func (ix *Index) entryTop() (int32, int) {
	v := ix.epTop.Load()
	return int32(uint32(v >> 32)), int(uint32(v))
}

// setEntryTop publishes a new entry point and top layer. Callers hold epLock
// in write mode, or run with no other goroutine.
func (ix *Index) setEntryTop(e int32, top int) {
	ix.epTop.Store(uint64(uint32(e))<<32 | uint64(uint32(top)))
}

// Len returns the number of rows in the graph.
func (ix *Index) Len() int { return int(ix.count.Load()) }

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
	expand  int64 // nodes expanded (popped and their neighbors read)
	passed  int64 // scored nodes that passed the filter (filtered search only)
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
	// Lock-free readers (Search) may run while an insert changes this list.
	// Each slot and the count are loaded atomically. The writer stores the
	// slots before the count, so the list read here can be shorter or longer
	// than the current one, but every ID in it is a valid node.
	n := min(int(atomic.LoadInt32(c)), len(sl))
	buf = buf[:0]
	for j := 0; j < n; j++ {
		buf = append(buf, atomic.LoadInt32(&sl[j]))
	}
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
		s.expand++
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

// searchLayerFiltered is searchLayer on layer 0 with a filter mask
// (CONTRACT.md section 11.3, the hnswlib / FAISS IDSelector behavior). A node
// enters the result heap only if it passes the mask. Every visited node that
// would have entered the result heap without the filter still enters the
// candidate heap and is expanded, so the walk crosses failing regions. The
// stop rule is unchanged: stop when the best candidate is worse than the worst
// result and the result heap holds ef items. Used only by Search, never by
// the build.
func (ix *Index) searchLayerFiltered(q []float32, ep item, ef int, s *scratch, mask []bool) {
	s.nextGen()
	s.cand.a, s.res.a = s.cand.a[:0], s.res.a[:0]
	s.visited[ep.id] = s.gen
	s.cand.push(ep)
	if mask[ep.id] {
		s.res.push(ep)
		s.passed++
	}
	for len(s.cand.a) > 0 {
		c := s.cand.pop()
		if len(s.res.a) >= ef && better(s.res.a[0], c) {
			break
		}
		s.expand++
		s.nbuf = ix.neighbors(c.id, 0, s.nbuf, false)
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
				if mask[e] {
					s.passed++
					s.res.push(it)
					if len(s.res.a) > ef {
						s.res.pop()
					}
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

	entry, top := ix.entryTop()
	if locked {
		epLock.RLock()
		entry, top = ix.entryTop()
		epLock.RUnlock()
		if lvl > top {
			// Rare: hold the entry lock for the whole insertion, so the
			// entry point and top layer change atomically with it.
			epLock.Lock()
			defer epLock.Unlock()
			entry, top = ix.entryTop()
		}
	}
	if entry < 0 { // first node
		ix.setEntryTop(i, lvl)
		return
	}

	cur := item{distance.Dot(q, ix.vec(entry)), entry}
	for l := top; l > lvl; l-- {
		cur = ix.greedy(q, cur, l, s, locked)
	}
	// Select the neighbors top-down (each layer starts from the best result
	// of the layer above), then write the edges bottom-up. A connect on layer
	// l changes only layer-l lists, so the order of the writes does not change
	// the graph of a one-thread build. Bottom-up matters for concurrent
	// Search: once node i appears in a list on layer l, its own lists on
	// layers l-1..0 are complete, so a search that descends through i never
	// finds an empty layer-0 list.
	lmax := min(lvl, top)
	sels := make([][]item, lmax+1)
	for l := lmax; l >= 0; l-- {
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
		sels[l] = append([]item(nil), s.sel...)
		cur = best
	}
	for l := 0; l <= lmax; l++ {
		sel := sels[l]
		if locked {
			ix.locks[i].Lock()
		}
		sl, c := ix.slots(i, l)
		for j, it := range sel {
			atomic.StoreInt32(&sl[j], it.id)
		}
		atomic.StoreInt32(c, int32(len(sel))) // count after slots
		if locked {
			ix.locks[i].Unlock()
		}
		for _, it := range sel {
			ix.connect(it.id, i, it.s, l, ix.capOf(l), s, locked)
		}
	}
	if lvl > top {
		ix.setEntryTop(i, lvl)
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
		atomic.StoreInt32(&sl[n], i)
		atomic.StoreInt32(c, int32(n+1))
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
		atomic.StoreInt32(&sl[j], it.id)
	}
	atomic.StoreInt32(c, int32(len(out)))
}

// ---------- repair pass (CONTRACT 6.6, parallel build) ----------

// reach0Set marks the nodes reachable from the entry point on layer 0 by BFS.
func (ix *Index) reach0Set() []bool {
	n := ix.Len()
	seen := make([]bool, n)
	entry, _ := ix.entryTop()
	if n == 0 {
		return seen
	}
	queue := []int32{entry}
	seen[entry] = true
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
//
// It returns the BFS miss count before the repair, all edges added, and the
// edges added by step B. No other goroutine may use the index meanwhile.
func (ix *Index) repair(s *scratch) (unreachBefore, added, addedUnreach int) {
	n := ix.Len()
	if n <= 1 {
		return
	}
	entry, _ := ix.entryTop()
	unreachBefore = n - ix.reachable0()
	ix.protected = make(map[uint64]bool)
	defer func() { ix.protected = nil }()
	indeg := make([]int32, n)
	for pass := 0; pass < 3; pass++ {
		// Step A.
		clear(indeg)
		for i := 0; i < n; i++ {
			sl, c := ix.slots(int32(i), 0)
			for _, e := range sl[:*c] {
				indeg[e]++
			}
		}
		for v := int32(0); int(v) < n; v++ {
			if indeg[v] > 0 || v == entry {
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
				added++
			}
		}
		// Step B.
		seen := ix.reach0Set()
		foundB := false
		for v := int32(0); int(v) < n; v++ {
			if seen[v] {
				continue
			}
			foundB = true
			if u := ix.searchFrom(v, s, true); u >= 0 {
				ix.addKeep(u, v)
				added++
				addedUnreach++
			}
		}
		if !foundB {
			return
		}
	}
	return
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
	entry, top := ix.entryTop()
	ep := item{distance.Dot(q, ix.vec(entry)), entry}
	for l := top; l >= 1; l-- {
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
		atomic.StoreInt32(&sl[n], v)
		atomic.StoreInt32(c, int32(n+1))
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
		atomic.StoreInt32(&sl[j], it.id)
		j++
	}
	for _, id := range keep {
		if j < ix.m0 {
			atomic.StoreInt32(&sl[j], id)
			j++
		}
	}
	atomic.StoreInt32(c, int32(j))
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
	return BuildCap(vectors, n, n, dim, params, threads, seed)
}

// BuildCap builds the graph on the first n rows and allocates levels, slots
// and counts for capN >= n rows, so that Insert can add rows n..capN-1 later
// while searches run. vectors holds at least capN rows (row-major). The levels
// of all capN rows are drawn at build time, in row order from one PRNG, so an
// inserted row gets the same level as in a build on capN rows. The build of
// the first n rows is the same as Build on n rows.
func BuildCap(vectors []float32, n, capN, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
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
	if capN < n || len(vectors) < capN*dim {
		return nil, fmt.Errorf("hnsw: capacity %d rows, build %d rows, vectors hold %d rows", capN, n, len(vectors)/max(dim, 1))
	}
	start := time.Now()
	ix := &Index{vectors: vectors[:capN*dim], capN: capN, dim: dim, m: m, m0: 2 * m, efC: efC}
	ix.setEntryTop(-1, 0)
	ix.scPool.New = func() any { return newScratch(capN) }
	ix.levels = Levels(capN, m, seed)
	ix.links0 = make([]int32, capN*ix.m0)
	ix.cnt0 = make([]int32, capN)
	ix.upperOff = make([]int32, capN)
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
			s := newScratch(capN)
			for i := 0; i < n; i++ {
				ix.insert(int32(i), s, false, nil)
			}
		} else {
			ix.locks = make([]sync.Mutex, capN)
			// Row 0 first, so every other worker has an entry point.
			s0 := newScratch(capN)
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
							ix.insert(int32(i), s, true, &ix.epLock)
						}
					}
				}(newScratch(capN))
			}
			wg.Wait()
			ix.locks = nil
		}
		ix.count.Store(int64(n))
		ix.unreachableBefore, ix.repairAdded, ix.repairAddedUnreach = ix.repair(newScratch(capN))
	}
	ix.addS = time.Since(start).Seconds() // includes the repair pass
	ix.buildThreads = threads
	if capN > n {
		ix.locks = make([]sync.Mutex, capN) // for Insert
	}
	return ix, nil
}

// Insert adds rows to the graph while other goroutines call Search. ids must
// be the next rows in order: ids[0] == Len(), ids[j] == ids[0]+j, and below
// the capacity given to BuildCap. vectors holds len(ids) rows (row-major).
// Each row is inserted as in the parallel build (Algorithm 1 with the per-node
// locks and the entry lock) and then published: Len() grows by one.
// Only one goroutine may call Insert at a time. Searches take no lock.
func Insert(ix *Index, ids []int64, vectors []float32) error {
	d := ix.dim
	if len(vectors) < len(ids)*d {
		return fmt.Errorf("hnsw: Insert got %d ids and %d floats", len(ids), len(vectors))
	}
	if ix.locks == nil && len(ids) > 0 {
		return fmt.Errorf("hnsw: Insert needs an index built by BuildCap with capacity > rows")
	}
	s := ix.scPool.Get().(*scratch)
	defer ix.scPool.Put(s)
	for j, id := range ids {
		if want := int64(ix.Len()); id != want || id >= int64(ix.capN) {
			return fmt.Errorf("hnsw: Insert row %d, want row %d (capacity %d)", id, want, ix.capN)
		}
		dst := ix.vectors[int(id)*d : int(id)*d+d]
		src := vectors[j*d : j*d+d]
		if &dst[0] != &src[0] {
			// Row id is not reachable yet, so no Search reads dst now. The
			// atomic stores of the edges to id publish these writes.
			copy(dst, src)
		}
		ix.insert(int32(id), s, true, &ix.epLock)
		ix.count.Store(id + 1)
	}
	return nil
}

// Repair runs the repair pass (CONTRACT 6.6) on the current graph, for
// example after Insert. No other goroutine may use the index meanwhile. It
// returns the layer-0 BFS miss count before the repair and the edges added.
func Repair(ix *Index) (unreachBefore, added int) {
	u, a, b := ix.repair(newScratch(ix.capN))
	ix.insUnreach, ix.insAdded, ix.insAddedUnreach = u, a, b
	return u, a
}

// Search returns the k best ids and scores for one query, best first.
// It takes no lock and is safe to call from many goroutines at once, also
// while Insert runs.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	ids := make([]int64, k)
	scores := make([]float32, k)
	for i := range ids {
		ids[i], scores[i] = -1, float32(math.Inf(-1))
	}
	entry, top := ix.entryTop()
	if ix.Len() == 0 || entry < 0 {
		return ids, scores
	}
	ef, _ := intParam(params, "ef", 64)
	ef = max(ef, k)
	s := ix.scPool.Get().(*scratch)
	defer ix.scPool.Put(s)
	s.dists, s.expand, s.passed = 0, 0, 0
	cur := item{distance.Dot(query, ix.vec(entry)), entry}
	s.dists++
	// The upper-layer descent ignores the filter (section 11.3).
	for l := top; l >= 1; l-- {
		cur = ix.greedy(query, cur, l, s, false)
	}
	mask := filterMask(params)
	ix.filterRows.Add(ix.passCount(mask))
	if mask != nil {
		ix.searchLayerFiltered(query, cur, ef, s, mask)
	} else {
		d0 := s.dists
		ix.searchLayer(query, cur, ef, 0, s, false)
		s.passed = s.dists - d0 + 1 // every layer-0 node scored passes (entry included)
	}
	ix.visitedSum.Add(s.expand)
	ix.passedSum.Add(s.passed)
	res := append(s.pool[:0], s.res.a...)
	s.pool = res
	sort.Slice(res, func(a, b int) bool { return better(res[a], res[b]) })
	for i := 0; i < k && i < len(res); i++ {
		ids[i], scores[i] = int64(res[i].id), res[i].s
	}
	ix.dists.Add(s.dists)
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
	return edges*4 + int64(ix.Len())
}

// TopLayer returns the highest layer of the graph.
func (ix *Index) TopLayer() int { _, t := ix.entryTop(); return t }

// Entry returns the entry point node.
func (ix *Index) Entry() int32 { e, _ := ix.entryTop(); return e }

// NodesPerLayer returns the number of nodes on each layer 0..top.
func (ix *Index) NodesPerLayer() []int {
	top := ix.TopLayer()
	out := make([]int, top+1)
	for _, l := range ix.levels[:ix.Len()] {
		for j := 0; j <= int(l) && j <= top; j++ {
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
func (ix *Index) DistanceComputations() int64 { return ix.dists.Load() }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any {
	out := map[string]any{
		"top_layer":                 ix.TopLayer(),
		"entry_point":               ix.Entry(),
		"nodes_per_layer":           ix.NodesPerLayer(),
		"unreachable_before_repair": ix.unreachableBefore,
		"repair_added":              ix.repairAdded,
		"repair_added_unreachable":  ix.repairAddedUnreach,
		"build_threads":             ix.buildThreads,
	}
	if ix.locks != nil { // built with room for Insert
		out["capacity_rows"] = ix.capN
		out["repair_after_inserts_unreachable_before"] = ix.insUnreach
		out["repair_after_inserts_added"] = ix.insAdded
		out["repair_after_inserts_added_unreachable"] = ix.insAddedUnreach
	}
	return out
}

// SearchCounters returns cumulative per-search counters: visited = layer-0
// nodes expanded, filter_rows = corpus rows that pass the filter (n for
// none), passing_scored = layer-0 nodes scored that passed the filter.
func (ix *Index) SearchCounters() map[string]float64 {
	return map[string]float64{"visited": float64(ix.visitedSum.Load()), "passing_scored": float64(ix.passedSum.Load()), "filter_rows": float64(ix.filterRows.Load())}
}

// filterMask returns the mask for params["filter"], or nil for no filter.
// bench loads every mask before the build, so an error here is a program bug.
func filterMask(params map[string]any) []bool {
	name, _ := params["filter"].(string)
	m, err := npy.FilterMask(DataDir, name)
	if err != nil {
		panic(err)
	}
	return m
}

// passCount returns the number of rows among the first Len() that pass mask
// (Len() for no mask). It is computed once per mask and cached on the index,
// so the timed search does not scan the mask. With a mask, the count is of
// the rows present at the first search with it.
func (ix *Index) passCount(mask []bool) int64 {
	if mask == nil {
		return int64(ix.Len())
	}
	if c, ok := ix.passCache.Load(&mask[0]); ok {
		return c.(int64)
	}
	var c int64
	for _, b := range mask[:ix.Len()] {
		if b {
			c++
		}
	}
	ix.passCache.Store(&mask[0], c)
	return c
}
