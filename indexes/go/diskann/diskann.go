// Package diskann is the DiskANN index (CONTRACT.md section 6.7), after
// Subramanya et al. 2019: the Vamana graph (greedy search, robust prune with
// alpha, two passes), PQ codes in RAM, and full vectors plus out-edges on disk.
//
// Recall floor (CONTRACT.md section 9): recall@10 >= 0.90 with l=100, for metric=ip and metric=l2 on the dev set at default params.
//
// Build.
//  1. Entry point = the row with the highest dot product with the mean row.
//  2. Init graph: r random out-edges per node (SplitMix64 seeded seed,
//     next_below(N), row order, no self, no repeats).
//  3. Two passes in row order, alpha = 1.0 then alpha. The graph build uses
//     full-precision vectors. Robust prune compares distances
//     d(a, b) = 2 - 2 a.b, the squared Euclidean distance of unit vectors, so
//     the alpha test is the paper's test on squared distances.
//  4. PQ codebooks (m = pq_m, metric, seed + j, no normalization) on all rows,
//     then encode all rows. The codebook training is train_s.
//  5. Write <OutPath>.diskann: one 4096-byte record per node (vector, then r
//     int32 out-edges, -1 = empty, zero padding). Then drop the corpus.
//
// Threads. With threads > 1, workers claim chunks of 64 consecutive rows from
// an atomic counter. One sync.Mutex per node guards its out-edge list. A
// worker holds at most one node lock at a time, so the graph stays valid (every
// list holds 1..r distinct ids) and no deadlock is possible. The parallel
// build is not deterministic. With threads == 1, the build follows the row
// order exactly.
//
// Search. Beam search over the on-disk graph, PQ table scores for the
// candidate list, exact rerank of the top `rerank` candidates with the full
// vectors of their records. The records are read through a memory map
// (io=mmap) or with pread on a file descriptor with the OS cache disabled
// (io=nocache: F_NOCACHE on darwin, O_DIRECT on linux).
package diskann

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"vro/indexes/go/distance"
	"vro/indexes/go/kmeans"
	"vro/indexes/go/splitmix"
)

// ksub is the number of centroids per PQ codebook (8 bits).
const ksub = 256

// pageSize is the record alignment of the on-disk file (section 6.7.1).
const pageSize = 4096

// chunk is the number of consecutive rows a build worker claims at once.
const chunk = 64

// pqIters is the k-means iteration count per codebook (pq default, 6.4).
const pqIters = 20

// BuildDefaults lists every build parameter with its default. Numbers are
// int64 and float64 so that bench parses command-line values with that type.
var BuildDefaults = map[string]any{"r": int64(64), "l_build": int64(100), "alpha": float64(1.2), "pq_m": int64(48), "metric": "ip"}

// SearchDefaults lists every search parameter with its default.
var SearchDefaults = map[string]any{"l": int64(100), "beam": int64(4), "rerank": int64(100), "io": "mmap"}

// OutPath is the --out JSON path. bench sets it before Build, so Build can
// write the record file next to the JSON (section 6.7 step 5).
var OutPath string

// Index is the DiskANN index. It holds no full vectors: only the PQ codes,
// the codebooks, the entry point, and the path of the record file.
type Index struct {
	trainS, addS float64
	dists        int64

	n, dim, r int
	m, dsub   int
	l2        bool
	entry     int32
	recBytes  int
	path      string
	diskBytes int64
	meanDeg   float64
	threads   int

	codebooks []float32 // (m, 256, dsub)
	codes     []uint8   // (n, m)

	// I/O state, opened on first use of each mode.
	mapped []byte
	ncFile *os.File

	diskReads, bytesRead float64

	sc *searchScratch
}

// ---------- parameters ----------

func intParam(p map[string]any, key string) (int, error) {
	v, ok := p[key]
	if !ok {
		if d, ok := BuildDefaults[key]; ok {
			v = d
		} else {
			v = SearchDefaults[key]
		}
	}
	switch x := v.(type) {
	case int:
		return x, nil
	case int64:
		return int(x), nil
	case float64:
		if x != math.Trunc(x) {
			return 0, fmt.Errorf("diskann: parameter %s needs an integer, got %v", key, x)
		}
		return int(x), nil
	}
	return 0, fmt.Errorf("diskann: parameter %s has type %T", key, v)
}

func floatParam(p map[string]any, key string) (float64, error) {
	v, ok := p[key]
	if !ok {
		v = BuildDefaults[key]
	}
	switch x := v.(type) {
	case float64:
		return x, nil
	case int64:
		return float64(x), nil
	case int:
		return float64(x), nil
	}
	return 0, fmt.Errorf("diskann: parameter %s has type %T", key, v)
}

func stringParam(p map[string]any, key string, defaults map[string]any) string {
	if v, ok := p[key].(string); ok {
		return v
	}
	return defaults[key].(string)
}

// ---------- candidate items ----------

// item is a node id with its score relative to a base vector (query or node).
type item struct {
	s  float32
	id int32
}

// better reports whether a ranks above b: higher score, then lower id.
func better(a, b item) bool {
	if a.s != b.s {
		return a.s > b.s
	}
	return a.id < b.id
}

// cand is one entry of a sorted candidate list.
type cand struct {
	item
	expanded bool
}

// insertSorted inserts c into list (sorted best first, capacity cap) and
// returns the new list. If the list is full and c is not better than the
// last entry, the list does not change.
func insertSorted(list []cand, c cand, capacity int) []cand {
	if len(list) >= capacity && !better(c.item, list[len(list)-1].item) {
		return list
	}
	pos := sort.Search(len(list), func(k int) bool { return better(c.item, list[k].item) })
	if len(list) < capacity {
		list = append(list, cand{})
	}
	copy(list[pos+1:], list[pos:len(list)-1])
	list[pos] = c
	return list
}

// stamps is a visited set with generation stamps, so a reset is O(1).
type stamps struct {
	mark []uint32
	gen  uint32
}

func newStamps(n int) stamps { return stamps{mark: make([]uint32, n)} }

func (s *stamps) next() {
	s.gen++
	if s.gen == 0 {
		clear(s.mark)
		s.gen = 1
	}
}

// ---------- build ----------

// graph is the in-RAM Vamana graph during the build.
type graph struct {
	vectors []float32
	n, dim  int
	r       int
	adj     []int32 // n * r slots
	cnt     []int32
	locks   []sync.Mutex
	locked  bool
	entry   int32
}

func (g *graph) vec(i int32) []float32 {
	return g.vectors[int(i)*g.dim : int(i)*g.dim+g.dim]
}

// neighbors copies the out-edges of node i into buf.
func (g *graph) neighbors(i int32, buf []int32) []int32 {
	if g.locked {
		g.locks[i].Lock()
	}
	o := int(i) * g.r
	buf = append(buf[:0], g.adj[o:o+int(g.cnt[i])]...)
	if g.locked {
		g.locks[i].Unlock()
	}
	return buf
}

// buildScratch holds the buffers of one build worker.
type buildScratch struct {
	seen    stamps // greedy search visited set
	dedup   stamps // robust prune candidate dedup
	list    []cand
	visited []item // V: expanded nodes with their score to the base
	nbuf    []int32
	pool    []item
	removed []bool
	out     []int32
}

func newBuildScratch(n int) *buildScratch {
	return &buildScratch{seen: newStamps(n), dedup: newStamps(n)}
}

// greedySearch runs Vamana's GreedySearch from the entry point for vector q
// with list size l. It fills s.visited with every expanded node (the set V).
func (g *graph) greedySearch(q []float32, l int, s *buildScratch) {
	s.seen.next()
	s.visited = s.visited[:0]
	e := g.entry
	s.seen.mark[e] = s.seen.gen
	s.list = append(s.list[:0], cand{item: item{distance.Dot(q, g.vec(e)), e}})
	for {
		k := -1
		for idx := range s.list {
			if !s.list[idx].expanded {
				k = idx
				break
			}
		}
		if k < 0 {
			return
		}
		s.list[k].expanded = true
		c := s.list[k].item
		s.visited = append(s.visited, c)
		s.nbuf = g.neighbors(c.id, s.nbuf)
		for _, nb := range s.nbuf {
			if s.seen.mark[nb] == s.seen.gen {
				continue
			}
			s.seen.mark[nb] = s.seen.gen
			s.list = insertSorted(s.list, cand{item: item{distance.Dot(q, g.vec(nb)), nb}}, l)
		}
	}
}

// robustPrune selects at most r out-neighbors of p from pool (scores are
// dots with p). pool must hold distinct ids and no p. With d(a,b) = 2 - 2 a.b,
// a candidate c is dropped when alpha * d(best, c) <= d(p, c), for each
// selected best. The result is written to s.out, best first.
func (g *graph) robustPrune(p int32, pool []item, alpha float32, s *buildScratch) []int32 {
	sort.Slice(pool, func(a, b int) bool { return better(pool[a], pool[b]) })
	if cap(s.removed) < len(pool) {
		s.removed = make([]bool, len(pool))
	}
	removed := s.removed[:len(pool)]
	clear(removed)
	s.out = s.out[:0]
	for a := range pool {
		if removed[a] {
			continue
		}
		best := pool[a]
		s.out = append(s.out, best.id)
		if len(s.out) >= g.r {
			break
		}
		vb := g.vec(best.id)
		for b := a + 1; b < len(pool); b++ {
			if removed[b] {
				continue
			}
			dBC := 2 - 2*distance.Dot(vb, g.vec(pool[b].id))
			dPC := 2 - 2*pool[b].s
			if alpha*dBC <= dPC {
				removed[b] = true
			}
		}
	}
	return s.out
}

// setNeighbors writes list as the out-edges of node i. Caller holds i's lock.
func (g *graph) setNeighbors(i int32, list []int32) {
	o := int(i) * g.r
	copy(g.adj[o:o+g.r], list)
	g.cnt[i] = int32(len(list))
}

// insertNode runs one Vamana step for row i with the given alpha.
func (g *graph) insertNode(i int32, lBuild int, alpha float32, s *buildScratch) {
	q := g.vec(i)
	g.greedySearch(q, lBuild, s)

	// Pool = V union N_out(i), without i.
	s.dedup.next()
	s.pool = s.pool[:0]
	s.dedup.mark[i] = s.dedup.gen
	for _, v := range s.visited {
		if s.dedup.mark[v.id] != s.dedup.gen {
			s.dedup.mark[v.id] = s.dedup.gen
			s.pool = append(s.pool, v)
		}
	}
	if g.locked {
		g.locks[i].Lock()
	}
	o := int(i) * g.r
	for _, nb := range g.adj[o : o+int(g.cnt[i])] {
		if s.dedup.mark[nb] != s.dedup.gen {
			s.dedup.mark[nb] = s.dedup.gen
			s.pool = append(s.pool, item{distance.Dot(q, g.vec(nb)), nb})
		}
	}
	out := g.robustPrune(i, s.pool, alpha, s)
	if len(out) > 0 {
		g.setNeighbors(i, out)
	}
	newOut := append([]int32(nil), g.adj[o:o+int(g.cnt[i])]...)
	if g.locked {
		g.locks[i].Unlock()
	}

	// Back edges j -> i, with a prune of j when it overflows.
	for _, j := range newOut {
		g.addBackEdge(j, i, alpha, s)
	}
}

// addBackEdge adds edge j -> i. If j then holds more than r edges, it is
// pruned with robust prune over N_out(j) union {i}.
func (g *graph) addBackEdge(j, i int32, alpha float32, s *buildScratch) {
	if g.locked {
		g.locks[j].Lock()
		defer g.locks[j].Unlock()
	}
	o := int(j) * g.r
	c := int(g.cnt[j])
	for _, nb := range g.adj[o : o+c] {
		if nb == i {
			return
		}
	}
	if c < g.r {
		g.adj[o+c] = i
		g.cnt[j]++
		return
	}
	vj := g.vec(j)
	s.pool = s.pool[:0]
	for _, nb := range g.adj[o : o+c] {
		s.pool = append(s.pool, item{distance.Dot(vj, g.vec(nb)), nb})
	}
	s.pool = append(s.pool, item{distance.Dot(vj, g.vec(i)), i})
	out := g.robustPrune(j, s.pool, alpha, s)
	g.setNeighbors(j, out)
}

// medoid returns the row with the highest dot product with the mean row.
// Ties go to the lower row.
func medoid(vectors []float32, n, dim int) int32 {
	sum := make([]float64, dim)
	for i := 0; i < n; i++ {
		row := vectors[i*dim : (i+1)*dim]
		for d, x := range row {
			sum[d] += float64(x)
		}
	}
	mean := make([]float32, dim)
	for d := range sum {
		mean[d] = float32(sum[d] / float64(n))
	}
	best, bestS := int32(0), float32(math.Inf(-1))
	for i := 0; i < n; i++ {
		if s := distance.Dot(mean, vectors[i*dim:(i+1)*dim]); s > bestS {
			best, bestS = int32(i), s
		}
	}
	return best
}

// initGraph gives every node r random distinct out-edges, no self-edge.
func (g *graph) initGraph(seed uint64) {
	rng := splitmix.New(seed)
	n := uint64(g.n)
	for i := 0; i < g.n; i++ {
		o := i * g.r
		c := 0
		for c < g.r {
			j := int32(rng.NextBelow(n))
			if int(j) == i {
				continue
			}
			dup := false
			for _, x := range g.adj[o : o+c] {
				if x == j {
					dup = true
					break
				}
			}
			if dup {
				continue
			}
			g.adj[o+c] = j
			c++
		}
		g.cnt[i] = int32(c)
	}
}

// pass runs one Vamana pass over all rows with the given alpha.
func (g *graph) pass(lBuild int, alpha float32, threads int) {
	if threads <= 1 {
		g.locked = false
		s := newBuildScratch(g.n)
		for i := 0; i < g.n; i++ {
			g.insertNode(int32(i), lBuild, alpha, s)
		}
		return
	}
	g.locked = true
	var next atomic.Int64
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := newBuildScratch(g.n)
			for {
				lo := int(next.Add(chunk)) - chunk
				if lo >= g.n {
					return
				}
				for i := lo; i < min(lo+chunk, g.n); i++ {
					g.insertNode(int32(i), lBuild, alpha, s)
				}
			}
		}()
	}
	wg.Wait()
}

// parallelFor runs f(lo, hi) over [0, n) split into threads parts.
func parallelFor(n, threads int, f func(lo, hi int)) {
	part := (n + threads - 1) / threads
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		lo, hi := t*part, min((t+1)*part, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(lo, hi)
		}()
	}
	wg.Wait()
}

// Build runs train and add. vectors is row-major (n, dim). The index keeps no
// reference to vectors after Build returns.
func Build(vectors []float32, n, dim int, params map[string]any, threads int, seed uint64) (*Index, error) {
	r, err := intParam(params, "r")
	if err != nil {
		return nil, err
	}
	lBuild, err := intParam(params, "l_build")
	if err != nil {
		return nil, err
	}
	alpha, err := floatParam(params, "alpha")
	if err != nil {
		return nil, err
	}
	m, err := intParam(params, "pq_m")
	if err != nil {
		return nil, err
	}
	metric := stringParam(params, "metric", BuildDefaults)
	switch {
	case metric != "ip" && metric != "l2":
		return nil, fmt.Errorf("diskann: metric must be ip or l2, got %q", metric)
	case r < 1 || lBuild < 1 || alpha < 1:
		return nil, fmt.Errorf("diskann: need r >= 1, l_build >= 1, alpha >= 1")
	case n <= r:
		return nil, fmt.Errorf("diskann: need more than r=%d rows, have %d", r, n)
	case m <= 0 || dim%m != 0:
		return nil, fmt.Errorf("diskann: dim %d is not divisible by pq_m=%d", dim, m)
	case n < ksub:
		return nil, fmt.Errorf("diskann: need at least %d rows for PQ, have %d", ksub, n)
	case OutPath == "":
		return nil, fmt.Errorf("diskann: OutPath is empty; set it before Build")
	}
	if threads < 1 {
		threads = 1
	}
	ix := &Index{
		n: n, dim: dim, r: r, m: m, dsub: dim / m, l2: metric == "l2",
		path: OutPath + ".diskann", threads: threads,
	}
	ix.recBytes = (dim*4 + r*4 + pageSize - 1) / pageSize * pageSize

	// Add, part 1: the Vamana graph on full-precision vectors.
	start := time.Now()
	g := &graph{
		vectors: vectors, n: n, dim: dim, r: r,
		adj: make([]int32, n*r), cnt: make([]int32, n), locks: make([]sync.Mutex, n),
	}
	g.entry = medoid(vectors, n, dim)
	g.initGraph(seed)
	g.pass(lBuild, 1.0, threads)
	g.pass(lBuild, float32(alpha), threads)
	ix.entry = g.entry
	addS := time.Since(start).Seconds()

	// Train: PQ codebooks on all rows.
	start = time.Now()
	if err := ix.trainPQ(vectors, threads, seed); err != nil {
		return nil, err
	}
	ix.trainS = time.Since(start).Seconds()

	// Add, part 2: encode all rows, write the record file.
	start = time.Now()
	ix.encode(vectors, threads)
	if err := ix.writeFile(vectors, g); err != nil {
		return nil, err
	}
	var edges int64
	for _, c := range g.cnt {
		edges += int64(c)
	}
	ix.meanDeg = float64(edges) / float64(n)
	ix.addS = addS + time.Since(start).Seconds()
	// g and vectors go out of scope here: the index holds no full vectors.
	return ix, nil
}

// trainPQ trains one codebook of 256 centroids per sub-vector on all rows
// (section 6.4 with train_size = N, seed + j, no normalization, the metric).
func (ix *Index) trainPQ(vectors []float32, threads int, seed uint64) error {
	n, dim, m, dsub := ix.n, ix.dim, ix.m, ix.dsub
	ix.codebooks = make([]float32, m*ksub*dsub)
	sub := make([]float32, n*dsub)
	for j := 0; j < m; j++ {
		for i := 0; i < n; i++ {
			copy(sub[i*dsub:(i+1)*dsub], vectors[i*dim+j*dsub:i*dim+(j+1)*dsub])
		}
		cfg := kmeans.Config{K: ksub, Iters: pqIters, Seed: seed + uint64(j), Threads: threads, L2: ix.l2, NoNormalize: true}
		cb, err := kmeans.Train(sub, n, dsub, cfg)
		if err != nil {
			return fmt.Errorf("diskann: codebook %d: %w", j, err)
		}
		copy(ix.codebooks[j*ksub*dsub:(j+1)*ksub*dsub], cb)
	}
	return nil
}

// encode sets code[i][j] = best centroid of sub-vector j of row i.
func (ix *Index) encode(vectors []float32, threads int) {
	dim, m, dsub := ix.dim, ix.m, ix.dsub
	ix.codes = make([]uint8, ix.n*m)
	parallelFor(ix.n, threads, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := vectors[i*dim : (i+1)*dim]
			for j := 0; j < m; j++ {
				c, _ := kmeans.Nearest(row[j*dsub:(j+1)*dsub], ix.codebooks[j*ksub*dsub:(j+1)*ksub*dsub], ksub, dsub, ix.l2)
				ix.codes[i*m+j] = uint8(c)
			}
		}
	})
}

// writeFile writes one record per node: dim float32 then r int32 edges
// (-1 = empty), little-endian, zero-padded to recBytes.
func (ix *Index) writeFile(vectors []float32, g *graph) error {
	f, err := os.Create(ix.path)
	if err != nil {
		return fmt.Errorf("diskann: %w", err)
	}
	if err := setNoCache(f); err != nil {
		f.Close()
		return fmt.Errorf("diskann: F_NOCACHE: %w", err)
	}
	w := bufio.NewWriterSize(f, 1<<20)
	rec := make([]byte, ix.recBytes)
	dim, r := ix.dim, ix.r
	for i := 0; i < ix.n; i++ {
		clear(rec)
		for d, x := range vectors[i*dim : (i+1)*dim] {
			binary.LittleEndian.PutUint32(rec[d*4:], math.Float32bits(x))
		}
		o, c := i*r, int(g.cnt[i])
		for e := 0; e < r; e++ {
			v := int32(-1)
			if e < c {
				v = g.adj[o+e]
			}
			binary.LittleEndian.PutUint32(rec[dim*4+e*4:], uint32(v))
		}
		if _, err := w.Write(rec); err != nil {
			f.Close()
			return fmt.Errorf("diskann: %w", err)
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return fmt.Errorf("diskann: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("diskann: %w", err)
	}
	st, err := os.Stat(ix.path)
	if err != nil {
		return fmt.Errorf("diskann: %w", err)
	}
	ix.diskBytes = st.Size()
	// The written pages sit in the OS cache. Drop them, so no search mode
	// starts with pages this process wrote.
	return evictCache(ix.path, ix.diskBytes)
}

// ---------- record I/O ----------

// alignedBuf returns a byte slice of size bytes whose start is 4096-aligned.
func alignedBuf(size int) []byte {
	b := make([]byte, size+pageSize)
	off := 0
	if rem := int(uintptr(unsafe.Pointer(&b[0])) % pageSize); rem != 0 {
		off = pageSize - rem
	}
	return b[off : off+size : off+size]
}

// openMmap maps the whole record file read-only.
func (ix *Index) openMmap() error {
	if ix.mapped != nil {
		return nil
	}
	if ix.ncFile != nil { // leaving io=nocache: close the uncached descriptor
		ix.ncFile.Close()
		ix.ncFile = nil
	}
	f, err := os.Open(ix.path)
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := syscall.Mmap(int(f.Fd()), 0, int(ix.diskBytes), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	ix.mapped = data
	return nil
}

// openNoCache opens the record file with the OS cache disabled.
func (ix *Index) openNoCache() error {
	if ix.ncFile != nil {
		return nil
	}
	// Drop this process's map and the cached pages of the file, so that the
	// reads below come from the SSD (section 6.7.1 rule 1).
	if ix.mapped != nil {
		if err := syscall.Munmap(ix.mapped); err != nil {
			return err
		}
		ix.mapped = nil
	}
	if err := evictCache(ix.path, ix.diskBytes); err != nil {
		return err
	}
	f, err := openUncached(ix.path)
	if err != nil {
		return err
	}
	ix.ncFile = f
	return nil
}

// readRecord returns record i. With io=mmap it is a view into the map. With
// io=nocache it is read into buf (4096-aligned) with pread.
func (ix *Index) readRecord(i int32, nocache bool, buf []byte) []byte {
	off := int64(i) * int64(ix.recBytes)
	ix.diskReads++
	ix.bytesRead += float64(ix.recBytes)
	if !nocache {
		return ix.mapped[off : off+int64(ix.recBytes)]
	}
	buf = buf[:ix.recBytes]
	for got := 0; got < len(buf); {
		k, err := syscall.Pread(int(ix.ncFile.Fd()), buf[got:], off+int64(got))
		if err != nil || k <= 0 {
			panic(fmt.Sprintf("diskann: pread record %d: %v", i, err))
		}
		got += k
	}
	return buf
}

// Close releases the map and the file. Tests call it.
func (ix *Index) Close() error {
	var err error
	if ix.mapped != nil {
		err = syscall.Munmap(ix.mapped)
		ix.mapped = nil
	}
	if ix.ncFile != nil {
		if e := ix.ncFile.Close(); err == nil {
			err = e
		}
		ix.ncFile = nil
	}
	return err
}

// ---------- search ----------

type searchScratch struct {
	seen   stamps
	list   []cand
	table  []float32 // (m, 256)
	buf    []byte    // aligned pread buffer
	vecs   []byte    // raw vector bytes of expanded nodes
	vecPos []int32   // node id -> slot in vecs, valid when vecGen matches
	vecGen []uint32
	step   []int32
	full   []float32
	final  *distance.TopK
}

func (ix *Index) scratch() *searchScratch {
	if ix.sc == nil {
		ix.sc = &searchScratch{
			seen:   newStamps(ix.n),
			table:  make([]float32, ix.m*ksub),
			buf:    alignedBuf(ix.recBytes),
			vecPos: make([]int32, ix.n),
			vecGen: make([]uint32, ix.n),
			full:   make([]float32, ix.dim),
		}
	}
	return ix.sc
}

// pqScore returns sum_j T[j][code[i][j]].
func (ix *Index) pqScore(T []float32, i int32) float32 {
	m := ix.m
	code := ix.codes[int(i)*m : int(i)*m+m]
	var s float32
	for j, c := range code {
		s += T[j*ksub+int(c)]
	}
	return s
}

// Search returns the k best ids and scores for one query, best first.
func Search(ix *Index, query []float32, k int, params map[string]any) ([]int64, []float32) {
	l, err1 := intParam(params, "l")
	beam, err2 := intParam(params, "beam")
	rerank, err3 := intParam(params, "rerank")
	if err := firstErr(err1, err2, err3); err != nil {
		panic(err)
	}
	l = max(l, k)
	beam = max(beam, 1)
	nocache := stringParam(params, "io", SearchDefaults) == "nocache"
	if nocache {
		if err := ix.openNoCache(); err != nil {
			panic(fmt.Sprintf("diskann: open %s: %v", ix.path, err))
		}
	} else if err := ix.openMmap(); err != nil {
		panic(fmt.Sprintf("diskann: mmap %s: %v", ix.path, err))
	}
	s := ix.scratch()
	m, dsub, dim, r := ix.m, ix.dsub, ix.dim, ix.r

	// PQ table for this query (6.4.1).
	T := s.table
	for j := 0; j < m; j++ {
		qj := query[j*dsub : (j+1)*dsub]
		cb := ix.codebooks[j*ksub*dsub : (j+1)*ksub*dsub]
		for c := 0; c < ksub; c++ {
			T[j*ksub+c] = kmeans.Score(qj, cb[c*dsub:(c+1)*dsub], ix.l2)
		}
	}

	// Beam search.
	s.seen.next()
	gen := s.seen.gen
	s.vecs = s.vecs[:0]
	e := ix.entry
	s.seen.mark[e] = gen
	s.list = append(s.list[:0], cand{item: item{ix.pqScore(T, e), e}})
	ix.dists++
	vecBytes := dim * 4
	for {
		s.step = s.step[:0]
		for idx := range s.list {
			if !s.list[idx].expanded {
				s.list[idx].expanded = true
				s.step = append(s.step, s.list[idx].id)
				if len(s.step) == beam {
					break
				}
			}
		}
		if len(s.step) == 0 {
			break
		}
		for _, id := range s.step {
			rec := ix.readRecord(id, nocache, s.buf)
			s.vecPos[id] = int32(len(s.vecs) / vecBytes)
			s.vecGen[id] = gen
			s.vecs = append(s.vecs, rec[:vecBytes]...)
			edges := rec[vecBytes : vecBytes+r*4]
			for e := 0; e < r; e++ {
				nb := int32(binary.LittleEndian.Uint32(edges[e*4:]))
				if nb < 0 {
					break
				}
				if s.seen.mark[nb] == gen {
					continue
				}
				s.seen.mark[nb] = gen
				ix.dists++
				s.list = insertSorted(s.list, cand{item: item{ix.pqScore(T, nb), nb}}, l)
			}
		}
	}

	// Rerank the top candidates with the full vectors from their records.
	if rerank <= 0 {
		fk := distance.NewTopK(k)
		for _, c := range s.list {
			fk.Push(int64(c.id), c.s)
		}
		return fk.Results()
	}
	if s.final == nil || s.final.K() != k {
		s.final = distance.NewTopK(k)
	}
	fk := s.final
	fk.Reset()
	for idx := 0; idx < min(rerank, len(s.list)); idx++ {
		id := s.list[idx].id
		var raw []byte
		if s.vecGen[id] == gen {
			p := int(s.vecPos[id]) * vecBytes
			raw = s.vecs[p : p+vecBytes]
		} else {
			raw = ix.readRecord(id, nocache, s.buf)[:vecBytes]
		}
		for d := 0; d < dim; d++ {
			s.full[d] = math.Float32frombits(binary.LittleEndian.Uint32(raw[d*4:]))
		}
		fk.Push(int64(id), distance.Dot(query, s.full))
		ix.dists++
	}
	return fk.Results()
}

func firstErr(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

// IndexBytes returns the computed memory of the index structure (section 4):
// bytes held in RAM only = PQ codes (n * m) + codebooks (m * 256 * dsub * 4)
// + the entry point (4). The record file is extra.disk_bytes.
func IndexBytes(ix *Index) int64 {
	return int64(ix.n)*int64(ix.m) + int64(ix.m*ksub*ix.dsub*4) + 4
}

// TrainSeconds returns the train time of the build.
func (ix *Index) TrainSeconds() float64 { return ix.trainS }

// AddSeconds returns the add time of the build.
func (ix *Index) AddSeconds() float64 { return ix.addS }

// DistanceComputations returns the total score computations of Search so far:
// PQ table scores plus rerank dot products.
func (ix *Index) DistanceComputations() int64 { return ix.dists }

// Extra returns index-specific build-time output keys (top-level "extra").
func (ix *Index) Extra() map[string]any {
	return map[string]any{
		"disk_bytes":      ix.diskBytes,
		"record_bytes":    ix.recBytes,
		"entry_point":     ix.entry,
		"mean_out_degree": ix.meanDeg,
		"build_threads":   ix.threads,
	}
}

// SearchCounters returns cumulative per-search counters: disk_reads (records
// read) and disk_bytes_read. bench reports the mean per query.
func (ix *Index) SearchCounters() map[string]float64 {
	return map[string]float64{"disk_reads": ix.diskReads, "disk_bytes_read": ix.bytesRead}
}

// Path returns the record file path. Tests use it.
func (ix *Index) Path() string { return ix.path }

// OutDegrees returns the out-degree of every node, read from the record file.
// Tests use it.
func (ix *Index) OutDegrees() ([]int, error) {
	f, err := os.Open(ix.path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)
	rec := make([]byte, ix.recBytes)
	out := make([]int, ix.n)
	for i := range out {
		if _, err := io.ReadFull(br, rec); err != nil {
			return nil, err
		}
		c := 0
		for e := 0; e < ix.r; e++ {
			v := int32(binary.LittleEndian.Uint32(rec[ix.dim*4+e*4:]))
			if v < 0 {
				break
			}
			if int(v) >= ix.n || v == int32(i) {
				return nil, fmt.Errorf("node %d has bad edge %d", i, v)
			}
			c++
		}
		out[i] = c
	}
	return out, nil
}
