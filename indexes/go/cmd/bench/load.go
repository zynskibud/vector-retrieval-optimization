package main

// Load runs (CONTRACT.md section 12): C client goroutines search in a closed
// loop for a fixed time, optionally while one inserter goroutine adds rows.

import (
	"fmt"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"vro/indexes/go/hnsw"
)

// defaultLoadDuration is the loop time when --duration is not given in a load
// run (section 12.1).
const defaultLoadDuration = 20.0

// insertBatch is the rows per insert call (section 12.2).
const insertBatch = 100

// loadMode reports whether the flags ask for a load run. With --clients 1, no
// --duration and no --insert-rate, bench runs the one-thread pass of section 4.
func (o options) loadMode() bool {
	return o.clients > 1 || o.duration > 0 || o.insertRate > 0
}

// loadIndexes lists the indexes that support concurrent search and Insert.
// buildCap builds on the first n rows with room for capN rows.
var loadIndexes = map[string]func(v []float32, n, capN, dim int, p map[string]any, threads int, seed uint64) (instance, error){
	"hnsw": func(v []float32, n, capN, dim int, p map[string]any, threads int, seed uint64) (instance, error) {
		ix, err := hnsw.BuildCap(v, n, capN, dim, p, threads, seed)
		if err != nil {
			return instance{}, err
		}
		return instance{
			idx:    ix,
			search: func(q []float32, k int, sp map[string]any) ([]int64, []float32) { return hnsw.Search(ix, q, k, sp) },
			bytes:  hnsw.IndexBytes(ix),
			insert: func(ids []int64, vecs []float32) error { return hnsw.Insert(ix, ids, vecs) },
			repair: func() { hnsw.Repair(ix) },
			rows:   ix.Len,
		}, nil
	},
}

// cpuSeconds returns the user + system CPU time of the process.
func cpuSeconds() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return float64(ru.Utime.Nano()+ru.Stime.Nano()) / 1e9
}

// inserter adds rows next..end-1 in batches of insertBatch at rate rows/s.
type inserter struct {
	inst      instance
	vectors   []float32
	dim       int
	next, end int
	rate      float64
	batchMS   []float64
	errors    int
}

// run inserts batches until the rows are exhausted or stop is closed. Batch j
// of this call is due at start + j * insertBatch / rate.
func (in *inserter) run(stop <-chan struct{}) (inserted int) {
	start := time.Now()
	interval := time.Duration(float64(insertBatch) / in.rate * float64(time.Second))
	for j := 0; in.next < in.end; j++ {
		if wait := time.Until(start.Add(time.Duration(j) * interval)); wait > 0 {
			select {
			case <-stop:
				return inserted
			case <-time.After(wait):
			}
		}
		select {
		case <-stop:
			return inserted
		default:
		}
		hi := min(in.next+insertBatch, in.end)
		ids := make([]int64, hi-in.next)
		for i := range ids {
			ids[i] = int64(in.next + i)
		}
		t0 := time.Now()
		err := in.inst.insert(ids, in.vectors[in.next*in.dim:hi*in.dim])
		in.batchMS = append(in.batchMS, float64(time.Since(t0).Nanoseconds())/1e6)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench: insert:", err)
			in.errors++
			return inserted
		}
		inserted += hi - in.next
		in.next = hi
	}
	return inserted
}

// safeSearch runs one search. A panic or an invalid result (an ID outside
// [0, rows) other than -1 padding, or a repeated ID) counts as an error.
func safeSearch(inst instance, query []float32, k int, sp map[string]any, rows int) (ids []int64, scores []float32, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			ids, scores, ok = nil, nil, false
		}
	}()
	ids, scores = inst.search(query, k, sp)
	if len(ids) != k || len(scores) != k {
		return ids, scores, false
	}
	seen := make(map[int64]bool, k)
	for _, id := range ids {
		if id == -1 {
			continue
		}
		if id < 0 || id >= int64(rows) || seen[id] {
			return ids, scores, false
		}
		seen[id] = true
	}
	return ids, scores, true
}

// loadRun runs one search setting as a load run. Worker w starts at query
// w*q/clients and loops over the queries until the duration ends. Worker 0
// starts at query 0 and always completes its first pass, which gives ids and
// scores. If ins is not nil, the inserter runs during the loop.
func loadRun(inst instance, queries []float32, dim, q, k int, sp map[string]any,
	clients int, duration float64, ins *inserter) searchRun {
	run := searchRun{SearchParams: sp, IDs: make([][]int64, q), Scores: make([][]score, q)}
	raw := make([][]float32, q)
	lat := make([][]float64, clients)
	var done, errs atomic.Int64
	dc0 := inst.idx.DistanceComputations()
	c0 := inst.idx.SearchCounters()
	// Bound for a valid result ID: rows are only added during the loop.
	maxRows := inst.rows
	if ins != nil {
		maxRows = func() int { return ins.end }
	}
	stopIns := make(chan struct{})
	var insWG sync.WaitGroup
	inserted := 0

	cpu0 := cpuSeconds()
	start := time.Now()
	deadline := start.Add(time.Duration(duration * float64(time.Second)))
	if ins != nil {
		insWG.Add(1)
		go func() { defer insWG.Done(); inserted = ins.run(stopIns) }()
	}
	var wg sync.WaitGroup
	for w := 0; w < clients; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			own := make([]float64, 0, 4096)
			for i := 0; ; i++ {
				qi := (w*q/clients + i) % q
				firstPass := w == 0 && i < q
				if !firstPass && !time.Now().Before(deadline) {
					break
				}
				query := queries[qi*dim : (qi+1)*dim]
				t0 := time.Now()
				ids, scores, ok := safeSearch(inst, query, k, sp, maxRows())
				own = append(own, float64(time.Since(t0).Nanoseconds())/1e6)
				done.Add(1)
				if !ok {
					errs.Add(1)
				}
				if firstPass {
					run.IDs[qi], raw[qi] = ids, scores
				}
			}
			lat[w] = own
		}(w)
	}
	wg.Wait()
	wall := time.Since(start).Seconds()
	cpu1 := cpuSeconds()
	close(stopIns)
	insWG.Wait()

	n := done.Load()
	for _, l := range lat {
		run.LatencyMS = append(run.LatencyMS, l...)
	}
	for i, s := range raw {
		if s == nil { // a failed first-pass search: pad the row
			run.IDs[i] = make([]int64, k)
			s = make([]float32, k)
			for j := range s {
				run.IDs[i][j] = -1
			}
		}
		run.Scores[i] = toScores(s)
	}
	run.TotalS = wall
	run.QPS = float64(n) / wall
	run.Extra = meanCounters(c0, inst.idx.SearchCounters(), int(n))
	if dc1 := inst.idx.DistanceComputations(); dc0 >= 0 && dc1 >= 0 {
		mean := float64(dc1-dc0) / float64(n)
		run.DistanceComputations = &mean
	}
	run.Extra["errors"] = errs.Load()
	run.Extra["cpu_pct"] = (cpu1 - cpu0) / wall * 100
	run.Extra["clients"] = clients
	run.Extra["duration_s"] = duration
	run.Extra["queries_done"] = n
	if ins != nil {
		run.Extra["inserted_rows"] = inserted
	}
	return run
}

// median returns the median of xs (0 for none).
func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}
