package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"time"

	"vro/indexes/go/diskann"
	"vro/indexes/go/flat"
	"vro/indexes/go/hnsw"
	"vro/indexes/go/ivf"
)

func benchmark(o options, spec indexSpec, vectors []float32, n, dim int, queries []float32, q int,
	buildParams map[string]any, searchSets []map[string]any, changes changeData) (*result, error) {
	diskann.OutPath = o.out
	flat.DataDir, ivf.DataDir, hnsw.DataDir = o.data, o.data, o.data
	// With inserts, the build takes the first 90% of the rows (section 12.2).
	nBuild := n
	if o.insertRate > 0 {
		nBuild = n * 9 / 10
	}
	var inst instance
	var err error
	var loadS float64
	switch {
	case o.loadPath != "":
		fmt.Fprintf(os.Stderr, "bench: loading %s from %s\n", o.index, o.loadPath)
		inst, loadS, err = loadIndex(o, dim, buildParams)
		if inst.rows == nil {
			inst.rows = func() int { return n }
		}
	case loadIndexes[o.index] != nil && o.loadMode():
		fmt.Fprintf(os.Stderr, "bench: building %s on %d rows\n", o.index, nBuild)
		inst, err = loadIndexes[o.index](vectors, nBuild, n, dim, buildParams, o.threads, o.seed)
	default:
		fmt.Fprintf(os.Stderr, "bench: building %s on %d rows\n", o.index, nBuild)
		inst, err = spec.build(vectors, n, dim, buildParams, o.threads, o.seed)
		inst.rows = func() int { return n }
	}
	if err != nil {
		return nil, err
	}
	res := &result{
		ContractVersion: 1, Language: "go", Index: o.index, DataDir: o.data,
		N: n, Dim: dim, Q: q, K: o.k, Threads: o.threads, Seed: o.seed,
		BuildParams: buildParams,
		Build: buildInfo{
			TrainS:     inst.idx.TrainSeconds(),
			AddS:       inst.idx.AddSeconds(),
			TotalS:     inst.idx.TrainSeconds() + inst.idx.AddSeconds(),
			PeakRSSMB:  peakRSSMB(),
			IndexBytes: inst.bytes,
		},
		Machine: machineInfo(),
	}
	var changeExtra map[string]any
	if o.changeMode() {
		inst, changeExtra, err = applyChanges(o, inst, changes)
		if err != nil {
			return nil, err
		}
	}
	fileExtra := map[string]any{}
	if o.loadPath != "" {
		// A loaded index has no build: train_s = add_s = 0 (section 15.2).
		res.Build.TrainS, res.Build.AddS, res.Build.TotalS = 0, 0, 0
		fileExtra["load_s"] = loadS
		fileExtra["loaded_from"] = o.loadPath
	}
	if o.savePath != "" {
		se, err := saveIndex(o, inst)
		if err != nil {
			return nil, err
		}
		for key, v := range se {
			fileExtra[key] = v
		}
	}
	defer func() {
		for key, v := range fileExtra {
			res.Extra[key] = v
		}
	}()
	warmup(inst, queries, dim, min(o.warmup, q), o.k, searchSets[0])
	if !o.loadMode() {
		for _, sp := range searchSets {
			fmt.Fprintf(os.Stderr, "bench: searching %v\n", sp)
			res.Searches = append(res.Searches, timedRun(inst, queries, dim, q, o.k, sp))
		}
		res.Extra = inst.idx.Extra()
		for key, v := range changeExtra {
			res.Extra[key] = v
		}
		return res, nil
	}
	var ins *inserter
	if o.insertRate > 0 {
		ins = &inserter{inst: inst, vectors: vectors, dim: dim, next: nBuild, end: n, rate: o.insertRate}
	}
	for _, sp := range searchSets {
		fmt.Fprintf(os.Stderr, "bench: load run %v, %d clients, %.1f s\n", sp, o.clients, o.duration)
		res.Searches = append(res.Searches, loadRun(inst, queries, dim, q, o.k, sp, o.clients, o.duration, ins))
	}
	res.Extra = map[string]any{}
	if ins != nil {
		// Rows left when the loops ended go in untimed (the insert tail),
		// then Repair once, then one one-thread pass with the first search
		// setting as searches[-1].
		duringLoop := ins.next - nBuild
		_, tailS := ins.tail()
		fmt.Fprintf(os.Stderr, "bench: insert tail: %d rows in %.2f s\n", ins.end-nBuild-duringLoop, tailS)
		inst.repair()
		sp := make(map[string]any, len(searchSets[0])+1)
		for key, v := range searchSets[0] {
			sp[key] = v
		}
		sp["phase"] = "after_inserts"
		fmt.Fprintf(os.Stderr, "bench: searching %v\n", sp)
		after := timedRun(inst, queries, dim, q, o.k, sp)
		inserted := ins.next - nBuild
		after.Extra["inserted_rows"] = inserted
		after.Extra["insert_p50_ms"] = median(ins.batchMS)
		after.Extra["insert_errors"] = ins.errors
		after.Extra["inserted_during_loop"] = duringLoop
		after.Extra["insert_tail_s"] = tailS
		res.Searches = append(res.Searches, after)
		res.Extra["inserted_rows"] = inserted
		res.Extra["insert_p50_ms"] = median(ins.batchMS)
		res.Extra["insert_rate"] = o.insertRate
		res.Extra["inserted_during_loop"] = duringLoop
		res.Extra["insert_tail_s"] = tailS
		res.Extra["build_rows"] = nBuild
	}
	for key, v := range inst.idx.Extra() {
		res.Extra[key] = v
	}
	res.Extra["clients"] = o.clients
	res.Extra["duration_s"] = o.duration
	return res, nil
}

// warmup runs the first count queries once, untimed, with the first parameter set.
func warmup(inst instance, queries []float32, dim, count, k int, sp map[string]any) {
	for i := 0; i < count; i++ {
		inst.search(queries[i*dim:(i+1)*dim], k, sp)
	}
}

// timedRun runs all queries one at a time on this goroutine and times each search call.
func timedRun(inst instance, queries []float32, dim, q, k int, sp map[string]any) searchRun {
	run := searchRun{
		SearchParams: sp,
		IDs:          make([][]int64, q),
		Scores:       make([][]score, q),
		LatencyMS:    make([]float64, q),
	}
	raw := make([][]float32, q)
	dc0 := inst.idx.DistanceComputations()
	c0 := inst.idx.SearchCounters()
	loopStart := time.Now()
	for i := 0; i < q; i++ {
		query := queries[i*dim : (i+1)*dim]
		start := time.Now()
		ids, scores := inst.search(query, k, sp)
		run.LatencyMS[i] = float64(time.Since(start).Nanoseconds()) / 1e6
		run.IDs[i], raw[i] = ids, scores
	}
	run.TotalS = time.Since(loopStart).Seconds() // wall time of the timed loop
	run.QPS = float64(q) / run.TotalS
	for i, s := range raw {
		run.Scores[i] = toScores(s)
	}
	run.Extra = meanCounters(c0, inst.idx.SearchCounters(), q)
	if dc1 := inst.idx.DistanceComputations(); dc0 >= 0 && dc1 >= 0 {
		mean := float64(dc1-dc0) / float64(q)
		run.DistanceComputations = &mean
	}
	return run
}

// meanCounters returns (after - before) / q for every counter.
func meanCounters(before, after map[string]float64, q int) map[string]any {
	out := make(map[string]any, len(after))
	for key, v := range after {
		out[key] = (v - before[key]) / float64(q)
	}
	return out
}

func toScores(s []float32) []score {
	out := make([]score, len(s))
	for i, v := range s {
		out[i] = score(v)
	}
	return out
}

// peakRSSMB returns ru_maxrss in MB (2^20 bytes). ru_maxrss is bytes on macOS, KB on Linux.
func peakRSSMB() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	b := float64(ru.Maxrss)
	if runtime.GOOS != "darwin" {
		b *= 1024
	}
	return b / (1 << 20)
}

func machineInfo() machine {
	m := machine{OS: runtime.GOOS, Arch: runtime.GOARCH, CPU: "unknown", Cores: runtime.NumCPU()}
	if runtime.GOOS == "darwin" {
		if out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			m.CPU = strings.TrimSpace(string(out))
		}
	}
	return m
}
