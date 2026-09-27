// Command bench builds one index, runs the queries, and writes one JSON file,
// as defined in indexes/CONTRACT.md sections 2 to 4.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"vro/indexes/go/npy"
)

// usageError marks errors that exit with code 2 (section 2).
type usageError struct{ msg string }

func (e usageError) Error() string { return e.msg }

func usagef(format string, a ...any) error { return usageError{fmt.Sprintf(format, a...)} }

// multiFlag is a repeatable string flag.
type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, " ") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

type options struct {
	index, data, out          string
	k, threads, warmup, limit int
	seed                      uint64
	builds, searches          multiFlag
	clients                   int     // section 12: concurrent search goroutines
	duration                  float64 // seconds of the load loop; 0 = not given
	insertRate                float64 // rows per second inserted during the loop
	// Section 13: changes applied after the build.
	deleteName, updateName string
	compact                bool
	compactMode            string
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		var ue usageError
		if errors.As(err, &ue) {
			os.Exit(2)
		}
		os.Exit(1)
	}
}

func parseFlags(args []string) (options, error) {
	var o options
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.StringVar(&o.index, "index", "", "flat | ivf | pq | ivf_pq | hnsw | diskann")
	fs.StringVar(&o.data, "data", "", "data directory")
	fs.StringVar(&o.out, "out", "", "output JSON path")
	fs.IntVar(&o.k, "k", 10, "results per query")
	fs.Var(&o.builds, "build", "build parameter KEY=VALUE (repeatable)")
	fs.Var(&o.searches, "search", "search parameter set KEY=VALUE[,KEY=VALUE] (repeatable)")
	fs.IntVar(&o.threads, "threads", runtime.NumCPU(), "threads for build")
	fs.Uint64Var(&o.seed, "seed", 42, "seed for every random choice")
	fs.IntVar(&o.warmup, "warmup", 100, "untimed warm-up queries")
	fs.IntVar(&o.limit, "limit", 0, "use only the first INT corpus rows (0 = all)")
	fs.IntVar(&o.clients, "clients", 1, "concurrent search goroutines (section 12)")
	fs.Float64Var(&o.duration, "duration", 0, "seconds of the load loop (0 = one pass; 20 in a load run)")
	fs.Float64Var(&o.insertRate, "insert-rate", 0, "rows per second inserted during the loop (hnsw)")
	fs.StringVar(&o.deleteName, "delete", "", "delete set del10 | del30 | del50 (section 13)")
	fs.StringVar(&o.updateName, "update", "", "update set upd10 (section 13)")
	fs.BoolVar(&o.compact, "compact", false, "compact after the delete or update")
	fs.StringVar(&o.compactMode, "compact-mode", "rebuild", "rebuild | repair")
	if err := fs.Parse(args); err != nil {
		return o, usageError{err.Error()}
	}
	if fs.NArg() > 0 {
		return o, usagef("unexpected arguments: %v", fs.Args())
	}
	if o.index == "" || o.data == "" || o.out == "" {
		return o, usagef("--index, --data and --out are required")
	}
	if o.k < 1 || o.threads < 1 || o.warmup < 0 || o.limit < 0 {
		return o, usagef("--k and --threads must be >= 1, --warmup and --limit >= 0")
	}
	if o.clients < 1 || o.duration < 0 || o.insertRate < 0 {
		return o, usagef("--clients must be >= 1, --duration and --insert-rate >= 0")
	}
	if _, ok := loadIndexes[o.index]; !ok && (o.clients > 1 || o.insertRate > 0) {
		return o, usagef("--clients > 1 and --insert-rate need a concurrent index (hnsw), not %q", o.index)
	}
	if err := checkChangeFlags(o); err != nil {
		return o, err
	}
	if o.loadMode() && o.duration == 0 {
		o.duration = defaultLoadDuration
	}
	return o, nil
}

func run(args []string) error {
	o, err := parseFlags(args)
	if err != nil {
		return err
	}
	spec, ok := registry[o.index]
	if !ok {
		return usagef("unknown index %q", o.index)
	}
	buildParams, err := resolveParams(spec.buildDefaults, o.builds, "build")
	if err != nil {
		return err
	}
	searchSets, err := searchParamSets(spec.searchDefaults, o.searches)
	if err != nil {
		return err
	}

	vectors, n, dim, err := npy.ReadFloat32(filepath.Join(o.data, "vectors.npy"))
	if err != nil {
		return err
	}
	if o.limit > 0 && o.limit < n {
		n = o.limit
		vectors = vectors[:n*dim]
	}
	queries, q, qdim, err := npy.ReadFloat32(filepath.Join(o.data, "queries.npy"))
	if err != nil {
		return err
	}
	if qdim != dim {
		return fmt.Errorf("queries have dim %d, corpus has dim %d", qdim, dim)
	}
	fillDerived(o.index, buildParams, n)
	// Read every filter mask before the build (section 11), so a missing or
	// short mask file fails fast (exit 1) and Search never reads a file.
	for _, sp := range searchSets {
		name, _ := sp["filter"].(string)
		mask, err := npy.FilterMask(o.data, name)
		if err != nil {
			return err
		}
		if mask != nil && len(mask) < n {
			return fmt.Errorf("filter %s has %d rows, corpus has %d", name, len(mask), n)
		}
	}

	// Read the change set before the build, like the filter masks.
	changes, err := readChanges(o, n, dim)
	if err != nil {
		return err
	}
	for _, sp := range searchSets {
		changeSearchParams(o, sp)
	}

	res, err := benchmark(o, spec, vectors, n, dim, queries, q, buildParams, searchSets, changes)
	if err != nil {
		return err
	}
	return writeJSON(o.out, res)
}
