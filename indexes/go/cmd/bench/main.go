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

	res, err := benchmark(o, spec, vectors, n, dim, queries, q, buildParams, searchSets)
	if err != nil {
		return err
	}
	return writeJSON(o.out, res)
}
