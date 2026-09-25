//go:build !darwin && !linux

package diskann

import "os"

// openUncached falls back to a plain open on systems without F_NOCACHE or
// O_DIRECT support here.
func openUncached(path string) (*os.File, error) {
	return os.Open(path)
}

// evictCache is a no-op on these systems.
func evictCache(path string, size int64) error { return nil }

// setNoCache is a no-op on these systems.
func setNoCache(f *os.File) error { return nil }
