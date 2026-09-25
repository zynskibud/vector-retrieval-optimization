package diskann

import (
	"os"
	"syscall"
)

// openUncached opens path read-only with O_DIRECT (section 6.7.1). Records,
// offsets and the read buffer are 4096-aligned.
func openUncached(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_RDONLY|syscall.O_DIRECT, 0)
}

// evictCache drops the cached pages of path with posix_fadvise(DONTNEED).
func evictCache(path string, size int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	const fadvDontNeed = 4
	if _, _, e := syscall.Syscall6(syscall.SYS_FADVISE64, f.Fd(), 0, uintptr(size), fadvDontNeed, 0, 0); e != 0 {
		return e
	}
	return nil
}

// setNoCache is a no-op: the written pages are dropped by evictCache.
func setNoCache(f *os.File) error { return nil }
