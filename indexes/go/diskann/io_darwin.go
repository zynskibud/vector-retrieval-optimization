package diskann

import (
	"os"
	"syscall"
	"unsafe"
)

// openUncached opens path read-only and sets F_NOCACHE = 1, so reads bypass
// the unified buffer cache (section 6.7.1).
func openUncached(path string) (*os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if _, _, e := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), syscall.F_NOCACHE, 1); e != 0 {
		f.Close()
		return nil, e
	}
	return f, nil
}

// evictCache drops the cached pages of path from the unified buffer cache:
// it maps the file and calls msync(MS_INVALIDATE), as vmtouch -e does on
// macOS. The file must not be mapped elsewhere in the process.
func evictCache(path string, size int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	defer syscall.Munmap(data)
	_, _, e := syscall.Syscall(syscall.SYS_MSYNC, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), syscall.MS_INVALIDATE)
	if e != 0 {
		return e
	}
	return nil
}

// setNoCache sets F_NOCACHE = 1 on f, so its writes do not fill the cache.
func setNoCache(f *os.File) error {
	if _, _, e := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), syscall.F_NOCACHE, 1); e != 0 {
		return e
	}
	return nil
}
