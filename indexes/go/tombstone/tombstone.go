// Package tombstone is the bit set of deleted rows (CONTRACT.md section 13.3),
// shared by flat, ivf and hnsw. Bit i is set when row i is deleted. The set
// holds one bit per row, so it takes ceil(N/8) bytes, rounded up to whole
// 64-bit words.
//
// Concurrency: Delete and Update in Phase 5 run with no concurrent searches,
// so the words are read and written with plain loads and stores.
package tombstone

// Set is a bit set over n rows.
type Set struct {
	words []uint64
	n     int
	count int
}

// New returns an empty set over n rows.
func New(n int) *Set { return &Set{words: make([]uint64, (n+63)/64), n: n} }

// Has reports whether row i is deleted. A nil set has no deleted rows.
func (s *Set) Has(i int) bool {
	return s != nil && s.words[i>>6]&(1<<(uint(i)&63)) != 0
}

// Add marks row i deleted. It reports whether the row was live before.
func (s *Set) Add(i int) bool {
	w, b := i>>6, uint64(1)<<(uint(i)&63)
	if s.words[w]&b != 0 {
		return false
	}
	s.words[w] |= b
	s.count++
	return true
}

// Count returns the number of deleted rows.
func (s *Set) Count() int {
	if s == nil {
		return 0
	}
	return s.count
}

// Bytes returns the memory of the bit set: 8 bytes per 64 rows.
func (s *Set) Bytes() int64 {
	if s == nil {
		return 0
	}
	return int64(len(s.words)) * 8
}

// Apply marks every row i < n with mask[i] true. It returns the rows newly deleted.
func Apply(s *Set, mask []bool) int {
	added := 0
	for i := 0; i < s.n && i < len(mask); i++ {
		if mask[i] && s.Add(i) {
			added++
		}
	}
	return added
}
