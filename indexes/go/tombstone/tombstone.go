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

// Bits returns the set as ceil(n/8) bytes: bit i is row i, least significant
// bit first within a byte (the .vro "tombstones" section, CONTRACT 15.1). A
// nil set gives all zero bytes.
func Bits(s *Set, n int) []byte {
	out := make([]byte, (n+7)/8)
	if s == nil {
		return out
	}
	for j := range out {
		out[j] = byte(s.words[j>>3] >> (8 * uint(j&7)))
	}
	if r := n & 7; r != 0 {
		out[len(out)-1] &= byte(1<<uint(r)) - 1
	}
	return out
}

// FromBits is the inverse of Bits. It returns nil when no bit is set, so an
// index loaded with no deleted row takes the same search path as a fresh one.
func FromBits(b []byte, n int) *Set {
	s := New(n)
	for i := 0; i < n; i++ {
		if b[i>>3]&(1<<uint(i&7)) != 0 {
			s.Add(i)
		}
	}
	if s.count == 0 {
		return nil
	}
	return s
}
