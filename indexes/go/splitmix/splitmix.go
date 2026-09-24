// Package splitmix implements the SplitMix64 PRNG of CONTRACT.md section 5.
// All four languages use this generator so that random choices match.
package splitmix

// Rng is a SplitMix64 generator. The zero value is seeded with 0.
type Rng struct {
	s uint64
}

// New returns a generator whose state starts at seed.
func New(seed uint64) *Rng {
	return &Rng{s: seed}
}

// NextU64 returns the next 64-bit value.
func (r *Rng) NextU64() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// NextF64 returns a float64 in [0, 1) built from the top 53 bits.
func (r *Rng) NextF64() float64 {
	return float64(r.NextU64()>>11) * (1.0 / (1 << 53))
}

// NextBelow returns NextU64() % n. The contract uses plain modulo, not rejection sampling.
func (r *Rng) NextBelow(n uint64) uint64 {
	return r.NextU64() % n
}
