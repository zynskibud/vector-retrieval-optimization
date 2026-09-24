// Package distance holds the dot product and the top-k selector shared by all indexes.
package distance

import (
	"math"
	"sort"
)

// Dot returns the dot product of a and b. The vectors must have equal length.
// The loop re-slices both inputs to 4 elements per step, so the compiler can
// prove every index is in bounds and emits no bounds checks in the loop.
func Dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	for len(a) >= 4 {
		x, y := a[:4:4], b[:4:4]
		s0 += x[0] * y[0]
		s1 += x[1] * y[1]
		s2 += x[2] * y[2]
		s3 += x[3] * y[3]
		a, b = a[4:], b[4:]
	}
	for i := range a {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// TopK keeps the k highest scores seen so far. It is a bounded min-heap:
// the root is the worst kept result, so a new score only enters if it beats the root.
type TopK struct {
	k      int
	ids    []int64
	scores []float32
}

// NewTopK returns an empty selector for k results.
func NewTopK(k int) *TopK {
	return &TopK{k: k, ids: make([]int64, 0, k), scores: make([]float32, 0, k)}
}

// Reset empties the selector and keeps its memory.
func (t *TopK) Reset() {
	t.ids, t.scores = t.ids[:0], t.scores[:0]
}

// Push offers one candidate. On equal scores the lower ID ranks higher.
func (t *TopK) Push(id int64, score float32) {
	if len(t.ids) < t.k {
		t.ids = append(t.ids, id)
		t.scores = append(t.scores, score)
		t.up(len(t.ids) - 1)
		return
	}
	if t.k == 0 || score < t.scores[0] || (score == t.scores[0] && id > t.ids[0]) {
		return
	}
	t.ids[0], t.scores[0] = id, score
	t.down(0)
}

// Results returns k ids and scores, best first. Missing slots hold -1 and -Inf.
func (t *TopK) Results() ([]int64, []float32) {
	n := len(t.ids)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		sa, sb := t.scores[idx[a]], t.scores[idx[b]]
		if sa != sb {
			return sa > sb
		}
		return t.ids[idx[a]] < t.ids[idx[b]]
	})
	ids := make([]int64, t.k)
	scores := make([]float32, t.k)
	for i := 0; i < t.k; i++ {
		if i < n {
			ids[i], scores[i] = t.ids[idx[i]], t.scores[idx[i]]
		} else {
			ids[i], scores[i] = -1, float32(math.Inf(-1))
		}
	}
	return ids, scores
}

// less orders the heap so the worst result is at the root.
func (t *TopK) less(i, j int) bool {
	if t.scores[i] != t.scores[j] {
		return t.scores[i] < t.scores[j]
	}
	return t.ids[i] > t.ids[j]
}

func (t *TopK) swap(i, j int) {
	t.ids[i], t.ids[j] = t.ids[j], t.ids[i]
	t.scores[i], t.scores[j] = t.scores[j], t.scores[i]
}

func (t *TopK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !t.less(i, p) {
			return
		}
		t.swap(i, p)
		i = p
	}
}

func (t *TopK) down(i int) {
	n := len(t.ids)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		m := l
		if r := l + 1; r < n && t.less(r, l) {
			m = r
		}
		if !t.less(m, i) {
			return
		}
		t.swap(i, m)
		i = m
	}
}

// K returns the number of results the selector keeps.
func (t *TopK) K() int { return t.k }
