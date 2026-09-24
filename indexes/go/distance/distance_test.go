package distance

import (
	"math"
	"testing"
)

func TestDot(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{5, 4, 3, 2, 1}
	if got := Dot(a, b); got != 35 {
		t.Fatalf("Dot = %v, want 35", got)
	}
}

func TestTopK(t *testing.T) {
	tk := NewTopK(3)
	for i, s := range []float32{0.1, 0.9, 0.5, 0.7, 0.2} {
		tk.Push(int64(i), s)
	}
	ids, scores := tk.Results()
	want := []int64{1, 3, 2}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids = %v scores = %v, want ids %v", ids, scores, want)
		}
	}
}

func TestTopKPadding(t *testing.T) {
	tk := NewTopK(3)
	tk.Push(7, 0.5)
	ids, scores := tk.Results()
	if ids[0] != 7 || ids[1] != -1 || ids[2] != -1 || !math.IsInf(float64(scores[2]), -1) {
		t.Fatalf("ids = %v scores = %v", ids, scores)
	}
}

func TestTopKTiesLowerIDFirst(t *testing.T) {
	tk := NewTopK(2)
	for _, id := range []int64{9, 5, 7, 3} {
		tk.Push(id, 1)
	}
	ids, _ := tk.Results()
	if ids[0] != 3 || ids[1] != 5 {
		t.Fatalf("ids = %v, want [3 5]", ids)
	}
}
