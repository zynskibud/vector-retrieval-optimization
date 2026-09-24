package splitmix

import "testing"

func TestReferenceValues(t *testing.T) {
	cases := []struct {
		seed uint64
		want uint64
	}{
		{42, 13679457532755275413},
		{0, 16294208416658607535},
	}
	for _, c := range cases {
		if got := New(c.seed).NextU64(); got != c.want {
			t.Errorf("seed %d: NextU64 = %d, want %d", c.seed, got, c.want)
		}
	}
}

func TestNextF64Range(t *testing.T) {
	r := New(7)
	for i := 0; i < 10000; i++ {
		if f := r.NextF64(); f < 0 || f >= 1 {
			t.Fatalf("NextF64 = %v, out of [0,1)", f)
		}
	}
}
