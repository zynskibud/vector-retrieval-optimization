//! Dot product and top-k selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Dot product of two equal-length vectors.
///
/// Sixteen independent accumulators remove the serial dependency on one sum,
/// so LLVM can vectorize with NEON without reassociating floats.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const W: usize = 16;
    let n = a.len().min(b.len());
    let body = n - n % W;
    let mut acc = [0.0f32; W];
    let mut i = 0;
    while i < body {
        // Fixed-size copies remove the bounds checks that block vectorization.
        let xa: [f32; W] = a[i..i + W].try_into().expect("W lanes");
        let xb: [f32; W] = b[i..i + W].try_into().expect("W lanes");
        acc = std::array::from_fn(|l| acc[l] + xa[l] * xb[l]);
        i += W;
    }
    let tail: f32 = a[body..n].iter().zip(&b[body..n]).map(|(x, y)| x * y).sum();
    // A plain left-to-right sum. A lane-wise or tree-shaped sum here makes LLVM
    // pick 2-lane vectors for the whole loop, which halves the speed.
    acc.iter().sum::<f32>() + tail
}

/// Squared Euclidean distance, for PQ with `metric=l2` (CONTRACT 6.4.1).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// A candidate in the top-k heap. The order is "worse first", so the
/// max-heap top is the worst kept candidate. Ties go to the lower ID.
#[derive(Debug, Clone, Copy)]
struct Worst {
    score: f32,
    id: i64,
}

impl PartialEq for Worst {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Worst {}
impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        // Greater = worse: lower score, or equal score and higher ID.
        other
            .score
            .total_cmp(&self.score)
            .then(self.id.cmp(&other.id))
    }
}

/// Keeps the `k` highest-scoring candidates seen so far.
#[derive(Debug, Clone)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Worst>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// The score a new candidate must beat once the heap is full.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap.peek().map_or(f32::NEG_INFINITY, |w| w.score)
        }
    }

    #[inline]
    pub fn push(&mut self, id: i64, score: f32) {
        if self.k == 0 {
            return;
        }
        let cand = Worst { score, id };
        if self.heap.len() < self.k {
            self.heap.push(cand);
        } else if let Some(mut top) = self.heap.peek_mut() {
            if cand < *top {
                *top = cand;
            }
        }
    }

    /// IDs and scores, best first, padded to `k` with -1 and `-inf`.
    pub fn into_sorted(self) -> (Vec<i64>, Vec<f32>) {
        let k = self.k;
        let sorted = self.heap.into_sorted_vec(); // ascending = best first
        let mut ids: Vec<i64> = sorted.iter().map(|w| w.id).collect();
        let mut scores: Vec<f32> = sorted.iter().map(|w| w.score).collect();
        ids.resize(k, -1);
        scores.resize(k, f32::NEG_INFINITY);
        (ids, scores)
    }
}
