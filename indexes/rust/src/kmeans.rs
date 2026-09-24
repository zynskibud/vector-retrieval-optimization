//! The shared k-means of CONTRACT 6.2, used by ivf, pq, ivf_pq, and diskann.
//!
//! The assignment step runs in parallel with rayon. Call it inside
//! `ThreadPool::install` (as `main` does) to limit it to `--threads` threads.

use crate::distance::{dot, l2_sq};
use crate::splitmix::SplitMix64;
use rayon::prelude::*;

/// How points are assigned to centers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assign {
    /// Highest dot product (IVF centers, PQ with `metric=ip`).
    Dot,
    /// Lowest squared Euclidean distance (PQ with `metric=l2`).
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KmeansOptions {
    pub assign: Assign,
    /// L2-normalize each center after the mean (step 3). PQ codebooks turn this off.
    pub normalize: bool,
}

impl Default for KmeansOptions {
    fn default() -> Self {
        Self {
            assign: Assign::Dot,
            normalize: true,
        }
    }
}

/// Runs k-means on `points` (row-major, `n` rows of `dim`), which are the training rows.
/// Returns the centers, row-major (k, dim).
pub fn kmeans(
    points: &[f32],
    dim: usize,
    k: usize,
    iters: usize,
    seed: u64,
    opts: KmeansOptions,
) -> Result<Vec<f32>, String> {
    let n = points.len() / dim;
    if k == 0 || n < k {
        return Err(format!(
            "k-means needs 1 <= k <= training rows, got k={k}, rows={n}"
        ));
    }
    let mut centers = init_centers(points, dim, k, seed);
    // Labels of step (a) of the previous iteration, before any empty-cluster fix.
    let mut prev: Vec<usize> = vec![usize::MAX; n];
    for _ in 0..iters {
        let (labels, fit) = assign_points(points, dim, &centers, opts.assign);
        if labels == prev {
            break;
        }
        prev = labels.clone();
        let mut labels = labels;
        centers = update_centers(points, dim, k, &mut labels, &fit, opts.normalize);
    }
    Ok(centers)
}

/// Step 2: k distinct training rows, drawn with `next_below(n)`; a repeat draws again.
fn init_centers(points: &[f32], dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = points.len() / dim;
    let mut rng = SplitMix64::new(seed);
    let mut taken = vec![false; n];
    let mut centers = Vec::with_capacity(k * dim);
    while centers.len() < k * dim {
        let r = rng.next_below(n as u64) as usize;
        if !taken[r] {
            taken[r] = true;
            centers.extend_from_slice(&points[r * dim..(r + 1) * dim]);
        }
    }
    centers
}

/// The best center of `p`, and its fit (higher is better: dot, or minus the distance).
/// Ties go to the lower center index.
#[inline]
pub fn best_center(p: &[f32], centers: &[f32], dim: usize, assign: Assign) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (c, center) in centers.chunks_exact(dim).enumerate() {
        let fit = match assign {
            Assign::Dot => dot(p, center),
            Assign::L2 => -l2_sq(p, center),
        };
        if fit > best.1 {
            best = (c, fit);
        }
    }
    best
}

/// Step 3, first half: each point's best center and fit, in parallel.
fn assign_points(
    points: &[f32],
    dim: usize,
    centers: &[f32],
    assign: Assign,
) -> (Vec<usize>, Vec<f32>) {
    points
        .par_chunks_exact(dim)
        .map(|p| best_center(p, centers, dim, assign))
        .unzip()
}

/// Step 3, second half, and step 4: means, the empty-cluster rule, and normalization.
fn update_centers(
    points: &[f32],
    dim: usize,
    k: usize,
    assign: &mut [usize],
    fit: &[f32],
    normalize: bool,
) -> Vec<f32> {
    let mut sums = vec![0.0f64; k * dim];
    let mut counts = vec![0usize; k];
    for (p, &c) in points.chunks_exact(dim).zip(assign.iter()) {
        counts[c] += 1;
        for (s, &x) in sums[c * dim..(c + 1) * dim].iter_mut().zip(p) {
            *s += x as f64;
        }
    }
    fix_empty_clusters(points, dim, assign, fit, &mut sums, &mut counts);
    let mut centers = vec![0.0f32; k * dim];
    for c in 0..k {
        let out = &mut centers[c * dim..(c + 1) * dim];
        let inv = 1.0 / counts[c].max(1) as f64;
        for (o, &s) in out.iter_mut().zip(&sums[c * dim..(c + 1) * dim]) {
            *o = (s * inv) as f32;
        }
        if normalize {
            normalize_in_place(out);
        }
    }
    centers
}

/// Step 4: each empty cluster, in index order, takes the worst-fit training point.
/// Only points whose cluster has at least 2 members are candidates, so no cluster
/// becomes empty by donating. Ties go to the lower point index.
fn fix_empty_clusters(
    points: &[f32],
    dim: usize,
    assign: &mut [usize],
    fit: &[f32],
    sums: &mut [f64],
    counts: &mut [usize],
) {
    let k = counts.len();
    for c in 0..k {
        if counts[c] > 0 {
            continue;
        }
        let worst = (0..assign.len())
            .filter(|&i| counts[assign[i]] > 1)
            .min_by(|&a, &b| fit[a].total_cmp(&fit[b]).then(a.cmp(&b)));
        let Some(i) = worst else { continue };
        let old = assign[i];
        let p = &points[i * dim..(i + 1) * dim];
        for (s, &x) in sums[old * dim..(old + 1) * dim].iter_mut().zip(p) {
            *s -= x as f64;
        }
        counts[old] -= 1;
        for (s, &x) in sums[c * dim..(c + 1) * dim].iter_mut().zip(p) {
            *s = x as f64;
        }
        counts[c] = 1;
        assign[i] = c;
    }
}

/// Scales `v` to length 1. A zero vector stays zero.
pub fn normalize_in_place(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}
