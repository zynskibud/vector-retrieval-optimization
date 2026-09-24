//! Exact search (CONTRACT 6.1): score every row, keep the top k with a heap.

use crate::distance::{dot, TopK};
use crate::{AnnIndex, BuildTimes, Matrix, Params, SearchResult};

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
}

pub fn search_defaults() -> Params {
    Params::new()
}

/// The corpus array is the whole index.
pub struct FlatIndex {
    vectors: Matrix,
}

pub fn build(
    vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
) -> Result<FlatIndex, String> {
    Ok(FlatIndex { vectors })
}

pub fn search(index: &FlatIndex, query: &[f32], k: usize, _params: &Params) -> SearchResult {
    let v = &index.vectors;
    let mut top = TopK::new(k);
    for (i, row) in v.data.chunks_exact(v.cols).enumerate() {
        let s = dot(query, row);
        if s > top.threshold() {
            top.push(i as i64, s);
        }
    }
    let (ids, scores) = top.into_sorted();
    SearchResult {
        ids,
        scores,
        distance_computations: Some(v.rows as u64),
        counters: Default::default(),
    }
}

/// Flat stores nothing beyond the corpus (CONTRACT section 4).
pub fn index_bytes(_index: &FlatIndex) -> u64 {
    0
}

impl AnnIndex for FlatIndex {
    fn search(&self, query: &[f32], k: usize, params: &Params) -> Result<SearchResult, String> {
        Ok(search(self, query, k, params))
    }
    fn index_bytes(&self) -> u64 {
        index_bytes(self)
    }
    fn build_times(&self) -> BuildTimes {
        BuildTimes::default()
    }
}
