//! Hierarchical navigable small world graph (CONTRACT 6.6).
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.95 at ef=64 on the dev set.
//!
//! Not implemented yet (Wave 2). `build` returns an error, and `bench` exits with code 1.

use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
        .with("m", Int(16))
        .with("ef_construct", Int(100))
}

pub fn search_defaults() -> Params {
    Params::new().with("ef", Int(64))
}

pub struct HnswIndex {
    times: BuildTimes,
}

pub fn build(
    _vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
) -> Result<HnswIndex, String> {
    Err("not implemented".into())
}

pub fn search(
    _index: &HnswIndex,
    _query: &[f32],
    _k: usize,
    _params: &Params,
) -> Result<SearchResult, String> {
    Err("not implemented".into())
}

pub fn index_bytes(_index: &HnswIndex) -> u64 {
    0
}

impl AnnIndex for HnswIndex {
    fn search(&self, query: &[f32], k: usize, params: &Params) -> Result<SearchResult, String> {
        search(self, query, k, params)
    }
    fn index_bytes(&self) -> u64 {
        index_bytes(self)
    }
    fn build_times(&self) -> BuildTimes {
        self.times
    }
}
