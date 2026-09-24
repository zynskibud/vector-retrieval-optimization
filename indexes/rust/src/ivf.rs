//! Inverted file index (CONTRACT 6.3): k-means centers, CSR ID lists, probe `nprobe` lists.
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.80 at nprobe=8 on the dev set.
//!
//! Not implemented yet (Wave 2). `build` returns an error, and `bench` exits with code 1.

use crate::params::default_train_size;
use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};

pub fn build_defaults(n: usize) -> Params {
    Params::new()
        .with("nlist", Int(1024))
        .with("train_size", Int(default_train_size(n, 1024) as i64))
        .with("iters", Int(20))
}

pub fn search_defaults() -> Params {
    Params::new().with("nprobe", Int(8))
}

pub struct IvfIndex {
    times: BuildTimes,
}

pub fn build(
    _vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
) -> Result<IvfIndex, String> {
    Err("not implemented".into())
}

pub fn search(
    _index: &IvfIndex,
    _query: &[f32],
    _k: usize,
    _params: &Params,
) -> Result<SearchResult, String> {
    Err("not implemented".into())
}

pub fn index_bytes(_index: &IvfIndex) -> u64 {
    0
}

impl AnnIndex for IvfIndex {
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
