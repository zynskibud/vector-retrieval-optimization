//! DiskANN / Vamana graph on disk with PQ codes in RAM (CONTRACT 6.7).
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.90 at l=100 on the dev set, for metric=ip and metric=l2.
//!
//! Not implemented yet (Wave 2). `build` returns an error, and `bench` exits with code 1.

use crate::{AnnIndex, BuildContext, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
        .with("r", Int(64))
        .with("l_build", Int(100))
        .with("alpha", Float(1.2))
        .with("pq_m", Int(48))
        .with("metric", Str("ip".into()))
}

pub fn search_defaults() -> Params {
    Params::new()
        .with("l", Int(100))
        .with("beam", Int(4))
        .with("rerank", Int(100))
        .with("io", Str("mmap".into()))
}

pub struct DiskannIndex {
    times: BuildTimes,
}

pub fn build(
    _vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
    _ctx: &BuildContext,
) -> Result<DiskannIndex, String> {
    Err("not implemented".into())
}

pub fn search(
    _index: &DiskannIndex,
    _query: &[f32],
    _k: usize,
    _params: &Params,
) -> Result<SearchResult, String> {
    Err("not implemented".into())
}

pub fn index_bytes(_index: &DiskannIndex) -> u64 {
    0
}

impl AnnIndex for DiskannIndex {
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
