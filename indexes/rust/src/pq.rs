//! Product quantization (CONTRACT 6.4): m codebooks of 256 centroids, uint8 codes, table scan.
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.50 at default params on the dev set, for metric=ip and metric=l2.
//!
//! Not implemented yet (Wave 2). `build` returns an error, and `bench` exits with code 1.

use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
        .with("m", Int(48))
        .with("nbits", Int(8))
        .with("metric", Str("ip".into()))
        .with("train_size", Int(100_000))
        .with("iters", Int(20))
}

pub fn search_defaults() -> Params {
    Params::new().with("rerank", Int(0))
}

pub struct PqIndex {
    times: BuildTimes,
}

pub fn build(
    _vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
) -> Result<PqIndex, String> {
    Err("not implemented".into())
}

pub fn search(
    _index: &PqIndex,
    _query: &[f32],
    _k: usize,
    _params: &Params,
) -> Result<SearchResult, String> {
    Err("not implemented".into())
}

pub fn index_bytes(_index: &PqIndex) -> u64 {
    0
}

impl AnnIndex for PqIndex {
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
