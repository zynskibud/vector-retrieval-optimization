//! Exact search (CONTRACT 6.1): score every row, keep the top k with a heap.
//!
//! Filter (CONTRACT 11.3): iterate all rows and skip each row that fails the mask,
//! so only passing rows are scored. `distance_computations` = scored rows.

use crate::distance::{dot, TopK};
use crate::{AnnIndex, BuildTimes, FilterMasks, Matrix, ParamValue::*, Params, SearchResult};

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
}

pub fn search_defaults() -> Params {
    Params::new().with("filter", Str("none".into()))
}

/// The corpus array is the whole index.
pub struct FlatIndex {
    vectors: Matrix,
    filters: FilterMasks,
}

impl FlatIndex {
    /// Sets the directory that holds `filter_<name>.npy` (CONTRACT 11).
    pub fn set_filter_dir(&mut self, dir: &str) {
        self.filters = FilterMasks::new(dir, self.vectors.rows);
    }
}

pub fn build(
    vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
) -> Result<FlatIndex, String> {
    Ok(FlatIndex {
        vectors,
        filters: FilterMasks::default(),
    })
}

/// Unfiltered search. Ignores `filter`; use [`search_filtered`] for CONTRACT 11.
pub fn search(index: &FlatIndex, query: &[f32], k: usize, _params: &Params) -> SearchResult {
    search_filtered(index, query, k, &Params::new()).expect("no filter, no error")
}

/// Search with the `filter` search parameter (missing key = `none`).
/// `extra.filter_rows` = corpus rows that pass; absent for `none`.
pub fn search_filtered(
    index: &FlatIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let v = &index.vectors;
    let mask = index.filters.for_params(params)?;
    let mut top = TopK::new(k);
    let mut scored = 0u64;
    let filter_rows = mask.as_ref().map(|m| m.rows);
    match mask.as_deref() {
        None => {
            for (i, row) in v.data.chunks_exact(v.cols).enumerate() {
                let s = dot(query, row);
                if s > top.threshold() {
                    top.push(i as i64, s);
                }
            }
            scored = v.rows as u64;
        }
        Some(mask) => {
            for (i, row) in v.data.chunks_exact(v.cols).enumerate() {
                if !mask.pass[i] {
                    continue;
                }
                scored += 1;
                let s = dot(query, row);
                if s > top.threshold() {
                    top.push(i as i64, s);
                }
            }
        }
    }
    let (ids, scores) = top.into_sorted();
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(scored),
        counters: filter_rows
            .map(|r| ("filter_rows".to_string(), r as f64))
            .into_iter()
            .collect(),
    })
}

/// Flat stores nothing beyond the corpus (CONTRACT section 4).
pub fn index_bytes(_index: &FlatIndex) -> u64 {
    0
}

impl AnnIndex for FlatIndex {
    fn set_filter_dir(&mut self, dir: &str) {
        FlatIndex::set_filter_dir(self, dir);
    }
    fn search(&self, query: &[f32], k: usize, params: &Params) -> Result<SearchResult, String> {
        search_filtered(self, query, k, params)
    }
    fn index_bytes(&self) -> u64 {
        index_bytes(self)
    }
    fn build_times(&self) -> BuildTimes {
        BuildTimes::default()
    }
}
