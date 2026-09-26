//! Hand-built vector indexes that follow `indexes/CONTRACT.md`.
//!
//! Every index module exposes the same interface:
//! `build(vectors, params, threads, seed) -> index`, `search(index, query, k, params)`,
//! and `index_bytes(index)`. The [`AnnIndex`] trait lets `main` dispatch on `--index`.

pub mod diskann;
pub mod distance;
pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod ivf_pq;
pub mod kmeans;
pub mod npy;
pub mod params;
pub mod pq;
pub mod splitmix;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub use npy::Matrix;
pub use params::{ParamValue, Params};

/// The result of one query: `k` IDs and scores, best first (ties: lower ID first).
/// A short result is padded with ID -1 and score `-inf`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub ids: Vec<i64>,
    pub scores: Vec<f32>,
    /// Dot products (or table lookups) done for this query, if the index counts them.
    pub distance_computations: Option<u64>,
    /// Other per-query counters, for example DiskANN's `disk_reads`. `bench` writes the
    /// mean over all queries of each key to the search run's `"extra"` object.
    pub counters: BTreeMap<String, f64>,
}

/// Wall time of the build steps, in seconds (CONTRACT section 4).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BuildTimes {
    pub train_s: f64,
    pub add_s: f64,
}

/// The common interface of all built indexes.
pub trait AnnIndex: Send + Sync {
    fn search(&self, query: &[f32], k: usize, params: &Params) -> Result<SearchResult, String>;
    fn index_bytes(&self) -> u64;
    fn build_times(&self) -> BuildTimes;
    /// Sets the data directory that holds `filter_<name>.npy` (CONTRACT 11).
    /// Indexes without the `filter` search key ignore it.
    fn set_filter_dir(&mut self, _dir: &str) {}
    /// Build-time keys for the top-level `"extra"` object of the output JSON.
    fn extra(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }
}

/// The index names of CONTRACT section 2, in order.
pub const INDEX_NAMES: [&str; 6] = ["flat", "ivf", "pq", "ivf_pq", "hnsw", "diskann"];

/// Default build parameters of an index, for a corpus of `n` rows.
/// Returns `None` for an unknown index name.
pub fn build_defaults(index: &str, n: usize) -> Option<Params> {
    Some(match index {
        "flat" => flat::build_defaults(n),
        "ivf" => ivf::build_defaults(n),
        "pq" => pq::build_defaults(n),
        "ivf_pq" => ivf_pq::build_defaults(n),
        "hnsw" => hnsw::build_defaults(n),
        "diskann" => diskann::build_defaults(n),
        _ => return None,
    })
}

/// Default search parameters of an index. Returns `None` for an unknown index name.
pub fn search_defaults(index: &str) -> Option<Params> {
    Some(match index {
        "flat" => flat::search_defaults(),
        "ivf" => ivf::search_defaults(),
        "pq" => pq::search_defaults(),
        "ivf_pq" => ivf_pq::search_defaults(),
        "hnsw" => hnsw::search_defaults(),
        "diskann" => diskann::search_defaults(),
        _ => return None,
    })
}

/// Build context that is not a parameter of the index itself.
#[derive(Debug, Clone, Default)]
pub struct BuildContext {
    /// Path of the output JSON. DiskANN writes `<out>.diskann` next to it.
    pub out_path: String,
}

/// The filter names of CONTRACT 11.1. `none` means no filter.
pub const FILTER_NAMES: [&str; 5] = ["none", "top50", "top10", "top1", "top01"];

/// Checks a `filter` value against [`FILTER_NAMES`].
pub fn check_filter_name(name: &str) -> Result<(), String> {
    if FILTER_NAMES.contains(&name) {
        Ok(())
    } else {
        Err(format!(
            "unknown filter: {name} (expected one of {})",
            FILTER_NAMES.join(", ")
        ))
    }
}

/// One filter mask: `pass[i]` is true when row i passes. `rows` counts the passing rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Mask {
    pub pass: Vec<bool>,
    pub rows: usize,
}

/// Filter masks of CONTRACT 11, loaded from `<data_dir>/filter_<name>.npy` on first use
/// and cached per name. Each mask holds one bool per corpus row (the first `n` rows).
#[derive(Debug, Default)]
pub struct FilterMasks {
    dir: Option<PathBuf>,
    n: usize,
    cache: Mutex<HashMap<String, Arc<Mask>>>,
}

impl FilterMasks {
    /// Masks for a corpus of `n` rows, read from `dir`.
    pub fn new(dir: impl Into<PathBuf>, n: usize) -> Self {
        Self {
            dir: Some(dir.into()),
            n,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The mask for the `filter` search parameter, or `None` for `none` or a missing key.
    /// An unknown name, a missing file, or a short file is an error.
    pub fn for_params(&self, params: &Params) -> Result<Option<Arc<Mask>>, String> {
        if !params.contains("filter") {
            return Ok(None);
        }
        self.get(params.get_str("filter")?)
    }

    /// The mask named `name`, or `None` for `none`.
    pub fn get(&self, name: &str) -> Result<Option<Arc<Mask>>, String> {
        check_filter_name(name)?;
        if name == "none" {
            return Ok(None);
        }
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = cache.get(name) {
            return Ok(Some(Arc::clone(m)));
        }
        let dir = self
            .dir
            .as_ref()
            .ok_or_else(|| format!("filter={name} needs a data directory"))?;
        let mask = npy::read_bool(&dir.join(format!("filter_{name}.npy")), Some(self.n))?;
        if mask.len() != self.n {
            return Err(format!(
                "filter_{name}.npy has {} rows, the corpus has {}",
                mask.len(),
                self.n
            ));
        }
        let rows = mask.iter().filter(|&&p| p).count();
        let mask = Arc::new(Mask { pass: mask, rows });
        cache.insert(name.to_string(), Arc::clone(&mask));
        Ok(Some(mask))
    }
}

/// Builds the named index. `params` must already hold every key, defaults filled in.
pub fn build(
    index: &str,
    vectors: Matrix,
    params: &Params,
    threads: usize,
    seed: u64,
    ctx: &BuildContext,
) -> Result<Box<dyn AnnIndex>, String> {
    fn boxed<T: AnnIndex + 'static>(r: Result<T, String>) -> Result<Box<dyn AnnIndex>, String> {
        r.map(|i| Box::new(i) as Box<dyn AnnIndex>)
    }
    match index {
        "flat" => boxed(flat::build(vectors, params, threads, seed)),
        "ivf" => boxed(ivf::build(vectors, params, threads, seed)),
        "pq" => boxed(pq::build(vectors, params, threads, seed)),
        "ivf_pq" => boxed(ivf_pq::build(vectors, params, threads, seed)),
        "hnsw" => boxed(hnsw::build(vectors, params, threads, seed)),
        "diskann" => boxed(diskann::build(vectors, params, threads, seed, ctx)),
        other => Err(format!("unknown index: {other}")),
    }
}
