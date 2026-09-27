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
pub mod vro;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
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
    /// Adds rows `ids` (the next rows in row order) with their `vectors` (row-major)
    /// while queries run (CONTRACT 12.2). Only hnsw supports it.
    fn insert(&self, _ids: &[i64], _vectors: &[f32]) -> Result<(), String> {
        Err("this index does not support inserts".into())
    }
    /// Runs the repair pass once after inserts. Returns (step A edges, step B edges).
    fn repair(&self) -> Result<(u64, u64), String> {
        Err("this index has no repair pass".into())
    }
    /// True if `search` is safe and lock-free for many client threads (CONTRACT 12).
    fn supports_concurrency(&self) -> bool {
        false
    }
    /// Marks every row `i` with `mask[i]` as deleted (CONTRACT 13.3: a tombstone).
    /// `mask` holds one value per corpus row. Only flat, ivf, hnsw support it.
    fn delete(&mut self, _mask: &[bool]) -> Result<(), String> {
        Err("this index does not support deletes".into())
    }
    /// Replaces the vectors of rows `ids` (row-major `vectors`) under the same IDs
    /// (CONTRACT 13.3). Only flat, ivf, hnsw support it.
    fn update(&mut self, _ids: &[i64], _vectors: &[f32]) -> Result<(), String> {
        Err("this index does not support updates".into())
    }
    /// Compaction of CONTRACT 13.3: drops the tombstoned rows (flat, ivf) or rebuilds
    /// the graph from the live rows (hnsw).
    fn compact(&mut self) -> Result<(), String> {
        Err("this index does not support compaction".into())
    }
    /// The in-place repair of `--compact-mode repair` (CONTRACT 13.3). Only hnsw has one.
    fn compact_repair(&mut self) -> Result<(), String> {
        Err("this index has no in-place repair; use --compact-mode rebuild".into())
    }
    /// Writes the index to a `.vro` file (CONTRACT 15.1) with the header's `build_params`
    /// and `seed`. Returns the file size in bytes. Only flat, ivf, hnsw support it.
    fn save(&self, _path: &Path, _build_params: &Params, _seed: u64) -> Result<u64, String> {
        Err("this index does not support save (CONTRACT 15: flat, ivf, hnsw)".into())
    }
    /// Build-time keys for the top-level `"extra"` object of the output JSON.
    fn extra(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }
}

/// The tombstone bit set of CONTRACT 13.3: bit i is set when row i is deleted.
/// It holds one bit per corpus row, so it takes N/8 bytes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tombstones {
    bits: Vec<u64>,
    rows: usize,
    deleted: usize,
}

impl Tombstones {
    /// A bit set for `rows` rows with the bits of `mask` set (`mask.len()` must equal `rows`).
    pub fn from_mask(mask: &[bool]) -> Self {
        let mut t = Self {
            bits: vec![0; mask.len().div_ceil(64)],
            rows: mask.len(),
            deleted: 0,
        };
        for (i, _) in mask.iter().enumerate().filter(|(_, &d)| d) {
            t.bits[i / 64] |= 1 << (i % 64);
            t.deleted += 1;
        }
        t
    }
    /// True when row `i` is deleted.
    #[inline]
    pub fn is_deleted(&self, i: usize) -> bool {
        (self.bits[i / 64] >> (i % 64)) & 1 == 1
    }
    /// Number of deleted rows.
    pub fn deleted(&self) -> usize {
        self.deleted
    }
    /// Number of rows the bit set covers.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Memory of the bit set: N/8 bytes, rounded up.
    pub fn bytes(&self) -> u64 {
        self.rows.div_ceil(8) as u64
    }
    /// The bit set as ceil(N/8) bytes: bit i (LSB first within a byte) is row i
    /// (CONTRACT 15.1). The u64 words are little-endian, so this is their byte image.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = self.bits.iter().flat_map(|w| w.to_le_bytes()).collect();
        out.truncate(self.rows.div_ceil(8));
        out
    }
    /// The inverse of [`Tombstones::to_bytes`] for `rows` rows. Bits past `rows` are ignored.
    pub fn from_bytes(bytes: &[u8], rows: usize) -> Self {
        let mask: Vec<bool> = (0..rows)
            .map(|i| bytes.get(i / 8).is_some_and(|b| (b >> (i % 8)) & 1 == 1))
            .collect();
        Self::from_mask(&mask)
    }
    /// One bool per row: true when the row is deleted.
    pub fn to_mask(&self) -> Vec<bool> {
        (0..self.rows).map(|i| self.is_deleted(i)).collect()
    }
}

/// The delete change sets of CONTRACT 13.1.
pub const DELETE_NAMES: [&str; 3] = ["del10", "del30", "del50"];
/// The update change sets of CONTRACT 13.1.
pub const UPDATE_NAMES: [&str; 1] = ["upd10"];
/// Indexes that accept `--delete`, `--update`, and `--compact` (CONTRACT 13).
pub const CHANGE_INDEXES: [&str; 3] = ["flat", "ivf", "hnsw"];

/// Checks the arguments of [`AnnIndex::update`]: IDs in `0..rows`, no repeat,
/// and `ids.len() * dim` values.
pub fn check_update(ids: &[i64], vectors: &[f32], rows: usize, dim: usize) -> Result<(), String> {
    if vectors.len() != ids.len() * dim {
        return Err(format!(
            "update: {} values for {} rows of dim {dim}",
            vectors.len(),
            ids.len()
        ));
    }
    let mut seen = vec![false; rows];
    for &id in ids {
        if id < 0 || id as usize >= rows {
            return Err(format!("update: row {id} is not in 0..{rows}"));
        }
        if std::mem::replace(&mut seen[id as usize], true) {
            return Err(format!("update: row {id} appears twice"));
        }
    }
    Ok(())
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

/// Builds the named index on only the first `build_rows` rows; the other rows are
/// added later with [`AnnIndex::insert`] (CONTRACT 12.2). Only hnsw supports it.
pub fn build_partial(
    index: &str,
    vectors: Matrix,
    params: &Params,
    threads: usize,
    seed: u64,
    build_rows: usize,
) -> Result<Box<dyn AnnIndex>, String> {
    if index != "hnsw" {
        return Err(format!("{index} does not support a partial build"));
    }
    hnsw::build_partial(vectors, params, threads, seed, build_rows)
        .map(|i| Box::new(i) as Box<dyn AnnIndex>)
}

/// Indexes that can be saved to and loaded from a `.vro` file (CONTRACT 15).
pub const SAVE_INDEXES: [&str; 3] = ["flat", "ivf", "hnsw"];

/// Checks a `.vro` header against what the caller expects (CONTRACT 15.1): the index
/// name, the dimension (if given), the set of build parameter keys of that index,
/// and the value of every key in `expected` (for example the `--build` flags).
pub fn check_vro_header(
    h: &vro::Header,
    index: &str,
    dim: Option<usize>,
    expected: &Params,
) -> Result<(), String> {
    if h.index != index {
        return Err(format!("the file holds index '{}', expected '{index}'", h.index));
    }
    if let Some(d) = dim {
        if h.dim != d {
            return Err(format!("the file has dim {}, expected {d}", h.dim));
        }
    }
    let defaults = build_defaults(index, h.n).ok_or_else(|| format!("unknown index: {index}"))?;
    let have: Vec<&String> = h.build_params.0.keys().collect();
    let want: Vec<&String> = defaults.0.keys().collect();
    if have != want {
        return Err(format!("the file's build_params keys {have:?} are not the keys of {index} {want:?}"));
    }
    for (k, v) in &expected.0 {
        match h.build_params.0.get(k) {
            Some(fv) if vro::param_eq(fv, v) => {}
            Some(fv) => {
                return Err(format!("build parameter {k}: the file has {fv:?}, requested {v:?}"))
            }
            None => return Err(format!("build parameter {k} is not in the file")),
        }
    }
    Ok(())
}

/// Loads a `.vro` file written by any language (CONTRACT 15). The header must match
/// `index`, `dim` (if given), and `expected` (see [`check_vro_header`]). `threads` is
/// the thread count that a later hnsw compaction rebuild uses. No rebuild, no repair.
pub fn load(
    index: &str,
    path: &Path,
    expected: &Params,
    dim: Option<usize>,
    threads: usize,
) -> Result<(Box<dyn AnnIndex>, vro::Header), String> {
    let mut r = vro::Reader::open(path)?;
    check_vro_header(&r.header, index, dim, expected)?;
    let header = r.header.clone();
    let idx: Box<dyn AnnIndex> = match index {
        "flat" => Box::new(flat::load(&mut r)?),
        "ivf" => Box::new(ivf::load(&mut r)?),
        "hnsw" => Box::new(hnsw::load(&mut r, threads)?),
        other => return Err(format!("{other} does not support load (CONTRACT 15: flat, ivf, hnsw)")),
    };
    Ok((idx, header))
}
