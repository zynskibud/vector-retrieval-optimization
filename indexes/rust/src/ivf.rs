//! Inverted file index (CONTRACT 6.3): k-means centers, CSR ID lists, probe `nprobe` lists.
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.75 at nprobe=8 on the dev set.
//!
//! Build: `train` runs the shared k-means (dot product, normalized centers) on the first
//! `train_size` rows. `add` assigns every corpus row to its best center in parallel and
//! stores the lists in CSR layout: `ids` holds the row IDs grouped by list, and list `c`
//! is `ids[offsets[c]..offsets[c + 1]]`.

use crate::distance::{dot, TopK};
use crate::kmeans::{best_center, kmeans, Assign, KmeansOptions};
use crate::params::default_train_size;
use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};
use rayon::prelude::*;
use std::time::Instant;

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
    vectors: Matrix,
    /// Centers, row-major (nlist, dim), L2-normalized.
    centers: Vec<f32>,
    nlist: usize,
    /// Row IDs grouped by list (CSR values).
    ids: Vec<i32>,
    /// nlist + 1 offsets into `ids`. The last one equals N.
    offsets: Vec<i32>,
    times: BuildTimes,
}

impl IvfIndex {
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    pub fn list_ids(&self) -> &[i32] {
        &self.ids
    }
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }
}

pub fn build(
    vectors: Matrix,
    params: &Params,
    _threads: usize,
    seed: u64,
) -> Result<IvfIndex, String> {
    let nlist = params.get_usize("nlist")?;
    let train_size = params.get_usize("train_size")?;
    let iters = params.get_usize("iters")?;
    let n = vectors.rows;
    let dim = vectors.cols;
    if n > i32::MAX as usize {
        return Err(format!("ivf stores int32 IDs, but N = {n}"));
    }
    if nlist == 0 || nlist > n {
        return Err(format!(
            "ivf needs 1 <= nlist <= N, got nlist={nlist}, N={n}"
        ));
    }
    let train_n = train_size.clamp(nlist, n);

    let t0 = Instant::now();
    let centers = kmeans(
        &vectors.data[..train_n * dim],
        dim,
        nlist,
        iters,
        seed,
        KmeansOptions {
            assign: Assign::Dot,
            normalize: true,
        },
    )?;
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let labels: Vec<usize> = vectors
        .data
        .par_chunks_exact(dim)
        .map(|row| best_center(row, &centers, dim, Assign::Dot).0)
        .collect();
    let mut counts = vec![0usize; nlist];
    for &c in &labels {
        counts[c] += 1;
    }
    let mut offsets = vec![0i32; nlist + 1];
    for c in 0..nlist {
        offsets[c + 1] = offsets[c] + counts[c] as i32;
    }
    // Rows go in ascending ID order within each list.
    let mut cursor: Vec<usize> = offsets[..nlist].iter().map(|&o| o as usize).collect();
    let mut ids = vec![0i32; n];
    for (i, &c) in labels.iter().enumerate() {
        ids[cursor[c]] = i as i32;
        cursor[c] += 1;
    }
    let add_s = t1.elapsed().as_secs_f64();

    Ok(IvfIndex {
        vectors,
        centers,
        nlist,
        ids,
        offsets,
        times: BuildTimes { train_s, add_s },
    })
}

pub fn search(
    index: &IvfIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let nprobe = params.get_usize("nprobe")?.min(index.nlist);
    let dim = index.vectors.cols;

    // Pick the nprobe best centers. Ties go to the lower center index.
    let mut probe = TopK::new(nprobe);
    for (c, center) in index.centers.chunks_exact(dim).enumerate() {
        let s = dot(query, center);
        if s >= probe.threshold() {
            probe.push(c as i64, s);
        }
    }
    let (lists, _) = probe.into_sorted();

    let mut top = TopK::new(k);
    let mut scanned = 0u64;
    for &c in lists.iter().filter(|&&c| c >= 0) {
        let c = c as usize;
        let (lo, hi) = (index.offsets[c] as usize, index.offsets[c + 1] as usize);
        scanned += (hi - lo) as u64;
        for &id in &index.ids[lo..hi] {
            let s = dot(query, index.vectors.row(id as usize));
            if s >= top.threshold() {
                top.push(id as i64, s);
            }
        }
    }
    let (ids, scores) = top.into_sorted();
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(index.nlist as u64 + scanned),
        counters: Default::default(),
    })
}

/// Centers (nlist x dim x 4) + one int32 ID per row + (nlist + 1) int32 offsets.
pub fn index_bytes(index: &IvfIndex) -> u64 {
    let nlist = index.nlist as u64;
    nlist * index.vectors.cols as u64 * 4 + index.ids.len() as u64 * 4 + (nlist + 1) * 4
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
    fn extra(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        m.insert("id_type".into(), "int32".into());
        m
    }
}
