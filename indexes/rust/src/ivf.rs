//! Inverted file index (CONTRACT 6.3): k-means centers, CSR ID lists, probe `nprobe` lists.
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.75 at nprobe=8 on the dev set.
//!
//! Build: `train` runs the shared k-means (dot product, normalized centers) on the first
//! `train_size` rows. `add` assigns every corpus row to its best center in parallel and
//! stores the lists in CSR layout: `ids` holds the row IDs grouped by list, and list `c`
//! is `ids[offsets[c]..offsets[c + 1]]`.
//!
//! Filter (CONTRACT 11.3): scan the `nprobe` lists as before and skip each row that
//! fails the mask. `distance_computations` = nlist + scored (passing) rows.
//!
//! Changes (CONTRACT 13.3): delete sets bits in a tombstone bit set, and the scan skips
//! tombstoned rows in the probed lists (not scored, not counted). Update overwrites the
//! row's vector, assigns it to its best center, and rebuilds the CSR lists once for the
//! whole batch (one O(N) counting pass), so the ID moves to its new list. Compact
//! rebuilds the CSR lists without the tombstoned IDs and drops the bit set. The lists
//! keep the row IDs, so the corpus array is kept as it is (it is not part of
//! `index_bytes`); the centers are not retrained.

use crate::distance::{dot, TopK};
use crate::kmeans::{best_center, kmeans, Assign, KmeansOptions};
use crate::params::default_train_size;
use crate::{
    check_update, AnnIndex, BuildTimes, FilterMasks, Matrix, ParamValue::*, Params, SearchResult,
    Tombstones,
};
use rayon::prelude::*;
use std::time::Instant;

pub fn build_defaults(n: usize) -> Params {
    Params::new()
        .with("nlist", Int(1024))
        .with("train_size", Int(default_train_size(n, 1024) as i64))
        .with("iters", Int(20))
}

pub fn search_defaults() -> Params {
    Params::new()
        .with("nprobe", Int(8))
        .with("filter", Str("none".into()))
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
    filters: FilterMasks,
    /// Deleted rows (CONTRACT 13.3), before compaction.
    tombstones: Option<Tombstones>,
}

/// CSR lists from one label per row: rows in ascending ID order within each list.
/// Rows with `skip(i)` true are left out.
fn csr(labels: &[usize], nlist: usize, skip: impl Fn(usize) -> bool) -> (Vec<i32>, Vec<i32>) {
    let mut counts = vec![0usize; nlist];
    for (i, &c) in labels.iter().enumerate() {
        if !skip(i) {
            counts[c] += 1;
        }
    }
    let mut offsets = vec![0i32; nlist + 1];
    for c in 0..nlist {
        offsets[c + 1] = offsets[c] + counts[c] as i32;
    }
    let mut cursor: Vec<usize> = offsets[..nlist].iter().map(|&o| o as usize).collect();
    let mut ids = vec![0i32; offsets[nlist] as usize];
    for (i, &c) in labels.iter().enumerate() {
        if !skip(i) {
            ids[cursor[c]] = i as i32;
            cursor[c] += 1;
        }
    }
    (ids, offsets)
}

impl IvfIndex {
    /// Sets the directory that holds `filter_<name>.npy` (CONTRACT 11).
    pub fn set_filter_dir(&mut self, dir: &str) {
        self.filters = FilterMasks::new(dir, self.vectors.rows);
    }
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    pub fn list_ids(&self) -> &[i32] {
        &self.ids
    }
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    /// List of every row in the lists, from the CSR layout (rows not in a list: `usize::MAX`).
    fn labels(&self) -> Vec<usize> {
        let mut labels = vec![usize::MAX; self.vectors.rows];
        for c in 0..self.nlist {
            let (lo, hi) = (self.offsets[c] as usize, self.offsets[c + 1] as usize);
            for &id in &self.ids[lo..hi] {
                labels[id as usize] = c;
            }
        }
        labels
    }

    /// Marks the rows with `mask[i]` true as deleted.
    pub fn delete(&mut self, mask: &[bool]) -> Result<(), String> {
        let n = self.vectors.rows;
        if mask.len() != n {
            return Err(format!("delete mask has {} rows, the corpus has {n}", mask.len()));
        }
        let mut all = self.tombstones.as_ref().map_or(vec![false; n], Tombstones::to_mask);
        all.iter_mut().zip(mask).for_each(|(a, &m)| *a |= m);
        self.tombstones = Some(Tombstones::from_mask(&all));
        Ok(())
    }

    /// Overwrites the vectors of rows `ids` and moves each ID to the list of its new center.
    pub fn update(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        let dim = self.vectors.cols;
        check_update(ids, vectors, self.vectors.rows, dim)?;
        let mut labels = self.labels();
        for (j, &id) in ids.iter().enumerate() {
            let id = id as usize;
            let v = &vectors[j * dim..(j + 1) * dim];
            self.vectors.data[id * dim..(id + 1) * dim].copy_from_slice(v);
            if labels[id] != usize::MAX {
                labels[id] = best_center(v, &self.centers, dim, Assign::Dot).0;
            }
        }
        let (l, o) = csr(&labels, self.nlist, |i| labels[i] == usize::MAX);
        self.ids = l;
        self.offsets = o;
        Ok(())
    }

    /// Drops the tombstoned IDs from the lists and the bit set.
    pub fn compact(&mut self) -> Result<(), String> {
        let Some(t) = self.tombstones.take() else {
            return Ok(());
        };
        let labels = self.labels();
        let (l, o) = csr(&labels, self.nlist, |i| {
            labels[i] == usize::MAX || t.is_deleted(i)
        });
        self.ids = l;
        self.offsets = o;
        Ok(())
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
    let (ids, offsets) = csr(&labels, nlist, |_| false);
    let add_s = t1.elapsed().as_secs_f64();

    Ok(IvfIndex {
        vectors,
        centers,
        nlist,
        ids,
        offsets,
        times: BuildTimes { train_s, add_s },
        filters: FilterMasks::default(),
        tombstones: None,
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
    let mask = index.filters.for_params(params)?;
    let pass = mask.as_ref().map(|m| m.pass.as_slice());
    let filter_rows = mask.as_ref().map(|m| m.rows);
    let tomb = index.tombstones.as_ref();

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
        for &id in &index.ids[lo..hi] {
            if pass.is_some_and(|p| !p[id as usize])
                || tomb.is_some_and(|t| t.is_deleted(id as usize))
            {
                continue;
            }
            scanned += 1;
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
        counters: filter_rows
            .map(|r| ("filter_rows".to_string(), r as f64))
            .into_iter()
            .collect(),
    })
}

/// Centers (nlist x dim x 4) + one int32 ID per listed row + (nlist + 1) int32 offsets
/// + the tombstone bit set (N/8 bytes) while it exists.
pub fn index_bytes(index: &IvfIndex) -> u64 {
    let nlist = index.nlist as u64;
    nlist * index.vectors.cols as u64 * 4
        + index.ids.len() as u64 * 4
        + (nlist + 1) * 4
        + index.tombstones.as_ref().map_or(0, Tombstones::bytes)
}

impl AnnIndex for IvfIndex {
    fn set_filter_dir(&mut self, dir: &str) {
        IvfIndex::set_filter_dir(self, dir);
    }
    fn search(&self, query: &[f32], k: usize, params: &Params) -> Result<SearchResult, String> {
        search(self, query, k, params)
    }
    fn index_bytes(&self) -> u64 {
        index_bytes(self)
    }
    fn build_times(&self) -> BuildTimes {
        self.times
    }
    fn delete(&mut self, mask: &[bool]) -> Result<(), String> {
        IvfIndex::delete(self, mask)
    }
    fn update(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        IvfIndex::update(self, ids, vectors)
    }
    fn compact(&mut self) -> Result<(), String> {
        IvfIndex::compact(self)
    }
    fn extra(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        m.insert("id_type".into(), "int32".into());
        m
    }
}
