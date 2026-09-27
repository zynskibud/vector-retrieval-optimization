//! Exact search (CONTRACT 6.1): score every row, keep the top k with a heap.
//!
//! Filter (CONTRACT 11.3): iterate all rows and skip each row that fails the mask,
//! so only passing rows are scored. `distance_computations` = scored rows.
//!
//! Changes (CONTRACT 13.3): delete sets bits in a tombstone bit set, and the scan skips
//! tombstoned rows (they are not scored). Update overwrites the row in place. Compact
//! copies the live rows, in row order, into a new array and keeps a position -> row ID
//! map (int32), so returned IDs stay the row IDs; the bit set is dropped.
//! `index_bytes` is 0 for a plain build (the corpus array is the index, section 4).
//! After a change, compaction shrinks the corpus array itself, so the array counts:
//! `index_bytes` = vector array + bit set (before compaction) or vector array + ID map
//! (after).
//!
//! Save and load (CONTRACT 15.1): sections `vectors` (N x dim) and `tombstones`. After a
//! compaction the array holds only the live rows, so save writes all N rows back in row
//! order: live rows from the array, dropped rows as zero vectors with their tombstone bit
//! set. The loaded index skips those rows, so it returns the same IDs and scores.

use crate::distance::{dot, TopK};
use crate::vro;
use std::path::Path;
use crate::{
    check_update, AnnIndex, BuildTimes, FilterMasks, Matrix, ParamValue::*, Params, SearchResult,
    Tombstones,
};

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
    /// Corpus rows at build time (row IDs are 0..rows).
    rows: usize,
    /// Deleted rows (CONTRACT 13.3), before compaction.
    tombstones: Option<Tombstones>,
    /// After compaction: row ID of each position in `vectors`.
    row_ids: Option<Vec<i32>>,
    /// True once a delete, update, or compaction ran: `index_bytes` then counts the array.
    changed: bool,
}

impl FlatIndex {
    /// Sets the directory that holds `filter_<name>.npy` (CONTRACT 11).
    pub fn set_filter_dir(&mut self, dir: &str) {
        self.filters = FilterMasks::new(dir, self.rows);
    }

    /// Marks the rows with `mask[i]` true as deleted.
    pub fn delete(&mut self, mask: &[bool]) -> Result<(), String> {
        if self.row_ids.is_some() {
            return Err("flat: delete after compaction is not supported".into());
        }
        if mask.len() != self.rows {
            return Err(format!("delete mask has {} rows, the corpus has {}", mask.len(), self.rows));
        }
        let mut all = self.tombstones.as_ref().map_or(vec![false; self.rows], Tombstones::to_mask);
        all.iter_mut().zip(mask).for_each(|(a, &m)| *a |= m);
        self.tombstones = Some(Tombstones::from_mask(&all));
        self.changed = true;
        Ok(())
    }

    /// Overwrites the vectors of rows `ids`.
    pub fn update(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        if self.row_ids.is_some() {
            return Err("flat: update after compaction is not supported".into());
        }
        let dim = self.vectors.cols;
        check_update(ids, vectors, self.rows, dim)?;
        for (j, &id) in ids.iter().enumerate() {
            let id = id as usize;
            self.vectors.data[id * dim..(id + 1) * dim]
                .copy_from_slice(&vectors[j * dim..(j + 1) * dim]);
        }
        self.changed = true;
        Ok(())
    }

    /// Drops the tombstoned rows from the array; keeps a position -> row ID map.
    pub fn compact(&mut self) -> Result<(), String> {
        if self.row_ids.is_some() {
            return Ok(());
        }
        let dim = self.vectors.cols;
        let t = self.tombstones.take();
        let live: Vec<usize> = (0..self.rows)
            .filter(|&i| !t.as_ref().is_some_and(|t| t.is_deleted(i)))
            .collect();
        let mut data = Vec::with_capacity(live.len() * dim);
        for &i in &live {
            data.extend_from_slice(self.vectors.row(i));
        }
        self.vectors = Matrix {
            data,
            rows: live.len(),
            cols: dim,
        };
        self.row_ids = Some(live.into_iter().map(|i| i as i32).collect());
        self.changed = true;
        Ok(())
    }
}

impl FlatIndex {
    /// Writes the `.vro` file (CONTRACT 15.1). Returns the file size in bytes.
    pub fn save(&self, path: &Path, build_params: &Params, seed: u64) -> Result<u64, String> {
        let (n, dim) = (self.rows, self.vectors.cols);
        let expanded;
        let (vectors, tomb) = match &self.row_ids {
            None => (&self.vectors.data, vro::tombstone_bytes(self.tombstones.as_ref(), n)),
            Some(r) => {
                let mut data = vec![0f32; n * dim];
                let mut dead = vec![true; n];
                for (pos, &id) in r.iter().enumerate() {
                    let id = id as usize;
                    data[id * dim..(id + 1) * dim].copy_from_slice(self.vectors.row(pos));
                    dead[id] = false;
                }
                expanded = data;
                (&expanded, Tombstones::from_mask(&dead).to_bytes())
            }
        };
        let sections = [
            vro::Section { name: "vectors", shape: vec![n, dim], data: vro::Data::F32(vectors) },
            vro::Section { name: "tombstones", shape: vec![n.div_ceil(8)], data: vro::Data::U8(&tomb) },
        ];
        vro::write(path, "flat", n, dim, build_params, seed, &sections)
    }
}

/// Loads a flat index from an open `.vro` file (header already checked).
pub fn load(r: &mut vro::Reader) -> Result<FlatIndex, String> {
    let vectors = r.read_vectors()?;
    let tombstones = r.read_tombstones()?;
    Ok(FlatIndex {
        rows: vectors.rows,
        vectors,
        filters: FilterMasks::default(),
        changed: tombstones.is_some(),
        tombstones,
        row_ids: None,
    })
}

pub fn build(
    vectors: Matrix,
    _params: &Params,
    _threads: usize,
    _seed: u64,
) -> Result<FlatIndex, String> {
    let rows = vectors.rows;
    Ok(FlatIndex {
        vectors,
        filters: FilterMasks::default(),
        rows,
        tombstones: None,
        row_ids: None,
        changed: false,
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
    let row_ids = index.row_ids.as_deref();
    let tomb = index.tombstones.as_ref();
    if mask.is_none() && tomb.is_none() {
        for (i, row) in v.data.chunks_exact(v.cols).enumerate() {
            let s = dot(query, row);
            if s > top.threshold() {
                top.push(i as i64, s);
            }
        }
        scored = v.rows as u64;
    } else {
        let pass = mask.as_ref().map(|m| m.pass.as_slice());
        for (i, row) in v.data.chunks_exact(v.cols).enumerate() {
            let id = row_ids.map_or(i, |r| r[i] as usize);
            if tomb.is_some_and(|t| t.is_deleted(i)) || pass.is_some_and(|p| !p[id]) {
                continue;
            }
            scored += 1;
            let s = dot(query, row);
            if s > top.threshold() {
                top.push(i as i64, s);
            }
        }
    }
    let (mut ids, scores) = top.into_sorted();
    // Positions map to row IDs in the same order, so ties keep the lower-ID-first rule.
    if let Some(r) = row_ids {
        ids.iter_mut()
            .filter(|id| **id >= 0)
            .for_each(|id| *id = r[*id as usize] as i64);
    }
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

/// Flat stores nothing beyond the corpus (CONTRACT section 4). After a change
/// (CONTRACT 13), the vector array + the bit set, or + the ID map after compaction.
pub fn index_bytes(index: &FlatIndex) -> u64 {
    if !index.changed {
        return 0;
    }
    index.vectors.data.len() as u64 * 4
        + index.tombstones.as_ref().map_or(0, Tombstones::bytes)
        + index.row_ids.as_ref().map_or(0, |r| r.len() as u64 * 4)
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
    fn delete(&mut self, mask: &[bool]) -> Result<(), String> {
        FlatIndex::delete(self, mask)
    }
    fn update(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        FlatIndex::update(self, ids, vectors)
    }
    fn compact(&mut self) -> Result<(), String> {
        FlatIndex::compact(self)
    }
    fn save(&self, path: &Path, build_params: &Params, seed: u64) -> Result<u64, String> {
        FlatIndex::save(self, path, build_params, seed)
    }
}
