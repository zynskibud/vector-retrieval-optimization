//! Product quantization (CONTRACT 6.4): m codebooks of 256 centroids, uint8 codes, table scan.
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.50 at default params on the dev set, for metric=ip and metric=l2.
//!
//! Build: the first `train_size` rows are split into m sub-vectors of dim/m values. Codebook j
//! is the shared k-means (6.2) on sub-vector j, with k = 256, seed = seed + j, no normalization,
//! and assignment by the metric (6.4.1). Every row is then encoded to m uint8 codes.
//! Search: one table T (m x 256) per query, score of row i = sum_j T[j*256 + code[i*m + j]].

use crate::distance::{dot, l2_sq, TopK};
use crate::kmeans::{best_center, kmeans, Assign, KmeansOptions};
use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};
use rayon::prelude::*;
use std::time::Instant;

/// Centroids per codebook (nbits = 8).
const KSUB: usize = 256;

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
    /// Full corpus, kept for rerank.
    vectors: Matrix,
    m: usize,
    /// dim / m.
    dsub: usize,
    assign: Assign,
    /// m codebooks, each (256, dsub), contiguous: codebooks[(j*256 + c)*dsub ..].
    codebooks: Vec<f32>,
    /// (N, m) codes, row-major.
    codes: Vec<u8>,
    times: BuildTimes,
}

impl PqIndex {
    /// The (N, m) codes, row-major.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
    /// The m codebooks, (m, 256, dim/m) row-major.
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }
    pub fn m(&self) -> usize {
        self.m
    }
}

pub fn build(
    vectors: Matrix,
    params: &Params,
    _threads: usize,
    seed: u64,
) -> Result<PqIndex, String> {
    let m = params.get_usize("m")?;
    let nbits = params.get_usize("nbits")?;
    let metric = params.get_str("metric")?;
    let train_size = params.get_usize("train_size")?;
    let iters = params.get_usize("iters")?;
    if nbits != 8 {
        return Err(format!("pq: only nbits=8 is supported, got nbits={nbits}"));
    }
    let assign = match metric {
        "ip" => Assign::Dot,
        "l2" => Assign::L2,
        other => return Err(format!("pq: metric must be ip or l2, got {other}")),
    };
    let dim = vectors.cols;
    if m == 0 || !dim.is_multiple_of(m) {
        return Err(format!("pq: dim {dim} must be divisible by m={m}"));
    }
    let dsub = dim / m;
    let n = vectors.rows;
    let train_n = train_size.min(n);
    if train_n < KSUB {
        return Err(format!(
            "pq: need at least {KSUB} training rows, got {train_n}"
        ));
    }
    let opts = KmeansOptions {
        assign,
        normalize: false,
    };

    // Train: one k-means per sub-vector position.
    let t0 = Instant::now();
    let mut codebooks = Vec::with_capacity(m * KSUB * dsub);
    let mut sub = vec![0.0f32; train_n * dsub];
    for j in 0..m {
        for (dst, row) in sub
            .chunks_exact_mut(dsub)
            .zip(vectors.data.chunks_exact(dim).take(train_n))
        {
            dst.copy_from_slice(&row[j * dsub..(j + 1) * dsub]);
        }
        let centers = kmeans(&sub, dsub, KSUB, iters, seed.wrapping_add(j as u64), opts)?;
        codebooks.extend_from_slice(&centers);
    }
    let train_s = t0.elapsed().as_secs_f64();

    // Add: encode every row, in parallel across rows.
    let t1 = Instant::now();
    let mut codes = vec![0u8; n * m];
    codes
        .par_chunks_exact_mut(m)
        .zip(vectors.data.par_chunks_exact(dim))
        .for_each(|(code, row)| {
            for (j, c) in code.iter_mut().enumerate() {
                let book = &codebooks[j * KSUB * dsub..(j + 1) * KSUB * dsub];
                let (best, _) = best_center(&row[j * dsub..(j + 1) * dsub], book, dsub, assign);
                *c = best as u8;
            }
        });
    let add_s = t1.elapsed().as_secs_f64();

    Ok(PqIndex {
        vectors,
        m,
        dsub,
        assign,
        codebooks,
        codes,
        times: BuildTimes { train_s, add_s },
    })
}

pub fn search(
    index: &PqIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let rerank = params.get_usize("rerank")?;
    let (m, dsub) = (index.m, index.dsub);
    let n = index.vectors.rows;

    // Table T (m x 256): higher is better in both metrics.
    let mut table = vec![0.0f32; m * KSUB];
    for j in 0..m {
        let qj = &query[j * dsub..(j + 1) * dsub];
        let book = &index.codebooks[j * KSUB * dsub..(j + 1) * KSUB * dsub];
        for (t, c) in table[j * KSUB..(j + 1) * KSUB]
            .iter_mut()
            .zip(book.chunks_exact(dsub))
        {
            *t = match index.assign {
                Assign::Dot => dot(qj, c),
                Assign::L2 => -l2_sq(qj, c),
            };
        }
    }

    let keep = if rerank > 0 { rerank } else { k };
    let mut top = TopK::new(keep);
    for (i, code) in index.codes.chunks_exact(m).enumerate() {
        let s: f32 = code
            .iter()
            .enumerate()
            .map(|(j, &c)| table[j * KSUB + c as usize])
            .sum();
        if s > top.threshold() {
            top.push(i as i64, s);
        }
    }
    let (ids, scores) = top.into_sorted();
    if rerank == 0 {
        return Ok(SearchResult {
            ids,
            scores,
            distance_computations: Some(n as u64),
            counters: Default::default(),
        });
    }

    let mut exact = TopK::new(k);
    let mut rescored = 0u64;
    for &id in ids.iter().filter(|&&id| id >= 0) {
        let x = index.vectors.row(id as usize);
        let s = match index.assign {
            Assign::Dot => dot(query, x),
            Assign::L2 => -l2_sq(query, x),
        };
        rescored += 1;
        exact.push(id, s);
    }
    let (ids, scores) = exact.into_sorted();
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(n as u64 + rescored),
        counters: Default::default(),
    })
}

/// Codebooks (m x 256 x dim/m x 4 bytes) + codes (N x m bytes). CONTRACT section 4.
pub fn index_bytes(index: &PqIndex) -> u64 {
    (index.codebooks.len() * 4 + index.codes.len()) as u64
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
