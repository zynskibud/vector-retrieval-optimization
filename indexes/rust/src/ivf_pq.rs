//! IVF with PQ-coded residuals (CONTRACT 6.5).
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.45 at default params on the dev set, for metric=ip and metric=l2.
//!
//! Train: the shared k-means (dot product, normalized centers) on the first `train_size` rows
//! gives `nlist` coarse centers. Each training row gets its best center c, and its residual
//! r = x - c. Codebook j is the shared k-means on sub-vector j of the training residuals,
//! with k = 256, seed = seed + j, no normalization, and assignment by the metric (6.4.1).
//! Add: every row is assigned to its best center, and its residual is encoded to m uint8
//! codes. Lists are stored in CSR layout: `ids` holds the row IDs grouped by list, `codes`
//! holds the m codes of each row in the same order, and list `c` is
//! `ids[offsets[c]..offsets[c + 1]]`.
//!
//! Search: the `nprobe` best centers by dot product.
//! - `ip`: one table T (m x 256) from q against the residual codebooks, the same for every
//!   list. Score of a row in list c = q . c + sum_j T[j*256 + code[j]].
//! - `l2`: per probed list, q' = q - c and T_c[j][k] = -||q'_j - codebook[j][k]||^2.
//!   Score = sum_j T_c[j*256 + code[j]].
//!
//! `rerank` > 0 keeps the top `rerank` candidates and re-scores them with the full vectors
//! (dot product for ip, minus the squared distance for l2).

use crate::distance::{dot, l2_sq, TopK};
use crate::kmeans::{best_center, kmeans, Assign, KmeansOptions};
use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};
use rayon::prelude::*;
use std::time::Instant;

/// Centroids per codebook (nbits = 8).
const KSUB: usize = 256;

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
        .with("nlist", Int(1024))
        .with("iters", Int(20))
        .with("m", Int(48))
        .with("nbits", Int(8))
        .with("metric", Str("ip".into()))
        .with("train_size", Int(100_000))
}

pub fn search_defaults() -> Params {
    Params::new().with("nprobe", Int(8)).with("rerank", Int(0))
}

pub struct IvfPqIndex {
    /// Full corpus, kept for rerank.
    vectors: Matrix,
    /// Coarse centers, row-major (nlist, dim), L2-normalized.
    centers: Vec<f32>,
    nlist: usize,
    m: usize,
    /// dim / m.
    dsub: usize,
    assign: Assign,
    /// m residual codebooks, (m, 256, dsub) row-major.
    codebooks: Vec<f32>,
    /// Row IDs grouped by list (CSR values).
    ids: Vec<i32>,
    /// (N, m) codes in list order: row `ids[p]` has codes `codes[p*m..(p+1)*m]`.
    codes: Vec<u8>,
    /// nlist + 1 offsets into `ids`. The last one equals N.
    offsets: Vec<i32>,
    times: BuildTimes,
}

impl IvfPqIndex {
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn list_ids(&self) -> &[i32] {
        &self.ids
    }
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }
    /// The (N, m) codes, in list order.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }
    pub fn centers(&self) -> &[f32] {
        &self.centers
    }
}

/// Writes x - c into `out`.
#[inline]
fn residual(x: &[f32], c: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(c) {
        *o = a - b;
    }
}

pub fn build(
    vectors: Matrix,
    params: &Params,
    _threads: usize,
    seed: u64,
) -> Result<IvfPqIndex, String> {
    let nlist = params.get_usize("nlist")?;
    let iters = params.get_usize("iters")?;
    let m = params.get_usize("m")?;
    let nbits = params.get_usize("nbits")?;
    let metric = params.get_str("metric")?;
    let train_size = params.get_usize("train_size")?;
    if nbits != 8 {
        return Err(format!(
            "ivf_pq: only nbits=8 is supported, got nbits={nbits}"
        ));
    }
    let assign = match metric {
        "ip" => Assign::Dot,
        "l2" => Assign::L2,
        other => return Err(format!("ivf_pq: metric must be ip or l2, got {other}")),
    };
    let n = vectors.rows;
    let dim = vectors.cols;
    if m == 0 || !dim.is_multiple_of(m) {
        return Err(format!("ivf_pq: dim {dim} must be divisible by m={m}"));
    }
    if n > i32::MAX as usize {
        return Err(format!("ivf_pq stores int32 IDs, but N = {n}"));
    }
    if nlist == 0 || nlist > n {
        return Err(format!(
            "ivf_pq needs 1 <= nlist <= N, got nlist={nlist}, N={n}"
        ));
    }
    let dsub = dim / m;
    let train_n = train_size.clamp(nlist, n);
    if train_n < KSUB {
        return Err(format!(
            "ivf_pq: need at least {KSUB} training rows, got {train_n}"
        ));
    }

    // Train: coarse centers, then residual codebooks.
    let t0 = Instant::now();
    let train = &vectors.data[..train_n * dim];
    let centers = kmeans(
        train,
        dim,
        nlist,
        iters,
        seed,
        KmeansOptions {
            assign: Assign::Dot,
            normalize: true,
        },
    )?;
    let mut resid = vec![0.0f32; train_n * dim];
    resid
        .par_chunks_exact_mut(dim)
        .zip(train.par_chunks_exact(dim))
        .for_each(|(r, x)| {
            let (c, _) = best_center(x, &centers, dim, Assign::Dot);
            residual(x, &centers[c * dim..(c + 1) * dim], r);
        });
    let opts = KmeansOptions {
        assign,
        normalize: false,
    };
    let mut codebooks = Vec::with_capacity(m * KSUB * dsub);
    let mut sub = vec![0.0f32; train_n * dsub];
    for j in 0..m {
        for (dst, row) in sub.chunks_exact_mut(dsub).zip(resid.chunks_exact(dim)) {
            dst.copy_from_slice(&row[j * dsub..(j + 1) * dsub]);
        }
        let book = kmeans(&sub, dsub, KSUB, iters, seed.wrapping_add(j as u64), opts)?;
        codebooks.extend_from_slice(&book);
    }
    drop(resid);
    drop(sub);
    let train_s = t0.elapsed().as_secs_f64();

    // Add: assign and encode every row in parallel, then group into CSR.
    let t1 = Instant::now();
    let encoded: Vec<(usize, Vec<u8>)> = vectors
        .data
        .par_chunks_exact(dim)
        .map_init(
            || vec![0.0f32; dim],
            |r, x| {
                let (c, _) = best_center(x, &centers, dim, Assign::Dot);
                residual(x, &centers[c * dim..(c + 1) * dim], r);
                let code: Vec<u8> = (0..m)
                    .map(|j| {
                        let book = &codebooks[j * KSUB * dsub..(j + 1) * KSUB * dsub];
                        best_center(&r[j * dsub..(j + 1) * dsub], book, dsub, assign).0 as u8
                    })
                    .collect();
                (c, code)
            },
        )
        .collect();
    let mut counts = vec![0usize; nlist];
    for (c, _) in &encoded {
        counts[*c] += 1;
    }
    let mut offsets = vec![0i32; nlist + 1];
    for c in 0..nlist {
        offsets[c + 1] = offsets[c] + counts[c] as i32;
    }
    // Rows go in ascending ID order within each list.
    let mut cursor: Vec<usize> = offsets[..nlist].iter().map(|&o| o as usize).collect();
    let mut ids = vec![0i32; n];
    let mut codes = vec![0u8; n * m];
    for (i, (c, code)) in encoded.into_iter().enumerate() {
        let p = cursor[c];
        ids[p] = i as i32;
        codes[p * m..(p + 1) * m].copy_from_slice(&code);
        cursor[c] += 1;
    }
    let add_s = t1.elapsed().as_secs_f64();

    Ok(IvfPqIndex {
        vectors,
        centers,
        nlist,
        m,
        dsub,
        assign,
        codebooks,
        ids,
        codes,
        offsets,
        times: BuildTimes { train_s, add_s },
    })
}

/// Fills `table` (m x 256) from `q` against the codebooks: dot product, or minus the squared distance.
fn fill_table(index: &IvfPqIndex, q: &[f32], assign: Assign, table: &mut [f32]) {
    let (m, dsub) = (index.m, index.dsub);
    for j in 0..m {
        let qj = &q[j * dsub..(j + 1) * dsub];
        let book = &index.codebooks[j * KSUB * dsub..(j + 1) * KSUB * dsub];
        for (t, c) in table[j * KSUB..(j + 1) * KSUB]
            .iter_mut()
            .zip(book.chunks_exact(dsub))
        {
            *t = match assign {
                Assign::Dot => dot(qj, c),
                Assign::L2 => -l2_sq(qj, c),
            };
        }
    }
}

pub fn search(
    index: &IvfPqIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let nprobe = params.get_usize("nprobe")?.min(index.nlist);
    let rerank = params.get_usize("rerank")?;
    let (m, dim) = (index.m, index.vectors.cols);

    // The nprobe best centers. Ties go to the lower center index.
    let mut probe = TopK::new(nprobe);
    for (c, center) in index.centers.chunks_exact(dim).enumerate() {
        let s = dot(query, center);
        if s >= probe.threshold() {
            probe.push(c as i64, s);
        }
    }
    let (lists, center_scores) = probe.into_sorted();

    let mut table = vec![0.0f32; m * KSUB];
    let mut qres = vec![0.0f32; dim];
    if index.assign == Assign::Dot {
        fill_table(index, query, Assign::Dot, &mut table);
    }

    let keep = if rerank > 0 { rerank } else { k };
    let mut top = TopK::new(keep);
    let mut scanned = 0u64;
    for (&c, &qc) in lists.iter().zip(&center_scores) {
        if c < 0 {
            continue;
        }
        let c = c as usize;
        let base = match index.assign {
            Assign::Dot => qc,
            Assign::L2 => {
                residual(query, &index.centers[c * dim..(c + 1) * dim], &mut qres);
                fill_table(index, &qres, Assign::L2, &mut table);
                0.0
            }
        };
        let (lo, hi) = (index.offsets[c] as usize, index.offsets[c + 1] as usize);
        scanned += (hi - lo) as u64;
        for (&id, code) in index.ids[lo..hi]
            .iter()
            .zip(index.codes[lo * m..hi * m].chunks_exact(m))
        {
            let s: f32 = base
                + code
                    .iter()
                    .enumerate()
                    .map(|(j, &b)| table[j * KSUB + b as usize])
                    .sum::<f32>();
            if s >= top.threshold() {
                top.push(id as i64, s);
            }
        }
    }
    let (ids, scores) = top.into_sorted();
    let mut dc = index.nlist as u64 + scanned;
    if rerank == 0 {
        return Ok(SearchResult {
            ids,
            scores,
            distance_computations: Some(dc),
            counters: Default::default(),
        });
    }

    let mut exact = TopK::new(k);
    for &id in ids.iter().filter(|&&id| id >= 0) {
        let x = index.vectors.row(id as usize);
        let s = match index.assign {
            Assign::Dot => dot(query, x),
            Assign::L2 => -l2_sq(query, x),
        };
        dc += 1;
        exact.push(id, s);
    }
    let (ids, scores) = exact.into_sorted();
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(dc),
        counters: Default::default(),
    })
}

/// IVF part (centers + int32 IDs + offsets) + PQ part (codebooks + codes). CONTRACT section 4.
pub fn index_bytes(index: &IvfPqIndex) -> u64 {
    let nlist = index.nlist as u64;
    let dim = index.vectors.cols as u64;
    nlist * dim * 4
        + index.ids.len() as u64 * 4
        + (nlist + 1) * 4
        + index.codebooks.len() as u64 * 4
        + index.codes.len() as u64
}

impl AnnIndex for IvfPqIndex {
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
        m.insert("id_bytes".into(), 4.into());
        m
    }
}
