//! DiskANN / Vamana graph on disk with PQ codes in RAM (CONTRACT 6.7).
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.90 at l=100 on the dev set, for metric=ip and metric=l2.
//!
//! Build (Subramanya et al. 2019):
//! 1. Entry point = the row with the highest dot product with the mean row (medoid).
//! 2. Every node gets `r` random out-edges (`next_below(N)`, seed `seed`, row order,
//!    self and repeats skipped).
//! 3. Two passes in row order, alpha = 1.0 then `alpha`: greedy search from the entry
//!    with list size `l_build` gives the visited set V; robust-prune(i, V u N(i), alpha, r);
//!    each out-neighbor j gets the edge j -> i, and j is pruned when it exceeds r.
//!    The graph uses full vectors. The pruning distance is d(a, b) = 2 - 2 a.b, which is the
//!    squared Euclidean distance of unit vectors, so the graph uses only dot products.
//!    With `threads == 1` the passes are strictly sequential. With more threads, rayon workers
//!    claim chunks of 64 consecutive rows; each read-modify-write of one node's list holds the
//!    mutex of that node's stripe (one lock at a time, so no deadlock). Reads during the greedy
//!    search are lock-free on atomics, as in `hnsw`.
//! 4. PQ (6.4, m = `pq_m`, codebooks trained on all rows, seed + j, no normalization,
//!    assignment by `metric`), then all rows encoded. PQ time is `train_s`.
//! 5. `<out>.diskann`: one record per node = dim f32 LE, then r i32 LE out-edges (-1 = empty),
//!    padded to a multiple of 4096 bytes. The corpus is then dropped.
//!
//! Search: beam search over PQ scores, records read through a memory map (`io=mmap`) or with
//! uncached `pread` (`io=nocache`, 6.7.1), then rerank of the top `rerank` list entries with
//! the full vectors of their records, scored by the dot product in both metrics.

use crate::distance::{dot, l2_sq, TopK};
use crate::kmeans::{best_center, kmeans, Assign, KmeansOptions};
use crate::splitmix::SplitMix64;
use crate::{AnnIndex, BuildContext, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};
use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering as AtOrd};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

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

/// Centroids per PQ codebook (nbits = 8).
const KSUB: usize = 256;
/// Record alignment on disk (CONTRACT 6.7.1).
const ALIGN: usize = 4096;
/// Mutexes in the lock stripe of the parallel build.
const LOCK_STRIPES: usize = 1 << 16;
/// Rows per chunk claimed by one parallel worker.
const CHUNK: usize = 64;
/// PQ k-means iterations (6.4 default).
const PQ_ITERS: usize = 20;

/// A scored node. Greater = better: higher score, or equal score and lower ID.
#[derive(Debug, Clone, Copy)]
struct Cand {
    score: f32,
    id: u32,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| o.id.cmp(&self.id))
    }
}

/// A sorted candidate list (best first) of at most `cap` entries, with an expanded flag.
struct CandList {
    cap: usize,
    items: Vec<(Cand, bool)>,
}

impl CandList {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            items: Vec::with_capacity(cap + 1),
        }
    }

    fn insert(&mut self, c: Cand) {
        if self.items.len() >= self.cap && c <= self.items[self.items.len() - 1].0 {
            return;
        }
        let pos = self.items.partition_point(|(x, _)| *x > c);
        self.items.insert(pos, (c, false));
        self.items.truncate(self.cap);
    }

    /// Marks up to `n` best unexpanded entries as expanded and returns them.
    fn take_unexpanded(&mut self, n: usize, out: &mut Vec<Cand>) {
        out.clear();
        for (c, e) in self.items.iter_mut() {
            if out.len() >= n {
                break;
            }
            if !*e {
                *e = true;
                out.push(*c);
            }
        }
    }
}

/// Generation-stamped visited set of N entries.
struct Visited {
    stamp: Vec<u32>,
    generation: u32,
}

impl Visited {
    fn new(n: usize) -> Self {
        Self {
            stamp: vec![0; n],
            generation: 0,
        }
    }
    fn next(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.fill(0);
            self.generation = 1;
        }
    }
    /// Marks `v`; returns `true` if it was not marked before.
    #[inline]
    fn mark(&mut self, v: u32) -> bool {
        let s = &mut self.stamp[v as usize];
        if *s == self.generation {
            false
        } else {
            *s = self.generation;
            true
        }
    }
}

// ---------------------------------------------------------------- build graph

/// Out-edge lists during the build: `r` fixed slots per node and a count.
struct Graph {
    r: usize,
    slots: Vec<AtomicI32>,
    counts: Vec<AtomicU32>,
    locks: Vec<Mutex<()>>,
}

impl Graph {
    fn read(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        let b = node as usize * self.r;
        let c = (self.counts[node as usize].load(AtOrd::Acquire) as usize).min(self.r);
        for slot in &self.slots[b..b + c] {
            let v = slot.load(AtOrd::Relaxed);
            if v >= 0 {
                out.push(v as u32);
            }
        }
    }

    fn write(&self, node: u32, ids: &[u32]) {
        let b = node as usize * self.r;
        for (slot, &id) in self.slots[b..b + self.r].iter().zip(ids) {
            slot.store(id as i32, AtOrd::Relaxed);
        }
        self.counts[node as usize].store(ids.len().min(self.r) as u32, AtOrd::Release);
    }

    fn lock(&self, node: u32) -> MutexGuard<'_, ()> {
        self.locks[node as usize % self.locks.len()]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
}

/// Pruning distance of unit vectors from a dot product: 2 - 2 a.b = ||a - b||^2.
#[inline]
fn dist(score: f32) -> f32 {
    2.0 - 2.0 * score
}

/// Robust prune (Vamana, Algorithm 2). `cands` holds scores against node `p`. Repeatedly
/// keeps the nearest remaining candidate p*, then removes every candidate x with
/// alpha * d(p*, x) <= d(p, x). Stops at `r` kept nodes.
fn robust_prune(p: u32, cands: &mut Vec<Cand>, alpha: f32, r: usize, vectors: &Matrix) -> Vec<u32> {
    cands.retain(|c| c.id != p);
    cands.sort_unstable_by(|a, b| b.cmp(a));
    cands.dedup_by_key(|c| c.id);
    let mut kept = Vec::with_capacity(r);
    let mut alive: Vec<Cand> = std::mem::take(cands);
    while !alive.is_empty() && kept.len() < r {
        let best = alive[0];
        kept.push(best.id);
        if kept.len() == r {
            break;
        }
        let bv = vectors.row(best.id as usize);
        alive = alive[1..]
            .iter()
            .copied()
            .filter(|x| alpha * dist(dot(bv, vectors.row(x.id as usize))) > dist(x.score))
            .collect();
    }
    kept
}

/// Greedy search (Vamana, Algorithm 1) from `entry` with list size `l`.
/// Returns the visited (expanded) set V with scores against `q`.
fn greedy_visit(
    q: &[f32],
    vectors: &Matrix,
    g: &Graph,
    entry: u32,
    l: usize,
    vis: &mut Visited,
    buf: &mut Vec<u32>,
) -> Vec<Cand> {
    vis.next();
    let mut list = CandList::new(l);
    vis.mark(entry);
    list.insert(Cand {
        score: dot(q, vectors.row(entry as usize)),
        id: entry,
    });
    let mut visited = Vec::new();
    let mut step = Vec::with_capacity(1);
    loop {
        list.take_unexpanded(1, &mut step);
        let Some(&c) = step.first() else { break };
        visited.push(c);
        g.read(c.id, buf);
        for &u in buf.iter() {
            if vis.mark(u) {
                list.insert(Cand {
                    score: dot(q, vectors.row(u as usize)),
                    id: u,
                });
            }
        }
    }
    visited
}

struct Builder<'a> {
    vectors: &'a Matrix,
    g: Graph,
    entry: u32,
    l_build: usize,
}

impl Builder<'_> {
    /// One Vamana step for row `i` with pruning slack `alpha`.
    fn insert(&self, i: u32, alpha: f32, vis: &mut Visited, buf: &mut Vec<u32>) {
        let r = self.g.r;
        let q = self.vectors.row(i as usize);
        let mut cands = greedy_visit(q, self.vectors, &self.g, self.entry, self.l_build, vis, buf);
        let chosen = {
            let _g = self.g.lock(i);
            self.g.read(i, buf);
            cands.extend(buf.iter().map(|&u| Cand {
                score: dot(q, self.vectors.row(u as usize)),
                id: u,
            }));
            let chosen = robust_prune(i, &mut cands, alpha, r, self.vectors);
            self.g.write(i, &chosen);
            chosen
        };
        for &j in &chosen {
            let _g = self.g.lock(j);
            self.g.read(j, buf);
            if buf.contains(&i) {
                continue;
            }
            buf.push(i);
            if buf.len() <= r {
                self.g.write(j, buf);
            } else {
                let jv = self.vectors.row(j as usize);
                let mut c: Vec<Cand> = buf
                    .iter()
                    .map(|&u| Cand {
                        score: dot(jv, self.vectors.row(u as usize)),
                        id: u,
                    })
                    .collect();
                let kept = robust_prune(j, &mut c, alpha, r, self.vectors);
                self.g.write(j, &kept);
            }
        }
    }

    fn pass(&self, alpha: f32, threads: usize) {
        let n = self.vectors.rows;
        if threads <= 1 {
            let mut vis = Visited::new(n);
            let mut buf = Vec::new();
            for i in 0..n as u32 {
                self.insert(i, alpha, &mut vis, &mut buf);
            }
        } else {
            (0..n.div_ceil(CHUNK)).into_par_iter().for_each_init(
                || (Visited::new(n), Vec::new()),
                |(vis, buf), c| {
                    for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                        self.insert(i as u32, alpha, vis, buf);
                    }
                },
            );
        }
    }
}

/// Row with the highest dot product with the mean row; ties to the lowest row.
fn medoid(vectors: &Matrix) -> u32 {
    let dim = vectors.cols;
    let mut sum = vec![0.0f64; dim];
    for row in vectors.data.chunks_exact(dim) {
        for (s, &x) in sum.iter_mut().zip(row) {
            *s += x as f64;
        }
    }
    let mean: Vec<f32> = sum
        .iter()
        .map(|s| (s / vectors.rows as f64) as f32)
        .collect();
    let mut best = Cand {
        score: f32::NEG_INFINITY,
        id: 0,
    };
    for (i, row) in vectors.data.chunks_exact(dim).enumerate() {
        let c = Cand {
            score: dot(&mean, row),
            id: i as u32,
        };
        if c > best {
            best = c;
        }
    }
    best.id
}

/// Random initial graph: `r` distinct out-edges per node, row order, one PRNG.
fn random_graph(n: usize, r: usize, seed: u64) -> Graph {
    let degree = r.min(n.saturating_sub(1));
    let mut rng = SplitMix64::new(seed);
    let slots: Vec<AtomicI32> = (0..n * r).map(|_| AtomicI32::new(-1)).collect();
    let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    let mut list: Vec<u32> = Vec::with_capacity(degree);
    for i in 0..n {
        list.clear();
        while list.len() < degree {
            let j = rng.next_below(n as u64) as u32;
            if j as usize != i && !list.contains(&j) {
                list.push(j);
            }
        }
        for (s, &j) in slots[i * r..].iter().zip(&list) {
            s.store(j as i32, AtOrd::Relaxed);
        }
        counts[i].store(degree as u32, AtOrd::Relaxed);
    }
    Graph {
        r,
        slots,
        counts,
        locks: (0..LOCK_STRIPES.min(n.max(1)))
            .map(|_| Mutex::new(()))
            .collect(),
    }
}

// ------------------------------------------------------------------------ PQ

/// Codebooks (m, 256, dsub) and codes (N, m) of CONTRACT 6.4, trained on all rows.
/// This repeats the train and encode loops of `pq::build`, which needs the corpus by value
/// and keeps it; here the corpus must be dropped after the file is written.
fn train_pq(
    vectors: &Matrix,
    m: usize,
    assign: Assign,
    seed: u64,
) -> Result<(Vec<f32>, Vec<u8>), String> {
    let (n, dim) = (vectors.rows, vectors.cols);
    let dsub = dim / m;
    let opts = KmeansOptions {
        assign,
        normalize: false,
    };
    let mut codebooks = Vec::with_capacity(m * KSUB * dsub);
    let mut sub = vec![0.0f32; n * dsub];
    for j in 0..m {
        for (dst, row) in sub
            .chunks_exact_mut(dsub)
            .zip(vectors.data.chunks_exact(dim))
        {
            dst.copy_from_slice(&row[j * dsub..(j + 1) * dsub]);
        }
        let centers = kmeans(
            &sub,
            dsub,
            KSUB,
            PQ_ITERS,
            seed.wrapping_add(j as u64),
            opts,
        )?;
        codebooks.extend_from_slice(&centers);
    }
    let mut codes = vec![0u8; n * m];
    codes
        .par_chunks_exact_mut(m)
        .zip(vectors.data.par_chunks_exact(dim))
        .for_each(|(code, row)| {
            for (j, c) in code.iter_mut().enumerate() {
                let book = &codebooks[j * KSUB * dsub..(j + 1) * KSUB * dsub];
                *c = best_center(&row[j * dsub..(j + 1) * dsub], book, dsub, assign).0 as u8;
            }
        });
    Ok((codebooks, codes))
}

// ----------------------------------------------------------------------- I/O

/// `fcntl(fd, F_NOCACHE, 1)` on macOS: reads and writes on this fd skip the page cache.
#[cfg(target_os = "macos")]
fn set_nocache(file: &File) -> Result<(), String> {
    use std::os::fd::AsRawFd;
    // SAFETY: the fd is open for the lifetime of `file`; F_NOCACHE takes an int argument
    // and touches no memory of this process.
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if rc == -1 {
        return Err(format!(
            "fcntl F_NOCACHE: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Linux: the uncached mode comes from O_DIRECT at open time.
#[cfg(not(target_os = "macos"))]
fn set_nocache(_file: &File) -> Result<(), String> {
    Ok(())
}

fn open_nocache(path: &str) -> Result<File, String> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_DIRECT);
    }
    let f = o.open(path).map_err(|e| format!("{path}: {e}"))?;
    set_nocache(&f)?;
    Ok(f)
}

/// `pread` of exactly `buf.len()` bytes at `offset`.
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::fd::AsRawFd;
    let mut done = 0usize;
    while done < buf.len() {
        let rest = &mut buf[done..];
        // SAFETY: `rest` is a valid writable buffer of `rest.len()` bytes, and the fd is open
        // for the lifetime of `file`. pread writes at most `rest.len()` bytes into it.
        let got = unsafe {
            libc::pread(
                file.as_raw_fd(),
                rest.as_mut_ptr().cast(),
                rest.len(),
                (offset + done as u64) as libc::off_t,
            )
        };
        if got < 0 {
            return Err(format!("pread: {}", std::io::Error::last_os_error()));
        }
        if got == 0 {
            return Err("pread: unexpected end of file".into());
        }
        done += got as usize;
    }
    Ok(())
}

/// Little-endian f32 values of `bytes`.
fn decode_f32(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// A byte buffer whose usable part starts at a 4096-byte aligned address (for O_DIRECT).
struct AlignedBuf {
    raw: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGN];
        let start = raw.as_ptr().align_offset(ALIGN);
        Self { raw, start, len }
    }
    fn get_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.start..self.start + self.len]
    }
    fn get(&self) -> &[u8] {
        &self.raw[self.start..self.start + self.len]
    }
}

// ---------------------------------------------------------------------- index

pub struct DiskannIndex {
    n: usize,
    dim: usize,
    r: usize,
    m: usize,
    dsub: usize,
    assign: Assign,
    codebooks: Vec<f32>,
    codes: Vec<u8>,
    entry: u32,
    path: String,
    record_bytes: usize,
    disk_bytes: u64,
    mean_out_degree: f64,
    build_threads: usize,
    times: BuildTimes,
    /// The open map (io=mmap) or uncached file (io=nocache). A switch from mmap to
    /// nocache invalidates and unmaps the map first (CONTRACT 6.7.1, item 1).
    handle: Mutex<Option<Handle>>,
    scratch: Mutex<Vec<Scratch>>,
}

/// Per-query working memory.
struct Scratch {
    vis: Visited,
    buf: AlignedBuf,
    /// Full vectors of expanded nodes, for the rerank.
    arena: Vec<f32>,
    arena_pos: HashMap<u32, usize>,
    nbrs: Vec<u32>,
}

impl DiskannIndex {
    pub fn entry_point(&self) -> u32 {
        self.entry
    }
    /// Path of the `.diskann` file.
    pub fn disk_path(&self) -> &str {
        &self.path
    }
    /// Bytes per node record on disk (a multiple of 4096).
    pub fn record_bytes(&self) -> usize {
        self.record_bytes
    }
    pub fn r(&self) -> usize {
        self.r
    }
    pub fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }
    pub fn mean_out_degree(&self) -> f64 {
        self.mean_out_degree
    }
}

pub fn build(
    vectors: Matrix,
    params: &Params,
    threads: usize,
    seed: u64,
    ctx: &BuildContext,
) -> Result<DiskannIndex, String> {
    let r = params.get_usize("r")?;
    let l_build = params.get_usize("l_build")?;
    let alpha = params.get_float("alpha")? as f32;
    let m = params.get_usize("pq_m")?;
    let assign = match params.get_str("metric")? {
        "ip" => Assign::Dot,
        "l2" => Assign::L2,
        other => return Err(format!("diskann: metric must be ip or l2, got {other}")),
    };
    let (n, dim) = (vectors.rows, vectors.cols);
    if r == 0 || l_build == 0 {
        return Err("diskann: r and l_build must be >= 1".into());
    }
    if n == 0 || n > i32::MAX as usize {
        return Err(format!("diskann: bad corpus size {n}"));
    }
    if m == 0 || !dim.is_multiple_of(m) {
        return Err(format!("diskann: dim {dim} must be divisible by pq_m={m}"));
    }
    if n < KSUB {
        return Err(format!(
            "diskann: need at least {KSUB} rows for PQ, got {n}"
        ));
    }

    // Graph (add_s part 1).
    let t_add = Instant::now();
    let entry = medoid(&vectors);
    let builder = Builder {
        vectors: &vectors,
        g: random_graph(n, r, seed),
        entry,
        l_build,
    };
    builder.pass(1.0, threads);
    builder.pass(alpha, threads);
    let mut add_s = t_add.elapsed().as_secs_f64();

    // PQ (train_s).
    let t_train = Instant::now();
    let (codebooks, codes) = train_pq(&vectors, m, assign, seed)?;
    let train_s = t_train.elapsed().as_secs_f64();

    // File (add_s part 2).
    let t_write = Instant::now();
    let path = format!("{}.diskann", ctx.out_path);
    let record_bytes = (dim * 4 + r * 4).div_ceil(ALIGN) * ALIGN;
    let mut edges_total = 0u64;
    {
        let file = File::create(&path).map_err(|e| format!("{path}: {e}"))?;
        // Keep the written pages out of the page cache, so io=nocache starts cold.
        set_nocache(&file)?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
        let mut rec = vec![0u8; record_bytes];
        let mut nb = Vec::with_capacity(r);
        for i in 0..n {
            rec.fill(0);
            for (dst, x) in rec.chunks_exact_mut(4).zip(vectors.row(i)) {
                dst.copy_from_slice(&x.to_le_bytes());
            }
            builder.g.read(i as u32, &mut nb);
            edges_total += nb.len() as u64;
            let e0 = dim * 4;
            for s in 0..r {
                let v: i32 = nb.get(s).map_or(-1, |&x| x as i32);
                rec[e0 + s * 4..e0 + s * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            w.write_all(&rec).map_err(|e| format!("{path}: {e}"))?;
        }
        let file = w.into_inner().map_err(|e| format!("{path}: {e}"))?;
        file.sync_all().map_err(|e| format!("{path}: {e}"))?;
    }
    add_s += t_write.elapsed().as_secs_f64();
    let disk_bytes = std::fs::metadata(&path)
        .map_err(|e| format!("{path}: {e}"))?
        .len();
    drop(builder);
    drop(vectors); // search must not hold the full vectors (6.7)

    Ok(DiskannIndex {
        n,
        dim,
        r,
        m,
        dsub: dim / m,
        assign,
        codebooks,
        codes,
        entry,
        path,
        record_bytes,
        disk_bytes,
        mean_out_degree: edges_total as f64 / n as f64,
        build_threads: threads,
        times: BuildTimes { train_s, add_s },
        handle: Mutex::new(None),
        scratch: Mutex::new(Vec::new()),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Io {
    Mmap,
    NoCache,
}

/// An open way to read records. Cloned per query (cheap `Arc` clone).
#[derive(Clone)]
enum Handle {
    Map(Arc<Mmap>),
    File(Arc<File>),
}

/// `msync(MS_INVALIDATE)` over the whole map: asks the OS to drop its cached pages.
fn invalidate_map(map: &Mmap) {
    if map.is_empty() {
        return;
    }
    // SAFETY: the pointer and length describe exactly this live mapping; MS_INVALIDATE
    // does not change the mapped bytes (the file is read-only after the build).
    unsafe {
        libc::msync(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MS_INVALIDATE,
        );
    }
}

/// Linux: `posix_fadvise(DONTNEED)` drops the file's pages from the page cache.
#[cfg(target_os = "linux")]
fn drop_page_cache(path: &str) {
    use std::os::fd::AsRawFd;
    if let Ok(f) = File::open(path) {
        // SAFETY: the fd is open for the lifetime of `f`; the call touches no process memory.
        unsafe {
            libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_page_cache(_path: &str) {}

impl DiskannIndex {
    /// The handle for `io`. A switch from mmap to nocache invalidates and unmaps the map
    /// before the uncached file opens; a switch back maps the file again.
    fn handle(&self, io: Io) -> Result<Handle, String> {
        let mut h = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        match (io, h.as_ref()) {
            (Io::Mmap, Some(Handle::Map(m))) => return Ok(Handle::Map(m.clone())),
            (Io::NoCache, Some(Handle::File(f))) => return Ok(Handle::File(f.clone())),
            _ => {}
        }
        if let Some(Handle::Map(m)) = h.take() {
            invalidate_map(&m);
            drop(m);
            drop_page_cache(&self.path);
        }
        let new = match io {
            Io::Mmap => {
                let f = File::open(&self.path).map_err(|e| format!("{}: {e}", self.path))?;
                // SAFETY: the file is written once during the build and never modified
                // while the index lives, so the mapped bytes do not change under us.
                let map = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap: {e}"))?;
                Handle::Map(Arc::new(map))
            }
            Io::NoCache => Handle::File(Arc::new(open_nocache(&self.path)?)),
        };
        *h = Some(new.clone());
        Ok(new)
    }

    /// Copies the record of `node` into `buf` and returns it.
    fn read_record<'a>(
        &self,
        h: &Handle,
        node: u32,
        buf: &'a mut AlignedBuf,
    ) -> Result<&'a [u8], String> {
        let off = node as usize * self.record_bytes;
        match h {
            Handle::Map(map) => {
                buf.get_mut()
                    .copy_from_slice(&map[off..off + self.record_bytes]);
            }
            Handle::File(f) => pread_exact(f, buf.get_mut(), off as u64)?,
        }
        Ok(buf.get())
    }

    fn table(&self, query: &[f32]) -> Vec<f32> {
        let (m, dsub) = (self.m, self.dsub);
        let mut table = vec![0.0f32; m * KSUB];
        for j in 0..m {
            let qj = &query[j * dsub..(j + 1) * dsub];
            let book = &self.codebooks[j * KSUB * dsub..(j + 1) * KSUB * dsub];
            for (t, c) in table[j * KSUB..(j + 1) * KSUB]
                .iter_mut()
                .zip(book.chunks_exact(dsub))
            {
                *t = match self.assign {
                    Assign::Dot => dot(qj, c),
                    Assign::L2 => -l2_sq(qj, c),
                };
            }
        }
        table
    }

    #[inline]
    fn pq_score(&self, table: &[f32], node: u32) -> f32 {
        let code = &self.codes[node as usize * self.m..(node as usize + 1) * self.m];
        code.iter()
            .enumerate()
            .map(|(j, &c)| table[j * KSUB + c as usize])
            .sum()
    }
}

pub fn search(
    index: &DiskannIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let l = params.get_usize("l")?.max(k).max(1);
    let beam = params.get_usize("beam")?.max(1);
    let rerank = params.get_usize("rerank")?;
    let io = match params.get_str("io")? {
        "mmap" => Io::Mmap,
        "nocache" => Io::NoCache,
        other => return Err(format!("diskann: io must be mmap or nocache, got {other}")),
    };
    let handle = index.handle(io)?;

    let mut s = index
        .scratch
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pop()
        .unwrap_or_else(|| Scratch {
            vis: Visited::new(index.n),
            buf: AlignedBuf::new(index.record_bytes),
            arena: Vec::new(),
            arena_pos: HashMap::new(),
            nbrs: Vec::new(),
        });
    s.vis.next();
    s.arena.clear();
    s.arena_pos.clear();

    let table = index.table(query);
    let dim = index.dim;
    let mut dc = 0u64;
    let mut reads = 0u64;
    let mut list = CandList::new(l);
    s.vis.mark(index.entry);
    list.insert(Cand {
        score: index.pq_score(&table, index.entry),
        id: index.entry,
    });
    dc += 1;
    let mut step = Vec::with_capacity(beam);
    loop {
        list.take_unexpanded(beam, &mut step);
        if step.is_empty() {
            break;
        }
        for c in &step {
            let Scratch {
                vis,
                buf,
                arena,
                arena_pos,
                nbrs,
            } = &mut s;
            let rec = index.read_record(&handle, c.id, buf)?;
            reads += 1;
            nbrs.clear();
            arena_pos.insert(c.id, arena.len());
            arena.extend(decode_f32(&rec[..dim * 4]));
            for b in rec[dim * 4..dim * 4 + index.r * 4].chunks_exact(4) {
                let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                if v >= 0 {
                    nbrs.push(v as u32);
                }
            }
            for &u in nbrs.iter() {
                if vis.mark(u) {
                    list.insert(Cand {
                        score: index.pq_score(&table, u),
                        id: u,
                    });
                    dc += 1;
                }
            }
        }
    }

    let (ids, scores) = if rerank == 0 {
        let mut top = TopK::new(k);
        for (c, _) in &list.items {
            top.push(c.id as i64, c.score);
        }
        top.into_sorted()
    } else {
        let mut top = TopK::new(k);
        for (c, _) in list.items.iter().take(rerank) {
            let pos = match s.arena_pos.get(&c.id) {
                Some(&p) => p,
                None => {
                    let rec = index.read_record(&handle, c.id, &mut s.buf)?;
                    reads += 1;
                    let p = s.arena.len();
                    s.arena.extend(decode_f32(&rec[..dim * 4]));
                    p
                }
            };
            let x = &s.arena[pos..pos + dim];
            // Rerank uses the dot product in both metrics (CONTRACT 6.7).
            let score = dot(query, x);
            dc += 1;
            top.push(c.id as i64, score);
        }
        top.into_sorted()
    };
    index
        .scratch
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(s);

    let mut counters = BTreeMap::new();
    counters.insert("disk_reads".to_string(), reads as f64);
    counters.insert(
        "disk_bytes_read".to_string(),
        (reads * index.record_bytes as u64) as f64,
    );
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(dc),
        counters,
    })
}

/// RAM only: PQ codes (N x m bytes) + codebooks (m x 256 x dim/m x 4) + entry point (4).
pub fn index_bytes(index: &DiskannIndex) -> u64 {
    (index.codes.len() + index.codebooks.len() * 4 + 4) as u64
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
    fn extra(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut e = serde_json::Map::new();
        e.insert("disk_bytes".into(), self.disk_bytes.into());
        e.insert("disk_path".into(), self.path.clone().into());
        e.insert("record_bytes".into(), self.record_bytes.into());
        e.insert("entry_point".into(), self.entry.into());
        e.insert("mean_out_degree".into(), self.mean_out_degree.into());
        e.insert("build_threads".into(), self.build_threads.into());
        e
    }
}
