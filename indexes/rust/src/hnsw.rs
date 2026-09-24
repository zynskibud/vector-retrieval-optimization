//! Hierarchical navigable small world graph (CONTRACT 6.6).
//!
//! Recall floor (CONTRACT 9): recall@10 >= 0.95 at ef=64 on the dev set.
//!
//! Malkov and Yashunin 2018: Algorithm 1 (insert), 2 (search-layer),
//! 4 (heuristic neighbor selection, extendCandidates = false,
//! keepPrunedConnections = false), 5 (search).
//!
//! Storage: one flat `i32` slot array per layer, with fixed slots per node
//! (2m on layer 0, m above) and a count per node. Layers >= 1 hold only the
//! nodes that exist on them; a per-layer map (N entries of `u32`) gives a node's
//! slot block. With `threads == 1`, rows are inserted strictly in row order.
//! With more threads, rows are inserted in parallel (rayon); each write to a
//! neighbor list holds a lock from a stripe of mutexes chosen by node ID. After
//! every build (any thread count), a repair pass (steps A and B) gives every node with zero layer-0 in-degree, and
//! every node that BFS from the entry does not reach, an incoming edge.
//! Parallel workers claim chunks of 64 consecutive rows.
//! The new node selects m neighbors on every layer; 2m is only the layer-0 cap.

use crate::distance::dot;
use crate::splitmix::SplitMix64;
use crate::{AnnIndex, BuildTimes, Matrix, ParamValue::*, Params, SearchResult};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering as AtOrd};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
        .with("m", Int(16))
        .with("ef_construct", Int(100))
}

pub fn search_defaults() -> Params {
    Params::new().with("ef", Int(64))
}

/// Highest level a node can get. Only a draw of u = 0 or a tiny u reaches it.
const MAX_LEVEL: usize = 32;
/// Number of mutexes in the lock stripe of the parallel build.
const LOCK_STRIPES: usize = 1 << 16;
/// Marks "node not on this layer" in a layer's node map.
const ABSENT: u32 = u32::MAX;

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

/// Per-thread (or per-query) working memory for search-layer.
struct Scratch {
    /// Generation-stamped visited set: node v is visited when `visited[v] == generation`.
    visited: Vec<u32>,
    generation: u32,
    candidates: BinaryHeap<Cand>,
    results: BinaryHeap<Reverse<Cand>>,
    neighbors: Vec<u32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            visited: vec![0; n],
            generation: 0,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            neighbors: Vec::new(),
        }
    }

    fn next_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.visited.fill(0);
            self.generation = 1;
        }
    }
}

/// Algorithm 2: search-layer. Returns up to `ef` nodes, best first.
/// `neighbors(node, out)` fills `out` with the node's neighbor IDs on this layer.
fn search_layer<F: Fn(u32, &mut Vec<u32>)>(
    query: &[f32],
    vectors: &Matrix,
    entry: &[Cand],
    ef: usize,
    neighbors: F,
    s: &mut Scratch,
    dist_count: &mut u64,
) -> Vec<Cand> {
    s.next_generation();
    let generation = s.generation;
    s.candidates.clear();
    s.results.clear();
    for &e in entry {
        if s.visited[e.id as usize] == generation {
            continue;
        }
        s.visited[e.id as usize] = generation;
        s.candidates.push(e);
        s.results.push(Reverse(e));
        if s.results.len() > ef {
            s.results.pop();
        }
    }
    let mut buf = std::mem::take(&mut s.neighbors);
    while let Some(c) = s.candidates.pop() {
        let worst = s.results.peek().expect("results not empty").0;
        if s.results.len() >= ef && c < worst {
            break;
        }
        neighbors(c.id, &mut buf);
        for &e in &buf {
            let v = &mut s.visited[e as usize];
            if *v == generation {
                continue;
            }
            *v = generation;
            let score = dot(query, vectors.row(e as usize));
            *dist_count += 1;
            let ce = Cand { score, id: e };
            let full = s.results.len() >= ef;
            if !full || ce > s.results.peek().expect("results not empty").0 {
                s.candidates.push(ce);
                s.results.push(Reverse(ce));
                if s.results.len() > ef {
                    s.results.pop();
                }
            }
        }
    }
    s.neighbors = buf;
    let mut out: Vec<Cand> = s.results.drain().map(|r| r.0).collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

/// Algorithm 4 with extendCandidates = false and keepPrunedConnections = false.
/// `cands` holds scores against the base node, best first. A candidate e is kept
/// when no already kept node r is closer to e than the base node is:
/// reject e if dot(e, r) > dot(e, base).
fn select_heuristic(vectors: &Matrix, cands: &[Cand], limit: usize) -> Vec<u32> {
    let mut kept: Vec<u32> = Vec::with_capacity(limit);
    for e in cands {
        if kept.len() >= limit {
            break;
        }
        let ev = vectors.row(e.id as usize);
        if kept
            .iter()
            .all(|&r| dot(ev, vectors.row(r as usize)) <= e.score)
        {
            kept.push(e.id);
        }
    }
    kept
}

/// One layer during the build. Atomics let threads read lists while other
/// threads write them under a lock, with no unsafe code.
struct BuildLayer {
    cap: usize,
    slots: Vec<AtomicI32>,
    counts: Vec<AtomicU32>,
    /// Node ID -> block index. Empty on layer 0, where the block index is the node ID.
    map: Vec<u32>,
}

impl BuildLayer {
    #[inline]
    fn block(&self, node: u32) -> usize {
        if self.map.is_empty() {
            node as usize
        } else {
            self.map[node as usize] as usize
        }
    }

    fn read(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        let b = self.block(node);
        let c = (self.counts[b].load(AtOrd::Acquire) as usize).min(self.cap);
        for slot in &self.slots[b * self.cap..b * self.cap + c] {
            let v = slot.load(AtOrd::Relaxed);
            if v >= 0 {
                out.push(v as u32);
            }
        }
    }

    fn write(&self, node: u32, ids: &[u32]) {
        let b = self.block(node);
        for (slot, &id) in self.slots[b * self.cap..].iter().zip(ids) {
            slot.store(id as i32, AtOrd::Relaxed);
        }
        self.counts[b].store(ids.len() as u32, AtOrd::Release);
    }
}

/// One layer of the finished graph.
struct Layer {
    cap: usize,
    slots: Vec<i32>,
    counts: Vec<u32>,
    map: Vec<u32>,
}

impl Layer {
    #[inline]
    fn block(&self, node: u32) -> Option<usize> {
        if self.map.is_empty() {
            Some(node as usize)
        } else {
            match self.map[node as usize] {
                ABSENT => None,
                b => Some(b as usize),
            }
        }
    }

    #[inline]
    fn list(&self, node: u32) -> &[i32] {
        match self.block(node) {
            Some(b) => &self.slots[b * self.cap..b * self.cap + self.counts[b] as usize],
            None => &[],
        }
    }

    fn read(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.list(node).iter().map(|&v| v as u32));
    }
}

pub struct HnswIndex {
    vectors: Matrix,
    levels: Vec<u8>,
    layers: Vec<Layer>,
    entry: u32,
    m: usize,
    ef_construct: usize,
    unreachable_before_repair: usize,
    repair_added: u64,
    repair_added_unreachable: u64,
    build_threads: usize,
    times: BuildTimes,
    pool: Mutex<Vec<Scratch>>,
}

/// Levels of CONTRACT 6.6: one SplitMix64 seeded `seed`, one `next_f64` per row, in row order.
pub fn draw_levels(n: usize, m: usize, seed: u64) -> Vec<u8> {
    let ml = 1.0 / (m as f64).ln();
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let u = rng.next_f64();
            let l = if u > 0.0 {
                (-u.ln() * ml).floor()
            } else {
                MAX_LEVEL as f64
            };
            (l as usize).min(MAX_LEVEL) as u8
        })
        .collect()
}

/// The graph during the build.
struct Builder<'a> {
    vectors: &'a Matrix,
    levels: &'a [u8],
    layers: Vec<BuildLayer>,
    locks: Vec<Mutex<()>>,
    /// (entry point, top layer).
    entry: Mutex<(u32, usize)>,
    m: usize,
    ef_construct: usize,
}

impl Builder<'_> {
    fn lock(&self, node: u32) -> MutexGuard<'_, ()> {
        self.locks[node as usize % self.locks.len()]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Adds `new` to the list of `node` on `layer`. If the list exceeds its limit,
    /// shrinks it with the heuristic over all its current neighbors.
    /// IDs in `keep` are protected: never pruned (repair pass, CONTRACT 6.6).
    fn add_links(&self, node: u32, layer: usize, new: &[u32], keep: &[u32], buf: &mut Vec<u32>) {
        let l = &self.layers[layer];
        let _g = self.lock(node);
        l.read(node, buf);
        for &id in new {
            if id != node && !buf.contains(&id) {
                buf.push(id);
            }
        }
        if buf.len() <= l.cap {
            l.write(node, buf);
            return;
        }
        let base = self.vectors.row(node as usize);
        let mut cands: Vec<Cand> = buf
            .iter()
            .filter(|id| !keep.contains(id))
            .map(|&id| Cand {
                score: dot(base, self.vectors.row(id as usize)),
                id,
            })
            .collect();
        cands.sort_unstable_by(|a, b| b.cmp(a));
        let mut kept: Vec<u32> = buf.iter().copied().filter(|id| keep.contains(id)).collect();
        kept.truncate(l.cap);
        let room = l.cap - kept.len();
        kept.extend(select_heuristic(self.vectors, &cands, room));
        l.write(node, &kept);
    }

    /// Algorithm 1 for row `i`.
    fn insert(&self, i: u32, s: &mut Scratch) {
        let level = self.levels[i as usize] as usize;
        let guard = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        let (ep, top) = *guard;
        // A node that raises the top layer holds the entry lock for its whole insert.
        let hold = if level > top {
            Some(guard)
        } else {
            drop(guard);
            None
        };
        let q = self.vectors.row(i as usize);
        let mut dc = 0u64;
        let mut cur = vec![Cand {
            score: dot(q, self.vectors.row(ep as usize)),
            id: ep,
        }];
        for layer in (level + 1..=top).rev() {
            let bl = &self.layers[layer];
            let w = search_layer(q, self.vectors, &cur, 1, |n, o| bl.read(n, o), s, &mut dc);
            cur.truncate(0);
            cur.push(w[0]);
        }
        let mut buf = Vec::new();
        for layer in (0..=level.min(top)).rev() {
            let bl = &self.layers[layer];
            let w = search_layer(
                q,
                self.vectors,
                &cur,
                self.ef_construct,
                |n, o| bl.read(n, o),
                s,
                &mut dc,
            );
            // The new node selects m on every layer; 2m is only the layer-0 cap.
            let chosen = select_heuristic(self.vectors, &w, self.m);
            self.add_links(i, layer, &chosen, &[], &mut buf);
            for &nb in &chosen {
                self.add_links(nb, layer, &[i], &[], &mut buf);
            }
            cur = w;
        }
        if let Some(mut g) = hold {
            *g = (i, level);
        }
    }

    /// Number of layer-0 nodes that BFS from `entry` does not reach.
    fn unreachable(&self, entry: u32) -> usize {
        self.reach(entry).iter().filter(|&&r| !r).count()
    }

    /// Layer-0 BFS from `entry`: `true` for every reached node.
    fn reach(&self, entry: u32) -> Vec<bool> {
        let l0 = &self.layers[0];
        let n = self.vectors.rows;
        let mut seen = vec![false; n];
        seen[entry as usize] = true;
        let mut stack = vec![entry];
        let mut buf = Vec::new();
        while let Some(v) = stack.pop() {
            l0.read(v, &mut buf);
            for &u in &buf {
                if !seen[u as usize] {
                    seen[u as usize] = true;
                    stack.push(u);
                }
            }
        }
        seen
    }

    /// search-layer on layer 0 from `entry` with `ef_construct` for the vector
    /// of `v`. Returns results nearest first, without `v`.
    fn search_from_entry(&self, v: u32, entry: u32, s: &mut Scratch) -> Vec<Cand> {
        let l0 = &self.layers[0];
        let q = self.vectors.row(v as usize);
        let start = [Cand {
            score: dot(q, self.vectors.row(entry as usize)),
            id: entry,
        }];
        let mut dc = 0;
        let mut w = search_layer(
            q,
            self.vectors,
            &start,
            self.ef_construct,
            |x, o| l0.read(x, o),
            s,
            &mut dc,
        );
        w.retain(|c| c.id != v);
        w
    }

    /// Adds the protected edge u -> v on layer 0.
    fn add_protected(
        &self,
        u: u32,
        v: u32,
        protected: &mut HashMap<u32, Vec<u32>>,
        buf: &mut Vec<u32>,
    ) {
        let keep = protected.entry(u).or_default();
        if !keep.contains(&v) {
            keep.push(v);
        }
        self.add_links(u, 0, &[v], keep, buf);
    }

    /// Repair pass of CONTRACT 6.6. Step A: every node with zero layer-0
    /// in-degree (row order, entry point skipped) gets an edge from the nearest
    /// node in its own list, or from the nearest result of a search from the entry
    /// if it has no out-edges. Step B: every node that a directed BFS from the
    /// entry does not reach (row order) gets an edge from the first search result
    /// (nearest first) with a free slot, else from the nearest result. Edges added
    /// by the repair are never pruned. Repeat while step B found a node, at most 3
    /// passes. Returns (edges added by step A, edges added by step B).
    fn repair(&self, entry: u32) -> (u64, u64) {
        let l0 = &self.layers[0];
        let n = self.vectors.rows;
        let mut s = Scratch::new(n);
        let mut buf = Vec::new();
        let mut protected: HashMap<u32, Vec<u32>> = HashMap::new();
        let (mut added_a, mut added_b) = (0u64, 0u64);
        for _ in 0..3 {
            // Step A.
            let mut indeg = vec![0u32; n];
            for v in 0..n as u32 {
                l0.read(v, &mut buf);
                for &u in &buf {
                    indeg[u as usize] += 1;
                }
            }
            for v in (0..n as u32).filter(|&v| indeg[v as usize] == 0 && v != entry) {
                let q = self.vectors.row(v as usize);
                l0.read(v, &mut buf);
                let best = buf
                    .iter()
                    .map(|&u| Cand {
                        score: dot(q, self.vectors.row(u as usize)),
                        id: u,
                    })
                    .max()
                    .or_else(|| self.search_from_entry(v, entry, &mut s).first().copied());
                if let Some(u) = best {
                    self.add_protected(u.id, v, &mut protected, &mut buf);
                    added_a += 1;
                }
            }
            // Step B.
            let seen = self.reach(entry);
            let unreached: Vec<u32> = (0..n as u32).filter(|&v| !seen[v as usize]).collect();
            if unreached.is_empty() {
                break;
            }
            for v in unreached {
                let w = self.search_from_entry(v, entry, &mut s);
                let u = w
                    .iter()
                    .find(|c| (l0.counts[l0.block(c.id)].load(AtOrd::Acquire) as usize) < l0.cap)
                    .or(w.first());
                if let Some(u) = u {
                    self.add_protected(u.id, v, &mut protected, &mut buf);
                    added_b += 1;
                }
            }
        }
        (added_a, added_b)
    }
}

pub fn build(
    vectors: Matrix,
    params: &Params,
    threads: usize,
    seed: u64,
) -> Result<HnswIndex, String> {
    let m = params.get_usize("m")?;
    let ef_construct = params.get_usize("ef_construct")?;
    if m < 2 {
        return Err(format!("m must be >= 2, got {m}"));
    }
    if ef_construct < 1 {
        return Err("ef_construct must be >= 1".into());
    }
    let n = vectors.rows;
    if n == 0 {
        return Err("empty corpus".into());
    }
    if n > i32::MAX as usize {
        return Err("corpus too large for int32 IDs".into());
    }
    let start = Instant::now();
    let levels = draw_levels(n, m, seed);
    let max_level = *levels.iter().max().expect("n > 0") as usize;

    let mut layers = Vec::with_capacity(max_level + 1);
    for layer in 0..=max_level {
        let cap = if layer == 0 { 2 * m } else { m };
        let (map, count) = if layer == 0 {
            (Vec::new(), n)
        } else {
            let mut map = vec![ABSENT; n];
            let mut c = 0u32;
            for (i, &l) in levels.iter().enumerate() {
                if l as usize >= layer {
                    map[i] = c;
                    c += 1;
                }
            }
            (map, c as usize)
        };
        layers.push(BuildLayer {
            cap,
            slots: (0..count * cap).map(|_| AtomicI32::new(-1)).collect(),
            counts: (0..count).map(|_| AtomicU32::new(0)).collect(),
            map,
        });
    }

    let builder = Builder {
        vectors: &vectors,
        levels: &levels,
        layers,
        locks: (0..LOCK_STRIPES.min(n)).map(|_| Mutex::new(())).collect(),
        entry: Mutex::new((0, levels[0] as usize)),
        m,
        ef_construct,
    };
    if threads <= 1 {
        let mut s = Scratch::new(n);
        for i in 1..n as u32 {
            builder.insert(i, &mut s);
        }
    } else {
        // Workers claim chunks of 64 consecutive rows (CONTRACT 6.6).
        const CHUNK: usize = 64;
        (0..n.div_ceil(CHUNK)).into_par_iter().for_each_init(
            || Scratch::new(n),
            |s, c| {
                for i in (c * CHUNK).max(1)..((c + 1) * CHUNK).min(n) {
                    builder.insert(i as u32, s);
                }
            },
        );
    }
    let (entry, _) = *builder.entry.lock().unwrap_or_else(|e| e.into_inner());
    let unreachable_before_repair = builder.unreachable(entry);
    let (repair_added, repair_added_unreachable) = builder.repair(entry);
    let (bm, bef) = (builder.m, builder.ef_construct);
    let layers: Vec<Layer> = builder
        .layers
        .into_iter()
        .map(|l| Layer {
            cap: l.cap,
            slots: l.slots.into_iter().map(AtomicI32::into_inner).collect(),
            counts: l.counts.into_iter().map(AtomicU32::into_inner).collect(),
            map: l.map,
        })
        .collect();
    let add_s = start.elapsed().as_secs_f64();
    Ok(HnswIndex {
        vectors,
        levels,
        layers,
        entry,
        m: bm,
        ef_construct: bef,
        unreachable_before_repair,
        repair_added,
        repair_added_unreachable,
        build_threads: threads,
        times: BuildTimes {
            train_s: 0.0,
            add_s,
        },
        pool: Mutex::new(Vec::new()),
    })
}

/// Algorithm 5: greedy descent to layer 1 with ef = 1, then search-layer on
/// layer 0 with max(ef, k). Returns the top k, padded with -1 and `-inf`.
pub fn search(
    index: &HnswIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let ef = params.get_usize("ef")?.max(k).max(1);
    let n = index.vectors.rows;
    let mut s = index
        .pool
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pop()
        .unwrap_or_else(|| Scratch::new(n));
    let mut dc = 1u64;
    let ep = index.entry;
    let mut cur = vec![Cand {
        score: dot(query, index.vectors.row(ep as usize)),
        id: ep,
    }];
    for layer in (1..index.layers.len()).rev() {
        let l = &index.layers[layer];
        let w = search_layer(
            query,
            &index.vectors,
            &cur,
            1,
            |nd, o| l.read(nd, o),
            &mut s,
            &mut dc,
        );
        cur.truncate(0);
        cur.push(w[0]);
    }
    let l0 = &index.layers[0];
    let w = search_layer(
        query,
        &index.vectors,
        &cur,
        ef,
        |nd, o| l0.read(nd, o),
        &mut s,
        &mut dc,
    );
    index.pool.lock().unwrap_or_else(|e| e.into_inner()).push(s);
    let mut ids: Vec<i64> = w.iter().take(k).map(|c| c.id as i64).collect();
    let mut scores: Vec<f32> = w.iter().take(k).map(|c| c.score).collect();
    ids.resize(k, -1);
    scores.resize(k, f32::NEG_INFINITY);
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(dc),
        counters: Default::default(),
    })
}

impl HnswIndex {
    /// Level of every node, in row order.
    pub fn levels(&self) -> &[u8] {
        &self.levels
    }
    /// The entry point of search.
    pub fn entry_point(&self) -> u32 {
        self.entry
    }
    /// Number of layers (top layer + 1).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    /// Slot limit per node on `layer`: 2m on layer 0, m above.
    pub fn layer_cap(&self, layer: usize) -> usize {
        self.layers[layer].cap
    }
    /// Stored count of every node block on `layer`, before clamping.
    pub fn layer_counts(&self, layer: usize) -> &[u32] {
        &self.layers[layer].counts
    }
    /// Neighbor IDs of `node` on `layer` (empty if the node is not on that layer).
    pub fn neighbors(&self, layer: usize, node: u32) -> &[i32] {
        self.layers[layer].list(node)
    }
    /// Layer-0 nodes that BFS from the entry point did not reach before the repair pass.
    pub fn unreachable_before_repair(&self) -> usize {
        self.unreachable_before_repair
    }
    /// Edges added by step A of the repair pass (zero in-degree).
    pub fn repair_added(&self) -> u64 {
        self.repair_added
    }
    /// Edges added by step B of the repair pass (unreachable by BFS).
    pub fn repair_added_unreachable(&self) -> u64 {
        self.repair_added_unreachable
    }
    /// Nodes on each layer, layer 0 first.
    pub fn nodes_per_layer(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.counts.len()).collect()
    }
    fn edges(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.counts.iter().map(|&c| c as u64).sum::<u64>())
            .sum()
    }
    fn bookkeeping_bytes(&self) -> u64 {
        let counts: u64 = self.layers.iter().map(|l| l.counts.len() as u64 * 4).sum();
        let maps: u64 = self.layers.iter().map(|l| l.map.len() as u64 * 4).sum();
        self.levels.len() as u64 + counts + maps
    }
}

/// Edges x 4 bytes, plus levels (1 byte per node), counts (4 bytes per node per
/// layer), and the node maps of layers >= 1 (4 bytes per corpus row per layer).
pub fn index_bytes(index: &HnswIndex) -> u64 {
    index.edges() * 4 + index.bookkeeping_bytes()
}

impl AnnIndex for HnswIndex {
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
        e.insert("top_layer".into(), (self.layers.len() - 1).into());
        e.insert("entry_point".into(), self.entry.into());
        e.insert("nodes_per_layer".into(), self.nodes_per_layer().into());
        e.insert("edges".into(), self.edges().into());
        e.insert("edge_bytes".into(), (self.edges() * 4).into());
        e.insert("bookkeeping_bytes".into(), self.bookkeeping_bytes().into());
        e.insert("m".into(), self.m.into());
        e.insert("ef_construct".into(), self.ef_construct.into());
        e.insert(
            "unreachable_before_repair".into(),
            self.unreachable_before_repair.into(),
        );
        e.insert("repair_added".into(), self.repair_added.into());
        e.insert(
            "repair_added_unreachable".into(),
            self.repair_added_unreachable.into(),
        );
        e.insert("build_threads".into(), self.build_threads.into());
        e
    }
}
