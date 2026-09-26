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
//!
//! Concurrency (CONTRACT 12): slots and counts stay atomics after the build. A query
//! takes no lock: relaxed loads of the lists, the entry point and top layer from one
//! atomic word, and a thread-local scratch. [`HnswIndex::insert`] adds rows in row order
//! under the per-node lock stripe (slots stored first, then the count, release); the
//! entry point changes under the global lock. The index owns all N rows from the build
//! on and draws all N levels then; [`build_partial`] links only the first rows, so an
//! insert never moves memory that a query reads.
//!
//! Filter (CONTRACT 11.3): the upper-layer descent ignores the filter. On layer 0,
//! [`search_layer0`] puts a node in the result heap only if it passes the mask; every
//! visited node still goes to the candidate heap (under the usual admission rule) and
//! is expanded. The stop rule and ef are unchanged, so a low selectivity ends the walk
//! with fewer than k results; no brute-force fallback. `extra.visited` = nodes expanded.
//!
//! Changes (CONTRACT 13.3):
//! - Delete = tombstone bit set. A tombstoned node is still expanded and its edges are
//!   followed, but it never enters the result list (the same rule as a failing filter).
//! - Update: overwrite the vector, drop the node's out-edges on every layer, remove it
//!   from the lists of its old neighbors (a scan over those lists), and run Algorithm 1
//!   again for the node with its existing level. During that insert the node itself is
//!   left out of every neighbor list read, so it cannot select itself; if the node is the
//!   entry point, the descent starts from its old neighbor on its highest non-empty layer.
//!   After the batch, the repair pass of CONTRACT 6.6 runs once.
//! - Compact, rebuild mode (the default): build a new graph from the live rows only, in
//!   row order, with the same m, ef_construct, seed and threads (levels are drawn again
//!   for the live rows in row order). A position -> row ID map (4 bytes per live row)
//!   turns positions back into row IDs; positions keep row order, so ties still go to
//!   the lower row ID. The bit set is dropped.
//! - Compact, repair mode (`--compact-mode repair`), in place: (1) on every layer, each
//!   live node whose list holds a tombstoned node gets a new list, chosen with the
//!   heuristic (up to the layer cap) from its live neighbors plus the live neighbors of
//!   its tombstoned neighbors; (2) the lists of the tombstoned nodes are cleared, so no
//!   edge leads to or from them; (3) if the entry point is tombstoned, the live node with
//!   the highest level (lowest row on a tie) becomes the entry point; (4) the repair pass
//!   of CONTRACT 6.6 runs over the live nodes. The tombstoned nodes keep their slots and
//!   the bit set stays, so memory shrinks only by the dropped edges.

use crate::distance::dot;
use crate::splitmix::SplitMix64;
use crate::{
    check_update, AnnIndex, BuildTimes, FilterMasks, Matrix, ParamValue::*, Params, SearchResult,
    Tombstones,
};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::cell::RefCell;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering as AtOrd};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

pub fn build_defaults(_n: usize) -> Params {
    Params::new()
        .with("m", Int(16))
        .with("ef_construct", Int(100))
}

pub fn search_defaults() -> Params {
    Params::new()
        .with("ef", Int(64))
        .with("filter", Str("none".into()))
}

/// Highest level a node can get. Only a draw of u = 0 or a tiny u reaches it.
const MAX_LEVEL: usize = 32;
/// Number of mutexes in the lock stripe of the parallel build.
const LOCK_STRIPES: usize = 1 << 16;
/// Marks "node not on this layer" in a layer's node map.
const ABSENT: u32 = u32::MAX;

/// Search-layer on layer 0 for queries (Algorithm 2 with a filter, CONTRACT 11.3).
/// A node enters `results` only if `ok(node)` is true (it passes the filter and is not
/// tombstoned, CONTRACT 13.3). Every
/// visited node with a score better than the worst result (or any score while the
/// result heap holds fewer than `ef`) enters `candidates` and is later expanded.
/// Stop when the best candidate is worse than the worst result and the result heap
/// is full. With `ok` always true this is the same walk as [`search_layer`].
/// Returns up to `ef` passing nodes, best first, and the number of expanded nodes.
fn search_layer0<F: Fn(u32, &mut Vec<u32>), P: Fn(u32) -> bool>(
    query: &[f32],
    vectors: &Matrix,
    entry: Cand,
    ef: usize,
    ok: P,
    neighbors: F,
    s: &mut Scratch,
    dist_count: &mut u64,
) -> (Vec<Cand>, u64) {
    s.next_generation();
    let generation = s.generation;
    s.candidates.clear();
    s.results.clear();
    s.visited[entry.id as usize] = generation;
    s.candidates.push(entry);
    if ok(entry.id) {
        s.results.push(Reverse(entry));
    }
    let mut expanded = 0u64;
    let mut buf = std::mem::take(&mut s.neighbors);
    while let Some(c) = s.candidates.pop() {
        if s.results.len() >= ef && c < s.results.peek().expect("results full").0 {
            break;
        }
        expanded += 1;
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
            if !full || ce > s.results.peek().expect("results full").0 {
                s.candidates.push(ce);
                if ok(e) {
                    s.results.push(Reverse(ce));
                    if s.results.len() > ef {
                        s.results.pop();
                    }
                }
            }
        }
    }
    s.neighbors = buf;
    let mut out: Vec<Cand> = s.results.drain().map(|r| r.0).collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    (out, expanded)
}

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

/// One layer of the graph, during and after the build. Slots and counts are atomics,
/// so searches read lists while an insert writes them under a lock, with no unsafe code
/// (CONTRACT 12.2). The layers are sized for all N rows at build time.
struct Layer {
    cap: usize,
    slots: Vec<AtomicI32>,
    counts: Vec<AtomicU32>,
    /// Node ID -> block index. Empty on layer 0, where the block index is the node ID.
    map: Vec<u32>,
}

impl Layer {
    #[inline]
    fn block(&self, node: u32) -> usize {
        if self.map.is_empty() {
            node as usize
        } else {
            self.map[node as usize] as usize
        }
    }

    #[inline]
    fn block_opt(&self, node: u32) -> Option<usize> {
        match self.map.get(node as usize) {
            None if self.map.is_empty() => Some(node as usize),
            Some(&b) if b != ABSENT => Some(b as usize),
            _ => None,
        }
    }

    /// Reads a list for the build and the repair: acquire load of the count.
    fn read(&self, node: u32, out: &mut Vec<u32>) {
        self.read_with(node, out, AtOrd::Acquire);
    }

    /// Reads a list for a query: relaxed loads only (CONTRACT 12.2). A concurrent
    /// insert can make the list look shorter or longer, never torn.
    fn read_relaxed(&self, node: u32, out: &mut Vec<u32>) {
        self.read_with(node, out, AtOrd::Relaxed);
    }

    #[inline]
    fn read_with(&self, node: u32, out: &mut Vec<u32>, order: AtOrd) {
        out.clear();
        let b = self.block(node);
        let c = (self.counts[b].load(order) as usize).min(self.cap);
        for slot in &self.slots[b * self.cap..b * self.cap + c] {
            let v = slot.load(AtOrd::Relaxed);
            if v >= 0 {
                out.push(v as u32);
            }
        }
    }

    /// Writes the slots first, then the count with a release store.
    fn write(&self, node: u32, ids: &[u32]) {
        let b = self.block(node);
        for (slot, &id) in self.slots[b * self.cap..].iter().zip(ids) {
            slot.store(id as i32, AtOrd::Release);
        }
        self.counts[b].store(ids.len() as u32, AtOrd::Release);
    }

    fn count(&self, block: usize) -> u32 {
        self.counts[block].load(AtOrd::Acquire)
    }
}

thread_local! {
    /// Per-thread search scratch, so concurrent queries share no lock (CONTRACT 12).
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new(0));
}

/// Runs `f` with this thread's scratch, grown to at least `n` rows.
fn with_scratch<R>(n: usize, f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        if s.visited.len() < n {
            s.visited.resize(n, 0);
        }
        f(&mut s)
    })
}

pub struct HnswIndex {
    graph: Graph,
    unreachable_before_repair: usize,
    repair_added: u64,
    repair_added_unreachable: u64,
    build_threads: usize,
    times: BuildTimes,
    filters: FilterMasks,
    /// Serializes calls to [`HnswIndex::insert`], so rows go in strictly in row order.
    insert_lock: Mutex<()>,
    /// Seed of the build, reused by the compaction rebuild.
    seed: u64,
    /// Corpus rows (row IDs are 0..rows), also after a compaction rebuild.
    rows: usize,
    /// Deleted nodes (CONTRACT 13.3).
    tombstones: Option<Tombstones>,
    /// After a compaction rebuild: row ID of each node.
    row_ids: Option<Vec<u32>>,
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

/// The graph. It owns all N corpus rows from the build on; rows `active..N` are
/// stored but not yet linked (CONTRACT 12.2: the build takes the first 90%).
struct Graph {
    vectors: Matrix,
    levels: Vec<u8>,
    layers: Vec<Layer>,
    locks: Vec<Mutex<()>>,
    /// (entry point, top layer). Changed only under this global lock.
    entry: Mutex<(u32, usize)>,
    /// Copy of `entry` as `(entry << 32) | top`, for lock-free reads by queries.
    entry_top: AtomicU64,
    /// Rows `0..active` are in the graph.
    active: AtomicUsize,
    m: usize,
    ef_construct: usize,
}

fn pack(entry: u32, top: usize) -> u64 {
    ((entry as u64) << 32) | top as u64
}

impl Graph {
    /// (entry point, top layer), read without a lock.
    fn entry_top(&self) -> (u32, usize) {
        let v = self.entry_top.load(AtOrd::Acquire);
        ((v >> 32) as u32, (v & 0xffff_ffff) as usize)
    }

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
        kept.extend(select_heuristic(&self.vectors, &cands, room));
        l.write(node, &kept);
    }

    /// Algorithm 1 for row `i`.
    fn insert(&self, i: u32, s: &mut Scratch) {
        self.insert_with(i, s, false, None);
    }

    /// Algorithm 1 for row `i`. With `skip_self`, node `i` is removed from every
    /// neighbor list that the insert reads (update, CONTRACT 13.3). `start` replaces
    /// the entry point and top layer as the start of the descent.
    fn insert_with(&self, i: u32, s: &mut Scratch, skip_self: bool, start: Option<(u32, usize)>) {
        let level = self.levels[i as usize] as usize;
        let guard = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        let (ep, top) = start.unwrap_or(*guard);
        let read = |bl: &Layer, n: u32, o: &mut Vec<u32>| {
            bl.read(n, o);
            if skip_self {
                o.retain(|&x| x != i);
            }
        };
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
            let w = search_layer(q, &self.vectors, &cur, 1, |n, o| read(bl, n, o), s, &mut dc);
            cur.truncate(0);
            cur.push(w[0]);
        }
        let mut buf = Vec::new();
        for layer in (0..=level.min(top)).rev() {
            let bl = &self.layers[layer];
            let w = search_layer(
                q,
                &self.vectors,
                &cur,
                self.ef_construct,
                |n, o| read(bl, n, o),
                s,
                &mut dc,
            );
            // The new node selects m on every layer; 2m is only the layer-0 cap.
            let chosen = select_heuristic(&self.vectors, &w, self.m);
            self.add_links(i, layer, &chosen, &[], &mut buf);
            for &nb in &chosen {
                self.add_links(nb, layer, &[i], &[], &mut buf);
            }
            cur = w;
        }
        if let Some(mut g) = hold {
            *g = (i, level);
            self.entry_top.store(pack(i, level), AtOrd::Release);
        }
    }

    /// Number of layer-0 nodes that BFS from `entry` does not reach.
    fn unreachable(&self, entry: u32) -> usize {
        self.reach(entry).iter().filter(|&&r| !r).count()
    }

    /// Layer-0 BFS from `entry` over the active rows: `true` for every reached node.
    fn reach(&self, entry: u32) -> Vec<bool> {
        let l0 = &self.layers[0];
        let n = self.active.load(AtOrd::Acquire);
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
            &self.vectors,
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
    /// Nodes with `dead[v]` true (compaction repair mode) are skipped.
    fn repair(&self, entry: u32, dead: Option<&[bool]>) -> (u64, u64) {
        let is_dead = |v: u32| dead.is_some_and(|d| d[v as usize]);
        let l0 = &self.layers[0];
        let n = self.active.load(AtOrd::Acquire);
        let mut s = Scratch::new(self.vectors.rows);
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
            for v in (0..n as u32).filter(|&v| indeg[v as usize] == 0 && v != entry && !is_dead(v)) {
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
            let unreached: Vec<u32> = (0..n as u32)
                .filter(|&v| !seen[v as usize] && !is_dead(v))
                .collect();
            if unreached.is_empty() {
                break;
            }
            for v in unreached {
                let w = self.search_from_entry(v, entry, &mut s);
                let u = w
                    .iter()
                    .find(|c| (l0.count(l0.block(c.id)) as usize) < l0.cap)
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
    let n = vectors.rows;
    build_partial(vectors, params, threads, seed, n)
}

/// Builds the graph on the first `build_rows` rows of `vectors`. The other rows stay
/// stored, unlinked, until [`HnswIndex::insert`] adds them (CONTRACT 12.2). Levels are
/// drawn for all N rows here, in row order, so an inserted row gets the same level as
/// in a full build. With `build_rows == N` this is the build of CONTRACT 6.6.
pub fn build_partial(
    vectors: Matrix,
    params: &Params,
    threads: usize,
    seed: u64,
    build_rows: usize,
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
    if n == 0 || build_rows == 0 {
        return Err("empty corpus".into());
    }
    if build_rows > n {
        return Err(format!("build_rows {build_rows} > corpus rows {n}"));
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
        layers.push(Layer {
            cap,
            slots: (0..count * cap).map(|_| AtomicI32::new(-1)).collect(),
            counts: (0..count).map(|_| AtomicU32::new(0)).collect(),
            map,
        });
    }

    let graph = Graph {
        vectors,
        entry: Mutex::new((0, levels[0] as usize)),
        entry_top: AtomicU64::new(pack(0, levels[0] as usize)),
        active: AtomicUsize::new(build_rows),
        levels,
        layers,
        locks: (0..LOCK_STRIPES.min(n)).map(|_| Mutex::new(())).collect(),
        m,
        ef_construct,
    };
    if threads <= 1 {
        let mut s = Scratch::new(n);
        for i in 1..build_rows as u32 {
            graph.insert(i, &mut s);
        }
    } else {
        // Workers claim chunks of 64 consecutive rows (CONTRACT 6.6).
        const CHUNK: usize = 64;
        (0..build_rows.div_ceil(CHUNK)).into_par_iter().for_each_init(
            || Scratch::new(n),
            |s, c| {
                for i in (c * CHUNK).max(1)..((c + 1) * CHUNK).min(build_rows) {
                    graph.insert(i as u32, s);
                }
            },
        );
    }
    let (entry, _) = graph.entry_top();
    let unreachable_before_repair = graph.unreachable(entry);
    let (repair_added, repair_added_unreachable) = graph.repair(entry, None);
    let add_s = start.elapsed().as_secs_f64();
    Ok(HnswIndex {
        graph,
        unreachable_before_repair,
        repair_added,
        repair_added_unreachable,
        build_threads: threads,
        times: BuildTimes {
            train_s: 0.0,
            add_s,
        },
        filters: FilterMasks::default(),
        insert_lock: Mutex::new(()),
        seed,
        rows: n,
        tombstones: None,
        row_ids: None,
    })
}

/// Algorithm 5: greedy descent to layer 1 with ef = 1, then search-layer on
/// layer 0 with max(ef, k). Returns the top k, padded with -1 and `-inf`.
/// Takes no lock: lists are read with relaxed atomic loads, the entry point from
/// one atomic word, and the scratch memory is per thread.
pub fn search(
    index: &HnswIndex,
    query: &[f32],
    k: usize,
    params: &Params,
) -> Result<SearchResult, String> {
    let ef = params.get_usize("ef")?.max(k).max(1);
    let g = &index.graph;
    let mask = index.filters.for_params(params)?;
    let pass = mask.as_ref().map(|m| m.pass.as_slice());
    let filter_rows = mask.as_ref().map(|m| m.rows);
    let tomb = index.tombstones.as_ref();
    let row_ids = index.row_ids.as_deref();
    let row_id = |node: u32| row_ids.map_or(node, |r| r[node as usize]);
    let ok = |node: u32| {
        !tomb.is_some_and(|t| t.is_deleted(node as usize))
            && pass.is_none_or(|p| p[row_id(node) as usize])
    };
    let mut dc = 1u64;
    let (ep, top) = g.entry_top();
    let (w, visited) = with_scratch(g.vectors.rows, |s| {
        let mut cur = vec![Cand {
            score: dot(query, g.vectors.row(ep as usize)),
            id: ep,
        }];
        for layer in (1..=top).rev() {
            let l = &g.layers[layer];
            let w = search_layer(
                query,
                &g.vectors,
                &cur,
                1,
                |nd, o| l.read_relaxed(nd, o),
                s,
                &mut dc,
            );
            cur.truncate(0);
            cur.push(w[0]);
        }
        let l0 = &g.layers[0];
        search_layer0(
            query,
            &g.vectors,
            cur[0],
            ef,
            ok,
            |nd, o| l0.read_relaxed(nd, o),
            s,
            &mut dc,
        )
    });
    let mut ids: Vec<i64> = w.iter().take(k).map(|c| row_id(c.id) as i64).collect();
    let mut scores: Vec<f32> = w.iter().take(k).map(|c| c.score).collect();
    ids.resize(k, -1);
    scores.resize(k, f32::NEG_INFINITY);
    Ok(SearchResult {
        ids,
        scores,
        distance_computations: Some(dc),
        counters: filter_rows
            .map(|r| ("filter_rows".to_string(), r as f64))
            .into_iter()
            .chain([("visited".to_string(), visited as f64)])
            .collect(),
    })
}

impl HnswIndex {
    /// Sets the directory that holds `filter_<name>.npy` (CONTRACT 11).
    pub fn set_filter_dir(&mut self, dir: &str) {
        self.filters = FilterMasks::new(dir, self.rows);
    }
    /// Adds rows to the graph while queries run (CONTRACT 12.2). `ids` must be the next
    /// rows in row order (starting at [`HnswIndex::active_rows`]), and `vectors` their
    /// rows, `ids.len() x dim` values, equal to the rows stored at build time.
    /// Each row is inserted with Algorithm 1 and its level from the build-time draw.
    pub fn insert(&self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        let _order = self.insert_lock.lock().unwrap_or_else(|e| e.into_inner());
        let g = &self.graph;
        let dim = g.vectors.cols;
        if vectors.len() != ids.len() * dim {
            return Err(format!(
                "insert: {} values for {} rows of dim {dim}",
                vectors.len(),
                ids.len()
            ));
        }
        let first = g.active.load(AtOrd::Acquire);
        for (j, &id) in ids.iter().enumerate() {
            if id != (first + j) as i64 || id as usize >= g.vectors.rows {
                return Err(format!(
                    "insert: row {id} out of order (next row is {}, corpus has {})",
                    first + j,
                    g.vectors.rows
                ));
            }
            if g.vectors.row(id as usize) != &vectors[j * dim..(j + 1) * dim] {
                return Err(format!("insert: vector of row {id} differs from the corpus"));
            }
        }
        with_scratch(g.vectors.rows, |s| {
            for &id in ids {
                g.insert(id as u32, s);
                g.active.store(id as usize + 1, AtOrd::Release);
            }
        });
        Ok(())
    }
    /// Rows in the graph now: the build rows plus the inserted rows.
    pub fn active_rows(&self) -> usize {
        self.graph.active.load(AtOrd::Acquire)
    }
    /// Runs the repair pass of CONTRACT 6.6 once over the active rows (after inserts).
    /// Returns (edges added by step A, edges added by step B). Call it when no insert runs.
    pub fn repair(&self) -> (u64, u64) {
        let _order = self.insert_lock.lock().unwrap_or_else(|e| e.into_inner());
        let (entry, _) = self.graph.entry_top();
        self.graph.repair(entry, None)
    }

    /// Marks the nodes with `mask[i]` true as deleted (CONTRACT 13.3).
    pub fn delete(&mut self, mask: &[bool]) -> Result<(), String> {
        if self.row_ids.is_some() {
            return Err("hnsw: delete after a compaction rebuild is not supported".into());
        }
        if mask.len() != self.rows {
            return Err(format!("delete mask has {} rows, the corpus has {}", mask.len(), self.rows));
        }
        let mut all = self.tombstones.as_ref().map_or(vec![false; self.rows], Tombstones::to_mask);
        all.iter_mut().zip(mask).for_each(|(a, &m)| *a |= m);
        self.tombstones = Some(Tombstones::from_mask(&all));
        Ok(())
    }

    /// Replaces the vectors of rows `ids` and re-inserts each node with its level
    /// (CONTRACT 13.3), then runs the repair pass once. Returns (step A, step B) edges.
    pub fn update(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(u64, u64), String> {
        if self.row_ids.is_some() {
            return Err("hnsw: update after a compaction rebuild is not supported".into());
        }
        let dim = self.graph.vectors.cols;
        let active = self.graph.active.load(AtOrd::Acquire);
        check_update(ids, vectors, active, dim)?;
        let mut s = Scratch::new(self.graph.vectors.rows);
        let mut buf = Vec::new();
        for (j, &id) in ids.iter().enumerate() {
            let i = id as u32;
            let row = id as usize;
            self.graph.vectors.data[row * dim..(row + 1) * dim]
                .copy_from_slice(&vectors[j * dim..(j + 1) * dim]);
            let g = &self.graph;
            let level = g.levels[row] as usize;
            // Drop the out-edges, and remove i from the lists of its old neighbors.
            let mut start = None;
            for layer in (0..=level).rev() {
                let l = &g.layers[layer];
                let mut old = Vec::new();
                l.read(i, &mut old);
                l.write(i, &[]);
                for &nb in &old {
                    let _g = g.lock(nb);
                    l.read(nb, &mut buf);
                    if buf.contains(&i) {
                        buf.retain(|&x| x != i);
                        l.write(nb, &buf);
                    }
                }
                if start.is_none() {
                    start = old.first().map(|&nb| (nb, layer));
                }
            }
            let (ep, _) = g.entry_top();
            let start = if ep == i { start } else { None };
            if ep == i && start.is_none() {
                continue; // The only node of the graph: nothing to link.
            }
            g.insert_with(i, &mut s, true, start);
        }
        let (entry, _) = self.graph.entry_top();
        Ok(self.graph.repair(entry, None))
    }

    /// Compaction, rebuild mode (CONTRACT 13.3): a new graph from the live rows, with
    /// the same parameters, seed, and threads. Call it inside the build's thread pool.
    pub fn compact(&mut self) -> Result<(), String> {
        if self.row_ids.is_some() {
            return Ok(());
        }
        let g = &self.graph;
        let n = g.active.load(AtOrd::Acquire);
        let dim = g.vectors.cols;
        let tomb = self.tombstones.as_ref();
        let live: Vec<u32> = (0..n as u32)
            .filter(|&i| !tomb.is_some_and(|t| t.is_deleted(i as usize)))
            .collect();
        let mut data = Vec::with_capacity(live.len() * dim);
        for &i in &live {
            data.extend_from_slice(g.vectors.row(i as usize));
        }
        let vectors = Matrix {
            data,
            rows: live.len(),
            cols: dim,
        };
        let params = Params::new()
            .with("m", Int(g.m as i64))
            .with("ef_construct", Int(g.ef_construct as i64));
        let mut fresh = build(vectors, &params, self.build_threads, self.seed)?;
        fresh.filters = std::mem::take(&mut self.filters);
        fresh.rows = self.rows;
        fresh.row_ids = Some(live);
        fresh.times = self.times;
        *self = fresh;
        Ok(())
    }

    /// Compaction, repair mode: the in-place repair described in the module docs.
    /// Returns (step A, step B) edges of the final repair pass.
    pub fn compact_repair(&mut self) -> Result<(u64, u64), String> {
        if self.row_ids.is_some() {
            return Err("hnsw: repair after a compaction rebuild is not supported".into());
        }
        let Some(t) = self.tombstones.as_ref() else {
            let (entry, _) = self.graph.entry_top();
            return Ok(self.graph.repair(entry, None));
        };
        let g = &self.graph;
        let n = g.active.load(AtOrd::Acquire);
        let dead: Vec<bool> = (0..g.vectors.rows).map(|i| i < n && t.is_deleted(i)).collect();
        let (mut cur, mut dl) = (Vec::new(), Vec::new());
        for (layer, l) in g.layers.iter().enumerate() {
            for v in (0..n as u32).filter(|&v| g.levels[v as usize] as usize >= layer && !dead[v as usize]) {
                l.read(v, &mut cur);
                if !cur.iter().any(|&u| dead[u as usize]) {
                    continue;
                }
                let mut cand: Vec<u32> = cur.iter().copied().filter(|&u| !dead[u as usize]).collect();
                for &d in cur.iter().filter(|&&u| dead[u as usize]) {
                    l.read(d, &mut dl);
                    for &u in &dl {
                        if u != v && !dead[u as usize] && !cand.contains(&u) {
                            cand.push(u);
                        }
                    }
                }
                let base = g.vectors.row(v as usize);
                let mut scored: Vec<Cand> = cand
                    .iter()
                    .map(|&u| Cand {
                        score: dot(base, g.vectors.row(u as usize)),
                        id: u,
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.cmp(a));
                let kept = select_heuristic(&g.vectors, &scored, l.cap);
                l.write(v, &kept);
            }
        }
        for (layer, l) in g.layers.iter().enumerate() {
            for v in (0..n as u32).filter(|&v| g.levels[v as usize] as usize >= layer && dead[v as usize]) {
                l.write(v, &[]);
            }
        }
        let (ep, _) = g.entry_top();
        if dead[ep as usize] {
            let best = (0..n as u32)
                .filter(|&v| !dead[v as usize])
                .max_by_key(|&v| (g.levels[v as usize], Reverse(v)))
                .ok_or("hnsw: every node is deleted")?;
            let top = g.levels[best as usize] as usize;
            *g.entry.lock().unwrap_or_else(|e| e.into_inner()) = (best, top);
            g.entry_top.store(pack(best, top), AtOrd::Release);
        }
        let (entry, _) = g.entry_top();
        Ok(g.repair(entry, Some(&dead)))
    }
    /// Level of every node, in row order (all N rows, inserted or not).
    pub fn levels(&self) -> &[u8] {
        &self.graph.levels
    }
    /// The entry point of search.
    pub fn entry_point(&self) -> u32 {
        self.graph.entry_top().0
    }
    /// Number of layers (top layer + 1) of the current graph.
    pub fn num_layers(&self) -> usize {
        self.graph.entry_top().1 + 1
    }
    /// Slot limit per node on `layer`: 2m on layer 0, m above.
    pub fn layer_cap(&self, layer: usize) -> usize {
        self.graph.layers[layer].cap
    }
    /// Stored count of every node block on `layer`, before clamping (a snapshot).
    pub fn layer_counts(&self, layer: usize) -> Vec<u32> {
        let l = &self.graph.layers[layer];
        (0..l.counts.len()).map(|b| l.count(b)).collect()
    }
    /// Neighbor IDs of `node` on `layer` (empty if the node is not on that layer), a snapshot.
    pub fn neighbors(&self, layer: usize, node: u32) -> Vec<i32> {
        let l = &self.graph.layers[layer];
        match l.block_opt(node) {
            Some(_) => {
                let mut out = Vec::new();
                l.read(node, &mut out);
                out.into_iter().map(|v| v as i32).collect()
            }
            None => Vec::new(),
        }
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
    /// Nodes on each layer, layer 0 first (all N rows, by their drawn level).
    pub fn nodes_per_layer(&self) -> Vec<usize> {
        self.graph.layers[..self.num_layers()]
            .iter()
            .map(|l| l.counts.len())
            .collect()
    }
    fn edges(&self) -> u64 {
        self.graph
            .layers
            .iter()
            .map(|l| {
                (0..l.counts.len())
                    .map(|b| l.count(b) as u64)
                    .sum::<u64>()
            })
            .sum()
    }
    fn bookkeeping_bytes(&self) -> u64 {
        let layers = &self.graph.layers[..self.num_layers()];
        let counts: u64 = layers.iter().map(|l| l.counts.len() as u64 * 4).sum();
        let maps: u64 = layers.iter().map(|l| l.map.len() as u64 * 4).sum();
        self.graph.levels.len() as u64 + counts + maps
    }
}

/// Edges x 4 bytes, plus levels (1 byte per node), counts (4 bytes per node per
/// layer), and the node maps of layers >= 1 (4 bytes per corpus row per layer).
/// With changes (CONTRACT 13): + the tombstone bit set (N/8 bytes), or + the
/// position -> row ID map (4 bytes per live row) after a compaction rebuild.
pub fn index_bytes(index: &HnswIndex) -> u64 {
    index.edges() * 4
        + index.bookkeeping_bytes()
        + index.tombstones.as_ref().map_or(0, Tombstones::bytes)
        + index.row_ids.as_ref().map_or(0, |r| r.len() as u64 * 4)
}

impl AnnIndex for HnswIndex {
    fn set_filter_dir(&mut self, dir: &str) {
        HnswIndex::set_filter_dir(self, dir);
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
    fn insert(&self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        HnswIndex::insert(self, ids, vectors)
    }
    fn repair(&self) -> Result<(u64, u64), String> {
        Ok(HnswIndex::repair(self))
    }
    fn delete(&mut self, mask: &[bool]) -> Result<(), String> {
        HnswIndex::delete(self, mask)
    }
    fn update(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(), String> {
        HnswIndex::update(self, ids, vectors).map(|_| ())
    }
    fn compact(&mut self) -> Result<(), String> {
        HnswIndex::compact(self)
    }
    fn compact_repair(&mut self) -> Result<(), String> {
        HnswIndex::compact_repair(self).map(|_| ())
    }
    fn supports_concurrency(&self) -> bool {
        true
    }
    fn extra(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut e = serde_json::Map::new();
        e.insert("top_layer".into(), (self.num_layers() - 1).into());
        e.insert("entry_point".into(), self.entry_point().into());
        e.insert("nodes_per_layer".into(), self.nodes_per_layer().into());
        e.insert("edges".into(), self.edges().into());
        e.insert("edge_bytes".into(), (self.edges() * 4).into());
        e.insert("bookkeeping_bytes".into(), self.bookkeeping_bytes().into());
        e.insert("m".into(), self.graph.m.into());
        e.insert("ef_construct".into(), self.graph.ef_construct.into());
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
