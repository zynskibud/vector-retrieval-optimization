//! CONTRACT section 9, test 4 for hnsw, plus graph invariants (CONTRACT 6.6).
//! One parallel build of the full dev set (100k rows) is shared by the tests.

use bench::{hnsw, npy, splitmix::SplitMix64, Matrix, ParamValue, Params};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dev_dir() -> PathBuf {
    repo_root().join("data/processed/dev")
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

struct Fixture {
    index: hnsw::HnswIndex,
    queries: Matrix,
    gt: npy::Array2<i64>,
    n: usize,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let dir = dev_dir();
        let vectors = npy::read_f32(&dir.join("vectors.npy"), None).unwrap();
        let queries = npy::read_f32(&dir.join("queries.npy"), None).unwrap();
        let gt = npy::read_i64(&dir.join("ground_truth.npy"), None).unwrap();
        let n = vectors.rows;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads())
            .build()
            .unwrap();
        let index = pool
            .install(|| hnsw::build(vectors, &hnsw::build_defaults(n), threads(), 42))
            .unwrap();
        Fixture {
            index,
            queries,
            gt,
            n,
        }
    })
}

fn ef(v: i64) -> Params {
    Params::new().with("ef", ParamValue::Int(v))
}

fn recall(f: &Fixture, ef_value: i64) -> f64 {
    let p = ef(ef_value);
    let mut hits = 0usize;
    for i in 0..f.queries.rows {
        let r = hnsw::search(&f.index, f.queries.row(i), 10, &p).unwrap();
        let truth: HashSet<i64> = f.gt.row(i)[..10].iter().copied().collect();
        hits += r.ids.iter().filter(|id| truth.contains(id)).count();
    }
    hits as f64 / (f.queries.rows * 10) as f64
}

#[test]
fn recall_floor_and_monotonic_in_ef() {
    let f = fixture();
    let r16 = recall(f, 16);
    let r64 = recall(f, 64);
    let r128 = recall(f, 128);
    eprintln!("recall@10: ef16={r16:.4} ef64={r64:.4} ef128={r128:.4}");
    assert!(r64 >= 0.95, "recall@10 at ef=64 is {r64}");
    assert!(r128 >= r64 && r64 >= r16, "{r16} {r64} {r128}");
}

#[test]
fn results_sorted_and_distance_count_in_range() {
    let f = fixture();
    let p = hnsw::search_defaults();
    for i in 0..f.queries.rows {
        let r = hnsw::search(&f.index, f.queries.row(i), 10, &p).unwrap();
        let dc = r.distance_computations.unwrap();
        assert!(
            dc > 0 && (dc as usize) < f.n,
            "query {i}: {dc} dot products"
        );
        assert_eq!(r.ids.len(), 10);
        assert!(r.ids.iter().all(|&id| id >= 0));
        for w in r.ids.windows(2).zip(r.scores.windows(2)) {
            let (ids, sc) = w;
            assert!(sc[0] > sc[1] || (sc[0] == sc[1] && ids[0] < ids[1]));
        }
    }
}

#[test]
fn levels_deterministic_and_distribution() {
    let f = fixture();
    let levels = f.index.levels();
    assert_eq!(levels, hnsw::draw_levels(f.n, 16, 42).as_slice());
    // Independent recomputation from the contract formula.
    let ml = 1.0 / 16f64.ln();
    let mut rng = SplitMix64::new(42);
    for &l in levels.iter().take(1000) {
        let u = rng.next_f64();
        assert_eq!(l as f64, (-u.ln() * ml).floor());
    }
    let frac = levels.iter().filter(|&&l| l >= 1).count() as f64 / f.n as f64;
    assert!((0.04..=0.09).contains(&frac), "fraction level>=1: {frac}");
    // Entry point: highest level, ties to the lowest row (holds for any build
    // in which the first node to reach the top level is the lowest such row).
    let top = *levels.iter().max().unwrap();
    assert_eq!(f.index.num_layers(), top as usize + 1);
    assert_eq!(levels[f.index.entry_point() as usize], top);
}

#[test]
fn slot_counts_within_limits() {
    let f = fixture();
    for layer in 0..f.index.num_layers() {
        let cap = f.index.layer_cap(layer);
        assert_eq!(cap, if layer == 0 { 32 } else { 16 });
        for &c in f.index.layer_counts(layer) {
            assert!(c as usize <= cap, "layer {layer}: count {c} > {cap}");
        }
    }
    let per_layer = f.index.nodes_per_layer();
    assert_eq!(per_layer[0], f.n);
    for (layer, &count) in per_layer.iter().enumerate() {
        let expect = f
            .index
            .levels()
            .iter()
            .filter(|&&l| l as usize >= layer)
            .count();
        assert_eq!(count, expect);
    }
}

fn reachable(index: &hnsw::HnswIndex, n: usize) -> usize {
    let mut seen = vec![false; n];
    let mut stack = vec![index.entry_point()];
    seen[index.entry_point() as usize] = true;
    let mut reached = 1usize;
    while let Some(v) = stack.pop() {
        for &u in index.neighbors(0, v) {
            assert!(u >= 0 && (u as usize) < n && u as u32 != v);
            if !seen[u as usize] {
                seen[u as usize] = true;
                reached += 1;
                stack.push(u as u32);
            }
        }
    }
    reached
}

fn recall_on(index: &hnsw::HnswIndex, q: &Matrix, truth: &[Vec<i64>]) -> f64 {
    let p = ef(64);
    let mut hits = 0usize;
    for (i, t) in truth.iter().enumerate() {
        let r = hnsw::search(index, q.row(i), 10, &p).unwrap();
        hits += r.ids.iter().filter(|id| t.contains(id)).count();
    }
    hits as f64 / (truth.len() * 10) as f64
}

/// Layer 0 is fully reachable after the repair pass, for threads = 1 and all
/// cores on 20,000 rows, and both builds give recall@10 at ef=64 within 0.01.
/// Truth is brute force over the 20,000 rows, computed here.
#[test]
fn layer0_connected_and_parallel_recall_matches() {
    let n = 20000;
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(n)).unwrap();
    let q = npy::read_f32(&dev_dir().join("queries.npy"), None).unwrap();
    let flat = bench::flat::build(v.clone(), &Params::new(), 1, 42).unwrap();
    let truth: Vec<Vec<i64>> = (0..q.rows)
        .map(|i| bench::flat::search(&flat, q.row(i), 10, &Params::new()).ids)
        .collect();
    let p = hnsw::build_defaults(n);
    let seq = hnsw::build(v.clone(), &p, 1, 42).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads())
        .build()
        .unwrap();
    let par = pool.install(|| hnsw::build(v, &p, threads(), 42)).unwrap();
    for (name, idx) in [("threads=1", &seq), ("all cores", &par)] {
        eprintln!(
            "{name}: unreachable_before_repair={} repair_added={} repair_added_unreachable={}",
            idx.unreachable_before_repair(),
            idx.repair_added(),
            idx.repair_added_unreachable()
        );
        assert_eq!(reachable(idx, n), n, "{name}: layer 0 not fully reachable");
    }
    let (rs, rp) = (recall_on(&seq, &q, &truth), recall_on(&par, &q, &truth));
    eprintln!("20k recall@10 ef=64: threads=1 {rs:.4}, all cores {rp:.4}");
    assert!((rs - rp).abs() <= 0.01, "{rs} vs {rp}");
}

#[test]
fn layer0_connected_full_dev() {
    let f = fixture();
    assert_eq!(reachable(&f.index, f.n), f.n);
}

#[test]
fn single_thread_build_is_deterministic() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(3000)).unwrap();
    let q = npy::read_f32(&dev_dir().join("queries.npy"), Some(50)).unwrap();
    let p = hnsw::build_defaults(v.rows);
    let a = hnsw::build(v.clone(), &p, 1, 42).unwrap();
    let b = hnsw::build(v, &p, 1, 42).unwrap();
    assert_eq!(a.entry_point(), b.entry_point());
    for layer in 0..a.num_layers() {
        for node in 0..3000u32 {
            assert_eq!(a.neighbors(layer, node), b.neighbors(layer, node));
        }
    }
    let s = hnsw::search_defaults();
    for i in 0..q.rows {
        let ra = hnsw::search(&a, q.row(i), 10, &s).unwrap();
        let rb = hnsw::search(&b, q.row(i), 10, &s).unwrap();
        assert_eq!(ra, rb);
    }
}

#[test]
fn pads_short_results() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(3)).unwrap();
    let q = v.row(0).to_vec();
    let index = hnsw::build(v, &hnsw::build_defaults(3), 1, 42).unwrap();
    let r = hnsw::search(&index, &q, 5, &hnsw::search_defaults()).unwrap();
    assert_eq!(r.ids[0], 0);
    assert_eq!(&r.ids[3..], &[-1, -1]);
    assert!(r.scores[4] == f32::NEG_INFINITY);
}

#[test]
fn bench_run_writes_two_searches() {
    let out = std::env::temp_dir().join(format!("rust-hnsw-test-{}.json", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args([
            "--index",
            "hnsw",
            "--data",
            "data/processed/dev",
            "--limit",
            "20000",
        ])
        .args([
            "--search", "ef=16", "--search", "ef=64", "--warmup", "10", "--out",
        ])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    assert_eq!(v["index"], "hnsw");
    assert_eq!(v["n"], 20000);
    assert!(v["build"]["index_bytes"].as_u64().unwrap() > 0);
    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 2);
    for (s, e) in searches.iter().zip([16, 64]) {
        assert_eq!(s["search_params"]["ef"], e);
        let ids = s["ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|r| r.as_array().unwrap().len() == 10));
        let dc = s["distance_computations"].as_f64().unwrap();
        assert!(dc > 0.0 && dc < 20000.0);
    }
}
