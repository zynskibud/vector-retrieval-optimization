//! CONTRACT section 9, tests 4, 5, and 6 for diskann, plus graph and file invariants (6.7).
//! All tests use the first 20,000 dev rows. Truth is brute force over those rows, computed here.

use bench::{diskann, npy, BuildContext, Matrix, ParamValue, Params};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

const N: usize = 20_000;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dev_dir() -> PathBuf {
    repo_root().join("data/processed/dev")
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn tmp_out(name: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "rust-diskann-test-{}-{name}.json",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

struct Data {
    vectors: Matrix,
    queries: Matrix,
    truth: Vec<HashSet<i64>>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let vectors = npy::read_f32(&dev_dir().join("vectors.npy"), Some(N)).unwrap();
        let queries = npy::read_f32(&dev_dir().join("queries.npy"), None).unwrap();
        let flat = bench::flat::build(vectors.clone(), &Params::new(), 1, 42).unwrap();
        let truth = (0..queries.rows)
            .map(|i| {
                bench::flat::search(&flat, queries.row(i), 10, &Params::new())
                    .ids
                    .into_iter()
                    .collect()
            })
            .collect();
        Data {
            vectors,
            queries,
            truth,
        }
    })
}

fn build_with(threads: usize, metric: &str, name: &str) -> diskann::DiskannIndex {
    let d = data();
    let params = diskann::build_defaults(N).with("metric", ParamValue::Str(metric.into()));
    let ctx = BuildContext {
        out_path: tmp_out(name),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| diskann::build(d.vectors.clone(), &params, threads, 42, &ctx))
        .unwrap()
}

/// Tests that use the shared index. The last one to finish deletes its `.diskann` file.
const SHARED_USERS: usize = 4;
static DONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Counts a finished user of the shared index on drop (also on panic).
struct SharedGuard;

impl Drop for SharedGuard {
    fn drop(&mut self) {
        if DONE.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 == SHARED_USERS {
            std::fs::remove_file(format!("{}.diskann", tmp_out("main"))).ok();
        }
    }
}

/// The shared all-cores build with metric=ip.
fn index() -> &'static diskann::DiskannIndex {
    static I: OnceLock<diskann::DiskannIndex> = OnceLock::new();
    I.get_or_init(|| build_with(threads(), "ip", "main"))
}

fn sp(l: i64, io: &str) -> Params {
    diskann::search_defaults()
        .with("l", ParamValue::Int(l))
        .with("io", ParamValue::Str(io.into()))
}

struct Run {
    ids: Vec<Vec<i64>>,
    latency_ms: Vec<f64>,
    disk_reads: f64,
    recall: f64,
}

fn run(index: &diskann::DiskannIndex, p: &Params) -> Run {
    let d = data();
    let q = d.queries.rows;
    let (mut ids, mut latency_ms) = (Vec::with_capacity(q), Vec::with_capacity(q));
    let (mut hits, mut reads) = (0usize, 0.0);
    for i in 0..q {
        let t = Instant::now();
        let r = diskann::search(index, d.queries.row(i), 10, p).unwrap();
        latency_ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(r.ids.len(), 10);
        for w in r.ids.windows(2).zip(r.scores.windows(2)) {
            let (id, sc) = w;
            assert!(sc[0] > sc[1] || (sc[0] == sc[1] && id[0] < id[1]) || id[1] == -1);
        }
        hits += r.ids.iter().filter(|id| d.truth[i].contains(id)).count();
        reads += r.counters["disk_reads"];
        ids.push(r.ids);
    }
    Run {
        ids,
        latency_ms,
        disk_reads: reads / q as f64,
        recall: hits as f64 / (q * 10) as f64,
    }
}

fn p50(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

#[test]
fn recall_floor_and_monotonic_in_l() {
    let _guard = SharedGuard;
    let idx = index();
    let r50 = run(idx, &sp(50, "mmap")).recall;
    let r100 = run(idx, &sp(100, "mmap")).recall;
    let r200 = run(idx, &sp(200, "mmap")).recall;
    eprintln!("20k recall@10 ip: l=50 {r50:.4} l=100 {r100:.4} l=200 {r200:.4}");
    assert!(r100 >= 0.90, "recall@10 at l=100 is {r100}");
    assert!(r200 >= r100 && r100 >= r50, "{r50} {r100} {r200}");
}

#[test]
fn nocache_matches_mmap_and_is_slower() {
    let _guard = SharedGuard;
    let idx = index();
    // mmap first, then nocache: the switch must drop the cached pages (6.7.1, item 1).
    let mmap = run(idx, &sp(100, "mmap"));
    let nocache = run(idx, &sp(100, "nocache"));
    for (i, (a, b)) in mmap.ids.iter().zip(&nocache.ids).enumerate() {
        assert_eq!(a, b, "query {i}: mmap and nocache ids differ");
    }
    let (pm, pn) = (p50(&mmap.latency_ms), p50(&nocache.latency_ms));
    eprintln!(
        "l=100 p50 mmap {pm:.3} ms, nocache {pn:.3} ms; disk_reads {:.1}",
        nocache.disk_reads
    );
    assert!(nocache.disk_reads > 0.0);
    assert_eq!(nocache.disk_reads, mmap.disk_reads);
    assert!(pn >= 2.0 * pm, "nocache p50 {pn} < 2 x mmap p50 {pm}");
}

#[test]
fn file_layout_and_out_degrees() {
    let _guard = SharedGuard;
    let idx = index();
    let bytes = std::fs::read(idx.disk_path()).unwrap();
    assert_eq!(idx.record_bytes(), 4096);
    assert_eq!(bytes.len(), N * 4096);
    assert_eq!(idx.disk_bytes(), (N * 4096) as u64);
    let d = data();
    let r = idx.r();
    assert_eq!(r, 64);
    let mut edges = 0usize;
    for (i, rec) in bytes.chunks_exact(4096).enumerate() {
        for (b, &x) in rec[..384 * 4].chunks_exact(4).zip(d.vectors.row(i)) {
            assert_eq!(f32::from_le_bytes(b.try_into().unwrap()), x);
        }
        let out: Vec<i32> = rec[384 * 4..384 * 4 + r * 4]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let valid: Vec<i32> = out.iter().copied().filter(|&v| v >= 0).collect();
        // Empty slots come after all edges.
        assert!(out[valid.len()..].iter().all(|&v| v == -1), "node {i}");
        assert!(
            (1..=r).contains(&valid.len()),
            "node {i}: {} edges",
            valid.len()
        );
        let uniq: HashSet<i32> = valid.iter().copied().collect();
        assert_eq!(uniq.len(), valid.len(), "node {i}: repeated edge");
        assert!(valid.iter().all(|&v| (v as usize) < N && v as usize != i));
        assert!(rec[384 * 4 + r * 4..].iter().all(|&b| b == 0));
        edges += valid.len();
    }
    let mean = edges as f64 / N as f64;
    assert!((idx.mean_out_degree() - mean).abs() < 1e-9);
    // RAM: codes + codebooks + entry point.
    assert_eq!(
        diskann::index_bytes(idx),
        (N * 48 + 48 * 256 * 8 * 4 + 4) as u64
    );
    eprintln!("mean out-degree {mean:.2}, entry {}", idx.entry_point());
}

#[test]
fn single_thread_and_parallel_recall_match() {
    let _guard = SharedGuard;
    let seq = build_with(1, "ip", "seq");
    let (rs, rp) = (
        run(&seq, &sp(100, "mmap")).recall,
        run(index(), &sp(100, "mmap")).recall,
    );
    std::fs::remove_file(seq.disk_path()).ok();
    eprintln!("20k recall@10 l=100: threads=1 {rs:.4}, all cores {rp:.4}");
    assert!((rs - rp).abs() <= 0.01, "{rs} vs {rp}");
}

#[test]
fn l2_metric_meets_floor() {
    let idx = build_with(threads(), "l2", "l2");
    let r = run(&idx, &sp(100, "mmap")).recall;
    std::fs::remove_file(idx.disk_path()).ok();
    eprintln!("20k recall@10 l2 l=100: {r:.4}");
    assert!(r >= 0.90, "metric=l2 recall {r}");
}

#[test]
fn bench_run_writes_json() {
    let out = tmp_out("bench");
    let status = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args(["--index", "diskann", "--data", "data/processed/dev"])
        .args(["--limit", "20000", "--warmup", "10"])
        .args(["--search", "l=50,io=mmap", "--search", "l=50,io=nocache"])
        .args(["--out", &out])
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    std::fs::remove_file(format!("{out}.diskann")).ok();
    assert_eq!(v["index"], "diskann");
    assert_eq!(v["n"], 20000);
    assert_eq!(v["extra"]["disk_bytes"], 20000 * 4096);
    assert!(v["build"]["index_bytes"].as_u64().unwrap() > 0);
    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 2);
    for (s, io) in searches.iter().zip(["mmap", "nocache"]) {
        assert_eq!(s["search_params"]["io"], io);
        assert_eq!(s["search_params"]["l"], 50);
        assert!(s["extra"]["disk_reads"].as_f64().unwrap() > 0.0);
        assert!(s["extra"]["disk_bytes_read"].as_f64().unwrap() > 0.0);
        let ids = s["ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|r| r.as_array().unwrap().len() == 10));
    }
    assert_eq!(searches[0]["ids"], searches[1]["ids"]);
}
