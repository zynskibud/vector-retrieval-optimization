//! CONTRACT section 9, tests 4 and 5 for ivf_pq: recall floor for metric=ip and metric=l2
//! on the dev set, CSR layout, determinism, and one real `bench` run.

use bench::{ivf_pq, npy, ParamValue, Params};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dev_dir() -> PathBuf {
    repo_root().join("data/processed/dev")
}

fn sp(nprobe: i64, rerank: i64) -> Params {
    Params::new()
        .with("nprobe", ParamValue::Int(nprobe))
        .with("rerank", ParamValue::Int(rerank))
}

/// Recall@10 of one search setting over all queries. Checks order, IDs, counters, and l2 signs.
fn recall(
    index: &ivf_pq::IvfPqIndex,
    n: usize,
    queries: &npy::Matrix,
    gt: &npy::Array2<i64>,
    p: &Params,
    l2: bool,
) -> f64 {
    let nlist = index.nlist() as u64;
    let mut hits = 0usize;
    for i in 0..queries.rows {
        let r = ivf_pq::search(index, queries.row(i), 10, p).unwrap();
        assert!(r.scores.windows(2).all(|w| w[0] >= w[1]), "not best first");
        assert!(r
            .ids
            .iter()
            .all(|&id| id == -1 || (0..n as i64).contains(&id)));
        let dc = r.distance_computations.unwrap();
        assert!(dc >= nlist && dc <= nlist + 2 * n as u64, "bad count {dc}");
        if l2 {
            assert!(
                r.scores.iter().all(|&s| s <= 0.0),
                "l2 score > 0: {:?}",
                r.scores
            );
        }
        let truth: HashSet<i64> = gt.row(i)[..10].iter().copied().collect();
        hits += r.ids.iter().filter(|id| truth.contains(id)).count();
    }
    hits as f64 / (queries.rows * 10) as f64
}

fn check_csr(index: &ivf_pq::IvfPqIndex, n: usize) {
    let off = index.offsets();
    let m = index.m();
    assert_eq!(off.len(), index.nlist() + 1);
    assert_eq!(off[0], 0);
    assert_eq!(*off.last().unwrap() as usize, n);
    assert!(off.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(index.codes().len(), n * m);
    let mut seen = vec![false; n];
    for &id in index.list_ids() {
        assert!(!seen[id as usize], "row {id} appears twice");
        seen[id as usize] = true;
    }
    assert!(seen.iter().all(|&s| s));
    let nlist = index.nlist() as u64;
    let (n, m) = (n as u64, m as u64);
    assert_eq!(
        ivf_pq::index_bytes(index),
        nlist * 384 * 4 + n * 4 + (nlist + 1) * 4 + m * 256 * (384 / m) * 4 + n * m
    );
}

fn check_metric(metric: &str) {
    let dir = dev_dir();
    let vectors = npy::read_f32(&dir.join("vectors.npy"), None).unwrap();
    let queries = npy::read_f32(&dir.join("queries.npy"), None).unwrap();
    let gt = npy::read_i64(&dir.join("ground_truth.npy"), None).unwrap();
    let n = vectors.rows;
    let params = ivf_pq::build_defaults(n).with("metric", ParamValue::Str(metric.into()));
    let index = ivf_pq::build(vectors, &params, 0, 42).unwrap();
    check_csr(&index, n);

    let l2 = metric == "l2";
    let r8 = recall(&index, n, &queries, &gt, &sp(8, 0), l2);
    let r8r = recall(&index, n, &queries, &gt, &sp(8, 100), l2);
    let r32 = recall(&index, n, &queries, &gt, &sp(32, 0), l2);
    eprintln!(
        "ivf_pq metric={metric}: nprobe=8 {r8:.4}, nprobe=8 rerank=100 {r8r:.4}, nprobe=32 {r32:.4}"
    );
    assert!(r8 >= 0.45, "metric={metric} recall {r8} < 0.45");
    assert!(
        r8r >= r8,
        "metric={metric} rerank=100 {r8r} < rerank=0 {r8}"
    );
    assert!(r32 >= r8, "metric={metric} nprobe=32 {r32} < nprobe=8 {r8}");
}

#[test]
fn ivf_pq_ip_recall() {
    check_metric("ip");
}

#[test]
fn ivf_pq_l2_recall() {
    check_metric("l2");
}

#[test]
fn ivf_pq_same_seed_same_codes() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(20_000)).unwrap();
    let n = v.rows;
    let params = ivf_pq::build_defaults(n)
        .with("nlist", ParamValue::Int(64))
        .with("train_size", ParamValue::Int(5000))
        .with("iters", ParamValue::Int(5));
    let a = ivf_pq::build(v.clone(), &params, 0, 7).unwrap();
    let b = ivf_pq::build(v, &params, 0, 7).unwrap();
    check_csr(&a, n);
    assert_eq!(a.centers(), b.centers());
    assert_eq!(a.codebooks(), b.codebooks());
    assert_eq!(a.list_ids(), b.list_ids());
    assert_eq!(a.offsets(), b.offsets());
    assert_eq!(a.codes(), b.codes());
}

#[test]
fn ivf_pq_bench_run_writes_two_searches() {
    let out = std::env::temp_dir().join(format!("rust-ivfpq-test-{}.json", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args(["--index", "ivf_pq", "--data", "data/processed/dev"])
        .args(["--limit", "20000", "--warmup", "10", "--build", "metric=l2"])
        .args([
            "--search",
            "nprobe=8,rerank=0",
            "--search",
            "nprobe=8,rerank=100",
        ])
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    assert_eq!(v["index"], "ivf_pq");
    assert_eq!(v["n"], 20000);
    assert_eq!(v["build_params"]["metric"], "l2");
    assert_eq!(v["extra"]["id_bytes"], 4);
    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 2);
    for s in searches {
        let ids = s["ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|r| r.as_array().unwrap().len() == 10));
        assert!(s["distance_computations"].as_f64().unwrap() >= 1024.0);
    }
}
