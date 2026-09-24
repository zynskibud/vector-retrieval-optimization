//! CONTRACT section 9, tests 4 and 5 for pq: recall floor for metric=ip and metric=l2 on the dev set.

use bench::{npy, pq, ParamValue, Params};
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

fn rerank(r: i64) -> Params {
    Params::new().with("rerank", ParamValue::Int(r))
}

/// Recall@10 of one search setting over all queries.
fn recall(index: &pq::PqIndex, queries: &npy::Matrix, gt: &npy::Array2<i64>, r: i64) -> f64 {
    let p = rerank(r);
    let mut hits = 0usize;
    for i in 0..queries.rows {
        let res = pq::search(index, queries.row(i), 10, &p).unwrap();
        assert!(
            res.scores.windows(2).all(|w| w[0] >= w[1]),
            "not best first"
        );
        let truth: HashSet<i64> = gt.row(i)[..10].iter().copied().collect();
        hits += res.ids.iter().filter(|id| truth.contains(id)).count();
    }
    hits as f64 / (queries.rows * 10) as f64
}

fn check_metric(metric: &str) {
    let dir = dev_dir();
    let vectors = npy::read_f32(&dir.join("vectors.npy"), None).unwrap();
    let queries = npy::read_f32(&dir.join("queries.npy"), None).unwrap();
    let gt = npy::read_i64(&dir.join("ground_truth.npy"), None).unwrap();
    let n = vectors.rows;
    let params = pq::build_defaults(n).with("metric", ParamValue::Str(metric.into()));
    let index = pq::build(vectors, &params, 0, 42).unwrap();
    let m = index.m();
    assert_eq!(index.codes().len(), n * m);
    assert_eq!(
        pq::index_bytes(&index),
        (m * 256 * (384 / m) * 4 + n * m) as u64
    );

    let r0 = recall(&index, &queries, &gt, 0);
    let r100 = recall(&index, &queries, &gt, 100);
    eprintln!("pq metric={metric}: recall@10 rerank=0 {r0:.4}, rerank=100 {r100:.4}");
    assert!(r0 >= 0.50, "metric={metric} recall {r0} < 0.50");
    assert!(
        r100 >= r0,
        "metric={metric} rerank=100 {r100} < rerank=0 {r0}"
    );

    let res = pq::search(&index, queries.row(0), 10, &rerank(0)).unwrap();
    assert_eq!(res.distance_computations, Some(n as u64));
    if metric == "l2" {
        for i in 0..queries.rows {
            for r in [0, 100] {
                let res = pq::search(&index, queries.row(i), 10, &rerank(r)).unwrap();
                assert!(res.scores.iter().all(|&s| s <= 0.0), "l2 score > 0");
            }
        }
    }
}

#[test]
fn pq_ip_recall() {
    check_metric("ip");
}

#[test]
fn pq_l2_recall() {
    check_metric("l2");
}

#[test]
fn pq_same_seed_same_codes() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(20_000)).unwrap();
    let params = pq::build_defaults(v.rows)
        .with("train_size", ParamValue::Int(5000))
        .with("iters", ParamValue::Int(5));
    let a = pq::build(v.clone(), &params, 0, 7).unwrap();
    let b = pq::build(v, &params, 0, 7).unwrap();
    assert_eq!(a.codes(), b.codes());
    assert_eq!(a.codebooks(), b.codebooks());
}

#[test]
fn pq_bad_params_are_errors() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(1000)).unwrap();
    let p = pq::build_defaults(v.rows).with("nbits", ParamValue::Int(4));
    assert!(pq::build(v.clone(), &p, 0, 42)
        .err()
        .unwrap()
        .contains("nbits"));
    let p = pq::build_defaults(v.rows).with("metric", ParamValue::Str("cos".into()));
    assert!(pq::build(v, &p, 0, 42).err().unwrap().contains("metric"));
}

#[test]
fn pq_bench_run_writes_json() {
    let out = std::env::temp_dir().join(format!("rust-bench-test-{}-pq.json", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args([
            "--index",
            "pq",
            "--data",
            "data/processed/dev",
            "--limit",
            "20000",
        ])
        .args([
            "--build",
            "metric=l2",
            "--search",
            "rerank=0",
            "--search",
            "rerank=100",
        ])
        .args(["--warmup", "10", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    assert_eq!(v["index"], "pq");
    assert_eq!(v["build_params"]["metric"], "l2");
    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 2);
    for s in searches {
        let ids = s["ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|r| r.as_array().unwrap().len() == 10));
    }
}
