//! CONTRACT section 9, test 7: the output JSON of a real `bench` run has every key and the right shapes.
//! Also checks the exit codes of section 2.

use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bench() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bench"));
    c.current_dir(repo_root());
    c
}

fn tmp_json(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "rust-bench-test-{}-{name}.json",
        std::process::id()
    ))
}

fn keys(v: &Value) -> Vec<&str> {
    let mut k: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    k.sort_unstable();
    k
}

#[test]
fn flat_run_writes_valid_json() {
    let out = tmp_json("flat");
    let status = bench()
        .args([
            "--index",
            "flat",
            "--data",
            "data/processed/dev",
            "--limit",
            "20000",
        ])
        .args(["--k", "10", "--warmup", "10", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();

    let mut top = vec![
        "contract_version",
        "language",
        "index",
        "data_dir",
        "n",
        "dim",
        "q",
        "k",
        "threads",
        "seed",
        "build_params",
        "build",
        "searches",
        "machine",
        "extra",
    ];
    top.sort_unstable();
    assert_eq!(keys(&v), top);
    assert_eq!(v["contract_version"], 1);
    assert_eq!(v["language"], "rust");
    assert_eq!(v["index"], "flat");
    assert_eq!(v["n"], 20000);
    assert_eq!(v["dim"], 384);
    assert_eq!(v["q"], 1000);
    assert_eq!(v["k"], 10);
    assert_eq!(v["seed"], 42);
    assert!(v["threads"].as_u64().unwrap() >= 1);
    assert!(v["build_params"].is_object());
    assert!(v["extra"].is_object());

    let b = &v["build"];
    assert_eq!(
        keys(b),
        ["add_s", "index_bytes", "peak_rss_mb", "total_s", "train_s"]
    );
    assert_eq!(b["index_bytes"], 0);
    assert!(b["peak_rss_mb"].as_f64().unwrap() > 0.0);

    let m = &v["machine"];
    assert_eq!(keys(m), ["arch", "cores", "cpu", "os"]);

    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 1);
    let s = &searches[0];
    assert_eq!(
        keys(s),
        [
            "distance_computations",
            "extra",
            "ids",
            "latency_ms",
            "qps",
            "scores",
            "search_params",
            "total_s"
        ]
    );
    assert_eq!(s["distance_computations"].as_f64(), Some(20000.0));
    assert_eq!(s["extra"], serde_json::json!({}));
    let ids = s["ids"].as_array().unwrap();
    let scores = s["scores"].as_array().unwrap();
    assert_eq!(ids.len(), 1000);
    assert_eq!(scores.len(), 1000);
    assert_eq!(s["latency_ms"].as_array().unwrap().len(), 1000);
    for (row_ids, row_scores) in ids.iter().zip(scores) {
        let row_ids = row_ids.as_array().unwrap();
        let row_scores = row_scores.as_array().unwrap();
        assert_eq!(row_ids.len(), 10);
        assert_eq!(row_scores.len(), 10);
        assert!(row_ids.iter().all(|x| x.is_i64()));
        assert!(row_scores.iter().all(|x| x.is_f64()));
    }
    assert!(s["qps"].as_f64().unwrap() > 0.0);
}

#[test]
fn unknown_index_exits_2() {
    let out = bench()
        .args([
            "--index",
            "nope",
            "--data",
            "data/processed/dev",
            "--out",
            "/dev/null",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn unknown_param_exits_2() {
    let out = bench()
        .args([
            "--index",
            "hnsw",
            "--data",
            "data/processed/dev",
            "--out",
            "/dev/null",
        ])
        .args(["--search", "ef=10,bogus=1"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("bogus"));
}

#[test]
fn stub_index_exits_1() {
    let out = bench()
        .args([
            "--index",
            "diskann", // still a stub until Wave 2b; that agent removes this test
            "--data",
            "data/processed/dev",
            "--limit",
            "1000",
        ])
        .args(["--out", "/dev/null"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not implemented"));
}
