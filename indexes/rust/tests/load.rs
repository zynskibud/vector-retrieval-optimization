//! CONTRACT 12.5: load runs of the hnsw `bench` on 20,000 dev rows.
//! (a) 8 clients beat 1 client in qps, with zero errors and first-pass recall >= 0.95.
//! (b) 4 clients with inserts at 2,000 rows/s (build on 18,000 rows): zero errors,
//!     2,000 rows inserted, after-inserts recall within 0.01 of a static 20,000-row build.
//! (c) the JSON has the load keys.
//! The runs are timed, so the tests hold one lock and run one at a time.

use bench::{flat, npy, Params};
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

const N: usize = 20000;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Brute-force top-10 over the first N rows, per query.
fn truth() -> &'static Vec<Vec<i64>> {
    static T: OnceLock<Vec<Vec<i64>>> = OnceLock::new();
    T.get_or_init(|| {
        let dev = repo_root().join("data/processed/dev");
        let v = npy::read_f32(&dev.join("vectors.npy"), Some(N)).unwrap();
        let q = npy::read_f32(&dev.join("queries.npy"), None).unwrap();
        let f = flat::build(v, &Params::new(), 1, 42).unwrap();
        (0..q.rows)
            .map(|i| flat::search(&f, q.row(i), 10, &Params::new()).ids)
            .collect()
    })
}

/// Runs `bench --index hnsw` on N rows with `extra` arguments and returns its JSON.
fn run(name: &str, extra: &[&str]) -> Value {
    let out = std::env::temp_dir().join(format!(
        "rust-load-test-{}-{name}.json",
        std::process::id()
    ));
    let o = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args(["--index", "hnsw", "--data", "data/processed/dev"])
        .args(["--limit", &N.to_string(), "--search", "ef=64", "--warmup", "10"])
        .args(extra)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    v
}

fn recall(search: &Value) -> f64 {
    let t = truth();
    let ids = search["ids"].as_array().unwrap();
    assert_eq!(ids.len(), t.len());
    let hits: usize = ids
        .iter()
        .zip(t)
        .map(|(row, tr)| {
            row.as_array()
                .unwrap()
                .iter()
                .filter(|id| tr.contains(&id.as_i64().unwrap()))
                .count()
        })
        .sum();
    hits as f64 / (t.len() * 10) as f64
}

fn num(s: &Value, key: &str) -> f64 {
    s["extra"][key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing extra.{key}"))
}

#[test]
fn eight_clients_beat_one() {
    let _g = serial();
    truth();
    let one = run("c1", &["--clients", "1", "--duration", "5"]);
    let eight = run("c8", &["--clients", "8", "--duration", "5"]);
    let (s1, s8) = (&one["searches"][0], &eight["searches"][0]);
    let (q1, q8) = (s1["qps"].as_f64().unwrap(), s8["qps"].as_f64().unwrap());
    let r8 = recall(s8);
    eprintln!(
        "20k ef=64: 1 client {q1:.0} qps cpu {:.0}%, 8 clients {q8:.0} qps cpu {:.0}%, recall {r8:.4}",
        num(s1, "cpu_pct"),
        num(s8, "cpu_pct")
    );
    assert!(q8 > q1, "8 clients {q8} qps <= 1 client {q1} qps");
    assert_eq!(num(s8, "errors"), 0.0);
    assert_eq!(num(s1, "errors"), 0.0);
    assert!(r8 >= 0.95, "first-pass recall {r8}");
    // (c) the load keys.
    for key in ["errors", "cpu_pct", "clients", "duration_s", "queries_done"] {
        num(s8, key);
    }
    assert_eq!(num(s8, "clients"), 8.0);
    assert_eq!(num(s8, "duration_s"), 5.0);
    let lat = s8["latency_ms"].as_array().unwrap().len() as f64;
    assert_eq!(lat, num(s8, "queries_done"));
    assert!((s8["qps"].as_f64().unwrap() * s8["total_s"].as_f64().unwrap() - lat).abs() < 1.0);
    assert_eq!(s8["ids"].as_array().unwrap().len(), 1000);
}

#[test]
fn inserts_during_search_match_static_build() {
    let _g = serial();
    truth();
    let stat = run("static", &[]);
    let r_static = recall(&stat["searches"][0]);
    let v = run(
        "ins",
        &["--clients", "4", "--duration", "10", "--insert-rate", "2000"],
    );
    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 2);
    let (load, after) = (&searches[0], &searches[1]);
    assert_eq!(after["search_params"]["phase"], "after_inserts");
    assert_eq!(after["search_params"]["ef"], 64);
    assert_eq!(num(load, "errors"), 0.0);
    assert_eq!(num(load, "insert_errors"), 0.0);
    assert_eq!(num(after, "inserted_rows"), 2000.0);
    assert_eq!(v["extra"]["inserted_rows"], 2000.0);
    assert!(num(after, "insert_p50_ms") > 0.0);
    assert_eq!(num(after, "build_rows"), 18000.0);
    assert_eq!(after["latency_ms"].as_array().unwrap().len(), 1000);
    let r_after = recall(after);
    let r_load = recall(load);
    eprintln!(
        "20k ef=64: static {r_static:.4}, during inserts (first pass) {r_load:.4}, after inserts {r_after:.4}, insert p50 {:.2} ms",
        num(after, "insert_p50_ms")
    );
    assert!((r_after - r_static).abs() <= 0.01, "{r_after} vs {r_static}");
}

#[test]
fn load_flags_rejected_for_other_indexes() {
    let o = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args(["--index", "flat", "--data", "data/processed/dev", "--out", "/tmp/x.json"])
        .args(["--clients", "4"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}
