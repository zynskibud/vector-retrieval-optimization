//! CONTRACT section 9, test 4 for ivf: recall floor on the dev set, CSR layout, counters,
//! and one real `bench` run with two search settings.

use bench::{ivf, npy, Matrix, ParamValue, Params};
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

struct Fixture {
    index: ivf::IvfIndex,
    n: usize,
    queries: Matrix,
    gt: npy::Array2<i64>,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let dir = dev_dir();
        let vectors = npy::read_f32(&dir.join("vectors.npy"), None).unwrap();
        let queries = npy::read_f32(&dir.join("queries.npy"), None).unwrap();
        let gt = npy::read_i64(&dir.join("ground_truth.npy"), None).unwrap();
        let n = vectors.rows;
        let index = ivf::build(vectors, &ivf::build_defaults(n), 1, 42).unwrap();
        Fixture {
            index,
            n,
            queries,
            gt,
        }
    })
}

fn recall(nprobe: i64) -> f64 {
    let f = fixture();
    let p = Params::new().with("nprobe", ParamValue::Int(nprobe));
    let nlist = f.index.nlist() as u64;
    let mut hits = 0usize;
    for i in 0..f.queries.rows {
        let r = ivf::search(&f.index, f.queries.row(i), 10, &p).unwrap();
        let dc = r.distance_computations.unwrap();
        assert!(dc >= nlist && dc <= nlist + f.n as u64, "bad count {dc}");
        assert!(r
            .ids
            .iter()
            .all(|&id| id == -1 || (0..f.n as i64).contains(&id)));
        assert!(r.scores.windows(2).all(|w| w[0] >= w[1]), "not best first");
        let truth: HashSet<i64> = f.gt.row(i)[..10].iter().copied().collect();
        hits += r.ids.iter().filter(|id| truth.contains(id)).count();
    }
    hits as f64 / (f.queries.rows * 10) as f64
}

#[test]
fn ivf_recall_meets_floor_and_grows_with_nprobe() {
    let r8 = recall(8);
    let r64 = recall(64);
    eprintln!("ivf dev recall@10: nprobe=8 {r8:.4}, nprobe=64 {r64:.4}");
    assert!(r8 >= 0.75, "recall@10 at nprobe=8 = {r8}");
    assert!(r64 >= r8);
}

#[test]
fn ivf_lists_are_valid_csr() {
    let f = fixture();
    let off = f.index.offsets();
    assert_eq!(off.len(), f.index.nlist() + 1);
    assert_eq!(off[0], 0);
    assert_eq!(*off.last().unwrap() as usize, f.n);
    assert!(off.windows(2).all(|w| w[0] <= w[1]));
    let mut seen = vec![false; f.n];
    for &id in f.index.list_ids() {
        assert!(!seen[id as usize], "row {id} appears twice");
        seen[id as usize] = true;
    }
    assert!(seen.iter().all(|&s| s));
    let nlist = f.index.nlist() as u64;
    assert_eq!(
        ivf::index_bytes(&f.index),
        nlist * 384 * 4 + f.n as u64 * 4 + (nlist + 1) * 4
    );
}

#[test]
fn ivf_bench_run_writes_two_searches() {
    let out = std::env::temp_dir().join(format!("rust-ivf-test-{}.json", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_bench"))
        .current_dir(repo_root())
        .args(["--index", "ivf", "--data", "data/processed/dev"])
        .args(["--limit", "20000", "--warmup", "10"])
        .args(["--search", "nprobe=4", "--search", "nprobe=16", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    assert_eq!(v["index"], "ivf");
    assert_eq!(v["n"], 20000);
    assert_eq!(v["extra"]["id_type"], "int32");
    let searches = v["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 2);
    for s in searches {
        let ids = s["ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1000);
        assert!(ids.iter().all(|r| r.as_array().unwrap().len() == 10));
        assert!(s["distance_computations"].as_f64().unwrap() >= 1024.0);
    }
}
