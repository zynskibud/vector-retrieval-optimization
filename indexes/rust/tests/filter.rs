//! CONTRACT 11.5: metadata filtering for flat, ivf, and hnsw on the first 20,000 dev rows.
//! The filtered truth is brute force over the passing rows, computed here.

use bench::{flat, hnsw, ivf, npy, AnnIndex, FilterMasks, Matrix, ParamValue, Params};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

const N: usize = 20_000;
const K: usize = 10;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dev_dir() -> PathBuf {
    repo_root().join("data/processed/dev")
}

struct Data {
    vectors: Matrix,
    queries: Matrix,
    masks: FilterMasks,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| Data {
        vectors: npy::read_f32(&dev_dir().join("vectors.npy"), Some(N)).unwrap(),
        queries: npy::read_f32(&dev_dir().join("queries.npy"), None).unwrap(),
        masks: FilterMasks::new(dev_dir(), N),
    })
}

fn indexes() -> &'static Vec<(&'static str, Box<dyn AnnIndex>)> {
    static I: OnceLock<Vec<(&'static str, Box<dyn AnnIndex>)>> = OnceLock::new();
    I.get_or_init(|| {
        let v = &data().vectors;
        let dir = dev_dir();
        let dir = dir.to_str().unwrap();
        let mut f = flat::build(v.clone(), &Params::new(), 1, 42).unwrap();
        f.set_filter_dir(dir);
        let ip = ivf::build_defaults(N).with("nlist", ParamValue::Int(256));
        let mut i = ivf::build(v.clone(), &ip, 1, 42).unwrap();
        i.set_filter_dir(dir);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut h = pool
            .install(|| hnsw::build(v.clone(), &hnsw::build_defaults(N), threads, 42))
            .unwrap();
        h.set_filter_dir(dir);
        vec![
            ("flat", Box::new(f) as Box<dyn AnnIndex>),
            ("ivf", Box::new(i) as Box<dyn AnnIndex>),
            ("hnsw", Box::new(h) as Box<dyn AnnIndex>),
        ]
    })
}

fn defaults(index: &str, filter: &str) -> Params {
    let mut p = bench::search_defaults(index).unwrap();
    p.insert("filter", ParamValue::Str(filter.into()));
    p
}

/// Exact top-K among passing rows, by a plain scan (ties: lower ID first).
fn filtered_truth(pass: &[bool], query: &[f32]) -> Vec<i64> {
    let v = &data().vectors;
    let mut scored: Vec<(f32, i64)> = (0..v.rows)
        .filter(|&i| pass[i])
        .map(|i| {
            let s: f32 = query.iter().zip(v.row(i)).map(|(a, b)| a * b).sum();
            (s, i as i64)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(K).map(|(_, i)| i).collect()
}

#[test]
fn top10_recall_meets_floor_and_ids_pass() {
    let d = data();
    let mask = d.masks.get("top10").unwrap().unwrap();
    let truth: Vec<Vec<i64>> = (0..d.queries.rows)
        .map(|q| filtered_truth(&mask.pass, d.queries.row(q)))
        .collect();
    for (name, index) in indexes() {
        let p = defaults(name, "top10");
        let (mut hits, mut total) = (0usize, 0usize);
        for (q, t) in truth.iter().enumerate() {
            let r = index.search(d.queries.row(q), K, &p).unwrap();
            assert_eq!(r.counters["filter_rows"], mask.rows as f64);
            for &id in &r.ids {
                assert!(id == -1 || mask.pass[id as usize], "{name}: id {id} fails");
            }
            let t: HashSet<i64> = t.iter().copied().collect();
            hits += r.ids.iter().filter(|id| t.contains(id)).count();
            total += t.len();
        }
        let recall = hits as f64 / total as f64;
        let floor = match *name {
            "flat" => 1.0,
            "ivf" => 0.70,
            _ => 0.85,
        };
        eprintln!("{name} filter=top10 recall@10 = {recall:.4} (floor {floor})");
        assert!(recall >= floor, "{name}: recall {recall} < {floor}");
    }
}

#[test]
fn top01_returns_only_passing_ids() {
    let d = data();
    let mask = d.masks.get("top01").unwrap().unwrap();
    for (name, index) in indexes() {
        let p = defaults(name, "top01");
        for q in 0..d.queries.rows {
            let r = index.search(d.queries.row(q), K, &p).unwrap();
            assert_eq!(r.ids.len(), K);
            for &id in &r.ids {
                assert!(id == -1 || mask.pass[id as usize], "{name}: id {id} fails");
            }
        }
    }
}

#[test]
fn filter_none_is_unchanged() {
    let d = data();
    for (name, index) in indexes() {
        let with = defaults(name, "none");
        let mut without = with.clone();
        without.0.remove("filter");
        for q in 0..100 {
            let a = index.search(d.queries.row(q), K, &with).unwrap();
            let b = index.search(d.queries.row(q), K, &without).unwrap();
            assert_eq!(a.ids, b.ids, "{name}");
            assert_eq!(a.distance_computations, b.distance_computations, "{name}");
            assert!(!a.counters.contains_key("filter_rows"), "{name}");
        }
    }
    // flat with filter=none is the plain unfiltered scan.
    let (_, f) = &indexes()[0];
    let plain = flat::build(d.vectors.clone(), &Params::new(), 1, 42).unwrap();
    for q in 0..100 {
        let a = f.search(d.queries.row(q), K, &defaults("flat", "none")).unwrap();
        let b = flat::search(&plain, d.queries.row(q), K, &Params::new());
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.distance_computations, Some(N as u64));
    }
}

#[test]
fn bad_filter_name_is_an_error() {
    let d = data();
    for (name, index) in indexes() {
        let p = defaults(name, "top99");
        assert!(index.search(d.queries.row(0), K, &p).is_err(), "{name}");
    }
}

fn bench() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bench"));
    c.current_dir(repo_root());
    c
}

fn tmp_json(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rust-filter-{}-{name}.json", std::process::id()))
}

#[test]
fn bench_hnsw_writes_filter_fields() {
    let out = tmp_json("hnsw");
    let status = bench()
        .args(["--index", "hnsw", "--data", "data/processed/dev", "--limit", "20000"])
        .args(["--warmup", "10", "--search", "filter=none", "--search", "filter=top10"])
        .args(["--search", "filter=top01", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    let searches = v["searches"].as_array().unwrap();
    let names: Vec<&str> = searches
        .iter()
        .map(|s| s["search_params"]["filter"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["none", "top10", "top01"]);
    assert!(searches[0]["extra"].get("filter_rows").is_none());
    for s in &searches[1..] {
        assert!(s["extra"]["filter_rows"].as_f64().unwrap() > 0.0);
    }
    for s in searches {
        assert_eq!(s["search_params"]["ef"].as_i64(), Some(64));
        assert!(s["extra"]["visited"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn bench_exit_codes_for_filter() {
    let run = |index: &str, search: &str| {
        bench()
            .args(["--index", index, "--data", "data/processed/dev", "--limit", "2000"])
            .args(["--search", search, "--out"])
            .arg(tmp_json("never"))
            .status()
            .unwrap()
            .code()
    };
    assert_eq!(run("flat", "filter=top99"), Some(2));
    for index in ["pq", "ivf_pq", "diskann"] {
        assert_eq!(run(index, "filter=none"), Some(2), "{index}");
        assert_eq!(run(index, "filter=top10"), Some(2), "{index}");
    }
}
