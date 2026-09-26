//! CONTRACT 13.5: deletes, updates, and compaction for flat, ivf, and hnsw on the
//! first 20,000 dev rows. The truth (all rows, and the remaining rows) is computed here.

use bench::{flat, hnsw, ivf, npy, AnnIndex, Matrix, ParamValue, Params};
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
    del30: Vec<bool>,
    upd_ids: Vec<i64>,
    upd_vecs: Vec<f32>,
    truth_all: Vec<Vec<i64>>,
    truth_live: Vec<Vec<i64>>,
}

/// Exact top-K over the rows with `keep(i)`, by a plain scan (ties: lower ID first).
fn truth(v: &Matrix, query: &[f32], keep: &dyn Fn(usize) -> bool) -> Vec<i64> {
    let mut scored: Vec<(f32, i64)> = (0..v.rows)
        .filter(|&i| keep(i))
        .map(|i| (bench::distance::dot(query, v.row(i)), i as i64))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(K).map(|(_, i)| i).collect()
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let vectors = npy::read_f32(&dev_dir().join("vectors.npy"), Some(N)).unwrap();
        let queries = npy::read_f32(&dev_dir().join("queries.npy"), None).unwrap();
        let del30 = npy::read_bool(&dev_dir().join("delete_del30.npy"), Some(N)).unwrap();
        let ids = npy::read_i64_1d(&dev_dir().join("update_upd10_ids.npy")).unwrap();
        let vecs = npy::read_f32(&dev_dir().join("update_upd10_vectors.npy"), None).unwrap();
        let keep: Vec<usize> = (0..ids.len()).filter(|&j| (ids[j] as usize) < N).collect();
        let upd_ids = keep.iter().map(|&j| ids[j]).collect();
        let upd_vecs = keep.iter().flat_map(|&j| vecs.row(j).to_vec()).collect();
        let truth_all = (0..queries.rows)
            .map(|q| truth(&vectors, queries.row(q), &|_| true))
            .collect();
        let truth_live = (0..queries.rows)
            .map(|q| truth(&vectors, queries.row(q), &|i| !del30[i]))
            .collect();
        Data {
            vectors,
            queries,
            del30,
            upd_ids,
            upd_vecs,
            truth_all,
            truth_live,
        }
    })
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Builds `name` on `v`: ivf with nlist = 256, hnsw at defaults with `threads`.
fn build(name: &str, v: &Matrix, threads: usize) -> Box<dyn AnnIndex> {
    match name {
        "flat" => Box::new(flat::build(v.clone(), &Params::new(), 1, 42).unwrap()),
        "ivf" => {
            let p = ivf::build_defaults(v.rows).with("nlist", ParamValue::Int(256));
            Box::new(ivf::build(v.clone(), &p, 1, 42).unwrap())
        }
        _ => Box::new(
            pool(threads)
                .install(|| hnsw::build(v.clone(), &hnsw::build_defaults(v.rows), threads, 42))
                .unwrap(),
        ),
    }
}

/// Recall@10 of `index` against `truth`. Panics if a returned ID has `bad(id)`.
fn recall(name: &str, index: &dyn AnnIndex, truth: &[Vec<i64>], bad: &dyn Fn(usize) -> bool) -> f64 {
    let d = data();
    let p = bench::search_defaults(name).unwrap();
    let (mut hits, mut total) = (0usize, 0usize);
    for (q, t) in truth.iter().enumerate() {
        let r = index.search(d.queries.row(q), K, &p).unwrap();
        for &id in &r.ids {
            assert!(id == -1 || !bad(id as usize), "{name}: deleted id {id} returned");
        }
        let t: HashSet<i64> = t.iter().copied().collect();
        hits += r.ids.iter().filter(|id| t.contains(id)).count();
        total += t.len();
    }
    hits as f64 / total as f64
}

const NAMES: [&str; 3] = ["flat", "ivf", "hnsw"];

#[test]
fn del30_returns_no_deleted_id_and_keeps_recall() {
    let d = data();
    for name in NAMES {
        let mut index = build(name, &d.vectors, cores());
        let before = recall(name, index.as_ref(), &d.truth_all, &|_| false);
        index.delete(&d.del30).unwrap();
        let after = recall(name, index.as_ref(), &d.truth_live, &|i| d.del30[i]);
        eprintln!("{name}: recall@10 undeleted {before:.4}, after del30 {after:.4}");
        if name == "flat" {
            assert_eq!(after, 1.0);
        }
        assert!((after - before).abs() <= 0.03, "{name}: {before} -> {after}");
    }
}

#[test]
fn upd10_query_equal_to_new_vector_returns_the_row() {
    let d = data();
    let dim = d.vectors.cols;
    assert!(d.upd_ids.len() >= 100);
    let step = d.upd_ids.len() / 100;
    for name in NAMES {
        let mut index = build(name, &d.vectors, cores());
        index.update(&d.upd_ids, &d.upd_vecs).unwrap();
        let p = bench::search_defaults(name).unwrap();
        for s in 0..100 {
            let j = s * step;
            let q = &d.upd_vecs[j * dim..(j + 1) * dim];
            let r = index.search(q, K, &p).unwrap();
            assert_eq!(r.ids[0], d.upd_ids[j], "{name}: updated row not top-1");
        }
    }
}

#[test]
fn compact_matches_a_fresh_build_and_shrinks() {
    let d = data();
    let live: Vec<usize> = (0..N).filter(|&i| !d.del30[i]).collect();
    let mut data_live = Vec::with_capacity(live.len() * d.vectors.cols);
    for &i in &live {
        data_live.extend_from_slice(d.vectors.row(i));
    }
    let vlive = Matrix {
        data: data_live,
        rows: live.len(),
        cols: d.vectors.cols,
    };
    // Truth over the remaining rows, in positions of `vlive`.
    let pos: Vec<usize> = {
        let mut p = vec![usize::MAX; N];
        live.iter().enumerate().for_each(|(j, &i)| p[i] = j);
        p
    };
    let truth_pos: Vec<Vec<i64>> = d
        .truth_live
        .iter()
        .map(|t| t.iter().map(|&i| pos[i as usize] as i64).collect())
        .collect();
    // One thread for hnsw, so the rebuild and the fresh build are the same graph.
    for name in NAMES {
        let mut index = build(name, &d.vectors, 1);
        index.delete(&d.del30).unwrap();
        let bytes_before = index.index_bytes();
        pool(1).install(|| index.compact()).unwrap();
        let bytes_after = index.index_bytes();
        let compacted = recall(name, index.as_ref(), &d.truth_live, &|i| d.del30[i]);
        let fresh = build(name, &vlive, 1);
        let fresh_recall = recall(name, fresh.as_ref(), &truth_pos, &|_| false);
        eprintln!(
            "{name}: compacted recall {compacted:.4}, fresh {fresh_recall:.4}, bytes {bytes_before} -> {bytes_after}"
        );
        assert!((compacted - fresh_recall).abs() <= 0.01, "{name}");
        if name != "hnsw" {
            assert!(bytes_after < bytes_before, "{name}: {bytes_before} -> {bytes_after}");
        }
    }
}

#[test]
fn hnsw_repair_mode_keeps_recall() {
    let d = data();
    let mut h = pool(cores())
        .install(|| hnsw::build(d.vectors.clone(), &hnsw::build_defaults(N), cores(), 42))
        .unwrap();
    let before = recall("hnsw", &h, &d.truth_all, &|_| false);
    h.delete(&d.del30).unwrap();
    let bytes = AnnIndex::index_bytes(&h);
    h.compact_repair().unwrap();
    let after = recall("hnsw", &h, &d.truth_live, &|i| d.del30[i]);
    eprintln!(
        "hnsw repair mode: recall {before:.4} -> {after:.4}, bytes {bytes} -> {}",
        AnnIndex::index_bytes(&h)
    );
    assert!((after - before).abs() <= 0.03);
    assert!(AnnIndex::index_bytes(&h) < bytes);
}

fn bench_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bench"));
    c.current_dir(repo_root());
    c
}

fn tmp_json(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rust-changes-{}-{name}.json", std::process::id()))
}

#[test]
fn bench_hnsw_delete_writes_change_fields() {
    let out = tmp_json("hnsw");
    let status = bench_cmd()
        .args(["--index", "hnsw", "--data", "data/processed/dev", "--limit", "20000"])
        .args(["--warmup", "10", "--delete", "del30", "--search", "ef=64", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    let sp = &v["searches"][0]["search_params"];
    assert_eq!(sp["deleted"].as_str(), Some("del30"));
    assert_eq!(sp["compacted"].as_i64(), Some(0));
    assert!(sp.get("updated").is_none());
    let deleted = data().del30.iter().filter(|&&x| x).count();
    assert_eq!(v["extra"]["deleted_rows"].as_u64(), Some(deleted as u64));
    assert!(v["extra"]["delete_s"].as_f64().is_some());
}

#[test]
fn plain_run_has_no_change_fields_and_bad_flags_exit_2() {
    let out = tmp_json("plain");
    let status = bench_cmd()
        .args(["--index", "flat", "--data", "data/processed/dev", "--limit", "2000"])
        .args(["--warmup", "10", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(&out).ok();
    let sp = v["searches"][0]["search_params"].as_object().unwrap();
    for key in ["deleted", "updated", "compacted"] {
        assert!(!sp.contains_key(key), "{key}");
    }
    let run = |args: &[&str]| {
        bench_cmd()
            .args(["--data", "data/processed/dev", "--limit", "2000", "--out"])
            .arg(tmp_json("never"))
            .args(args)
            .status()
            .unwrap()
            .code()
    };
    for index in ["pq", "ivf_pq", "diskann"] {
        assert_eq!(run(&["--index", index, "--delete", "del10"]), Some(2), "{index}");
    }
    assert_eq!(run(&["--index", "flat", "--delete", "del99"]), Some(2));
    assert_eq!(run(&["--index", "flat", "--compact"]), Some(2));
    assert_eq!(run(&["--index", "flat", "--delete", "del10", "--update", "upd10"]), Some(2));
    assert_eq!(
        run(&["--index", "ivf", "--delete", "del10", "--compact", "--compact-mode", "repair"]),
        Some(2)
    );
}
