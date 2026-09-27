//! CONTRACT 15.4: save and load of the `.vro` index file for flat, ivf, hnsw on the
//! first 20,000 dev rows. A loaded index must return the same IDs and scores as the
//! index that wrote the file: at the default setting, after `del30`, with a filter,
//! and after a compaction. The header must parse, and a wrong dim, index, magic, or
//! build parameter must be refused. The bench binary: `--save` then `--load` in a
//! second process gives identical IDs.

use bench::{flat, hnsw, ivf, npy, vro, AnnIndex, Matrix, ParamValue, Params};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const N: usize = 20_000;
const K: usize = 10;
const NAMES: [&str; 3] = ["flat", "ivf", "hnsw"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dev_dir() -> PathBuf {
    repo_root().join("data/processed/dev")
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rust-vro-test-{}-{name}", std::process::id()))
}

struct Data {
    vectors: Matrix,
    queries: Matrix,
    del30: Vec<bool>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| Data {
        vectors: npy::read_f32(&dev_dir().join("vectors.npy"), Some(N)).unwrap(),
        queries: npy::read_f32(&dev_dir().join("queries.npy"), None).unwrap(),
        del30: npy::read_bool(&dev_dir().join("delete_del30.npy"), Some(N)).unwrap(),
    })
}

fn build_params(name: &str) -> Params {
    match name {
        "ivf" => ivf::build_defaults(N).with("nlist", ParamValue::Int(256)),
        other => bench::build_defaults(other, N).unwrap(),
    }
}

fn build(name: &str) -> Box<dyn AnnIndex> {
    let v = data().vectors.clone();
    let p = build_params(name);
    let mut idx: Box<dyn AnnIndex> = match name {
        "flat" => Box::new(flat::build(v, &p, 1, 42).unwrap()),
        "ivf" => Box::new(ivf::build(v, &p, 1, 42).unwrap()),
        _ => Box::new(hnsw::build(v, &p, 1, 42).unwrap()),
    };
    idx.set_filter_dir(dev_dir().to_str().unwrap());
    idx
}

fn save_load(name: &str, idx: &dyn AnnIndex, tag: &str) -> Box<dyn AnnIndex> {
    let path = tmp(&format!("{name}-{tag}.vro"));
    let bytes = idx.save(&path, &build_params(name), 42).unwrap();
    assert_eq!(bytes, std::fs::metadata(&path).unwrap().len());
    let (mut loaded, h) = bench::load(name, &path, &Params::new(), Some(384), 1).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!((h.n, h.dim, h.index.as_str()), (N, 384, name));
    loaded.set_filter_dir(dev_dir().to_str().unwrap());
    loaded
}

/// Asserts equal IDs and bit-identical scores for all 1,000 queries.
fn assert_same(name: &str, a: &dyn AnnIndex, b: &dyn AnnIndex, params: &Params, what: &str) {
    let d = data();
    for q in 0..d.queries.rows {
        let ra = a.search(d.queries.row(q), K, params).unwrap();
        let rb = b.search(d.queries.row(q), K, params).unwrap();
        assert_eq!(ra.ids, rb.ids, "{name} {what}: query {q} ids differ");
        let bits = |s: &[f32]| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&ra.scores), bits(&rb.scores), "{name} {what}: query {q} scores differ");
        assert_eq!(ra.distance_computations, rb.distance_computations, "{name} {what}: query {q}");
    }
}

fn with_filter(name: &str, f: &str) -> Params {
    let mut p = bench::search_defaults(name).unwrap();
    p.insert("filter", ParamValue::Str(f.into()));
    p
}

#[test]
fn round_trip_default_filter_and_delete() {
    for name in NAMES {
        let mut idx = build(name);
        let defaults = bench::search_defaults(name).unwrap();
        let mut loaded = save_load(name, idx.as_ref(), "plain");
        assert_same(name, idx.as_ref(), loaded.as_ref(), &defaults, "default");
        assert_same(name, idx.as_ref(), loaded.as_ref(), &with_filter(name, "top10"), "filter=top10");

        // Phase 5 on the loaded index: the same delete gives the same results.
        idx.delete(&data().del30).unwrap();
        loaded.delete(&data().del30).unwrap();
        assert_same(name, idx.as_ref(), loaded.as_ref(), &defaults, "delete after load");

        // A file written after del30 carries the tombstones section.
        let reloaded = save_load(name, idx.as_ref(), "del30");
        assert_same(name, idx.as_ref(), reloaded.as_ref(), &defaults, "del30");
        assert_same(name, idx.as_ref(), reloaded.as_ref(), &with_filter(name, "top10"), "del30 + filter");
        let d = data();
        for q in 0..100 {
            for id in reloaded.search(d.queries.row(q), K, &defaults).unwrap().ids {
                assert!(id < 0 || !d.del30[id as usize], "{name}: deleted id {id} returned");
            }
        }
    }
}

#[test]
fn round_trip_after_compaction() {
    for name in NAMES {
        let mut idx = build(name);
        idx.delete(&data().del30).unwrap();
        idx.compact().unwrap();
        let loaded = save_load(name, idx.as_ref(), "compact");
        let defaults = bench::search_defaults(name).unwrap();
        assert_same(name, idx.as_ref(), loaded.as_ref(), &defaults, "compacted");
    }
}

#[test]
fn round_trip_after_hnsw_repair_mode() {
    let mut idx = build("hnsw");
    idx.delete(&data().del30).unwrap();
    idx.compact_repair().unwrap();
    let loaded = save_load("hnsw", idx.as_ref(), "repair");
    assert_same("hnsw", idx.as_ref(), loaded.as_ref(), &bench::search_defaults("hnsw").unwrap(), "repair");
}

fn expected_sections(name: &str) -> Vec<&'static str> {
    let mut s = vec!["vectors", "tombstones"];
    match name {
        "ivf" => s.extend(["centers", "list_ids", "list_offsets"]),
        "hnsw" => s.extend([
            "levels",
            "entry",
            "layer0_slots",
            "layer0_counts",
            "upper_slots",
            "upper_counts",
            "upper_offsets",
        ]),
        _ => {}
    }
    s
}

#[test]
fn header_parses_and_sections_are_aligned() {
    for name in NAMES {
        let idx = build(name);
        let path = tmp(&format!("{name}-header.vro"));
        let file_bytes = idx.save(&path, &build_params(name), 42).unwrap();
        let raw = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(&raw[..8], b"VROIDX01");
        let hlen = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let text = std::str::from_utf8(&raw[12..12 + hlen]).unwrap();
        assert!(text.is_ascii());
        let v: Value = serde_json::from_str(text).unwrap();
        for key in ["index", "n", "dim", "build_params", "seed", "contract_version", "language", "sections"] {
            assert!(v.get(key).is_some(), "{name}: header has no {key}");
        }
        assert_eq!(v["index"], name);
        assert_eq!(v["n"], N);
        assert_eq!(v["dim"], 384);
        assert_eq!(v["seed"], 42);
        assert_eq!(v["contract_version"], 1);
        assert_eq!(v["language"], "rust");
        assert_eq!(vro::params_from_json(&v["build_params"]).unwrap(), build_params(name));

        let secs = v["sections"].as_array().unwrap();
        let names: Vec<&str> = secs.iter().map(|s| s["name"].as_str().unwrap()).collect();
        assert_eq!(names, expected_sections(name), "{name}: section names");
        let first = secs[0]["offset"].as_u64().unwrap() as usize;
        assert_eq!(first, (12 + hlen).div_ceil(64) * 64, "{name}: padding to the next multiple of 64");
        assert!(raw[12 + hlen..first].iter().all(|&b| b == 0));
        let mut end = (12 + hlen) as u64;
        for s in secs {
            let off = s["offset"].as_u64().unwrap();
            let bytes = s["bytes"].as_u64().unwrap();
            let size = if s["dtype"] == "u8" { 1 } else { 4 };
            let count: u64 = s["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).product();
            assert_eq!(off % 64, 0, "{name}: {} offset", s["name"]);
            assert!(off >= end && off - end < 64, "{name}: {} placement", s["name"]);
            assert_eq!(bytes, count * size, "{name}: {} bytes", s["name"]);
            end = off + bytes;
        }
        assert_eq!(end, file_bytes);
        assert_eq!(raw.len() as u64, file_bytes);

        // Shapes and dtypes of CONTRACT 15.1.
        let h = vro::read_header(&{
            let p = tmp(&format!("{name}-header2.vro"));
            std::fs::write(&p, &raw).unwrap();
            p
        })
        .unwrap();
        std::fs::remove_file(tmp(&format!("{name}-header2.vro"))).ok();
        let sec = |n: &str| h.section(n).unwrap().clone();
        assert_eq!((sec("vectors").dtype.as_str(), sec("vectors").shape.clone()), ("f32", vec![N, 384]));
        assert_eq!((sec("tombstones").dtype.as_str(), sec("tombstones").shape.clone()), ("u8", vec![N / 8]));
        // The first vector, raw little-endian f32 at the vectors offset.
        let o = sec("vectors").offset as usize;
        let x0 = f32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        assert_eq!(x0.to_bits(), data().vectors.data[0].to_bits());
        if name == "ivf" {
            assert_eq!(sec("centers").shape, vec![256, 384]);
            assert_eq!((sec("list_ids").dtype.as_str(), sec("list_ids").shape.clone()), ("int32", vec![N]));
            assert_eq!(sec("list_offsets").shape, vec![257]);
        }
        if name == "hnsw" {
            let big_l = sec("upper_counts").shape[0];
            assert_eq!(sec("levels").shape, vec![N]);
            assert_eq!(sec("entry").shape, vec![1]);
            assert_eq!(sec("layer0_slots").shape, vec![N, 32]);
            assert_eq!(sec("layer0_counts").shape, vec![N]);
            assert_eq!(sec("upper_slots").shape, vec![big_l, 16]);
            assert_eq!(sec("upper_offsets").shape, vec![N + 1]);
            let lo = sec("levels").offset as usize;
            let sum: usize = raw[lo..lo + N].iter().map(|&l| l as usize).sum();
            assert_eq!(sum, big_l, "L = sum of levels");
            let uo = sec("upper_offsets").offset as usize + 4 * N;
            assert_eq!(i32::from_le_bytes(raw[uo..uo + 4].try_into().unwrap()) as usize, big_l);
        }
    }
}

/// Writes `raw` with `from` replaced by `to` (same length) in the header, and tries a load.
fn load_patched(name: &str, raw: &[u8], from: &str, to: &str) -> Result<(), String> {
    let hlen = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&raw[12..12 + hlen]).unwrap();
    assert!(text.contains(from), "{name}: header has no {from}");
    assert_eq!(from.len(), to.len());
    let mut bad = raw.to_vec();
    bad[12..12 + hlen].copy_from_slice(text.replacen(from, to, 1).as_bytes());
    let p = tmp(&format!("{name}-patched.vro"));
    std::fs::write(&p, &bad).unwrap();
    let r = bench::load(name, &p, &Params::new(), None, 1).map(|_| ());
    std::fs::remove_file(&p).ok();
    r
}

#[test]
fn wrong_files_are_refused() {
    for name in NAMES {
        let idx = build(name);
        let path = tmp(&format!("{name}-bad.vro"));
        idx.save(&path, &build_params(name), 42).unwrap();
        let raw = std::fs::read(&path).unwrap();

        // dim changed in the header: the vectors shape no longer matches.
        let e = load_patched(name, &raw, "\"dim\":384", "\"dim\":385").unwrap_err();
        assert!(e.contains("shape") || e.contains("dim"), "{name}: {e}");
        // dim that the caller does not expect.
        assert!(bench::load(name, &path, &Params::new(), Some(128), 1).is_err());
        // Another index.
        let other = if name == "flat" { "ivf" } else { "flat" };
        assert!(load_err(other, &path, &Params::new()).contains("index"));
        // A build parameter that conflicts with the header.
        if name != "flat" {
            let (k, v) = if name == "ivf" { ("nlist", 128) } else { ("m", 8) };
            let p = Params::new().with(k, ParamValue::Int(v));
            assert!(load_err(name, &path, &p).contains(k));
            // The same value is accepted.
            let same = build_params(name);
            assert!(bench::load(name, &path, &same, None, 1).is_ok());
        }
        // Wrong magic.
        let mut bad = raw.clone();
        bad[7] = b'2';
        let p = tmp(&format!("{name}-magic.vro"));
        std::fs::write(&p, &bad).unwrap();
        assert!(load_err(name, &p, &Params::new()).contains("magic"));
        // A truncated file.
        std::fs::write(&p, &raw[..raw.len() - 1]).unwrap();
        assert!(bench::load(name, &p, &Params::new(), None, 1).is_err());
        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&path).ok();
    }
}

/// The error of a load that must fail.
fn load_err(name: &str, path: &Path, p: &Params) -> String {
    match bench::load(name, path, p, None, 1) {
        Ok(_) => panic!("{name}: load of {} must fail", path.display()),
        Err(e) => e,
    }
}

fn run_bench(name: &str, extra: &[&str], out: &Path) -> std::process::ExitStatus {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bench"));
    c.current_dir(repo_root())
        .args(["--index", name, "--data", "data/processed/dev", "--limit", "20000"])
        .args(["--warmup", "10", "--out"])
        .arg(out)
        .args(extra);
    c.status().unwrap()
}

#[test]
fn bench_save_then_load_in_a_second_process() {
    for name in NAMES {
        let file = tmp(&format!("{name}-bench.vro"));
        let file_s = file.to_str().unwrap();
        let out1 = tmp(&format!("{name}-save.json"));
        let out2 = tmp(&format!("{name}-load.json"));
        let mut build_args = vec!["--save", file_s];
        if name == "ivf" {
            build_args.extend(["--build", "nlist=256"]);
        }
        assert!(run_bench(name, &build_args, &out1).success(), "{name}: save run");
        assert!(run_bench(name, &["--load", file_s], &out2).success(), "{name}: load run");
        let read = |p: &Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
        let (a, b) = (read(&out1), read(&out2));
        assert_eq!(a["searches"][0]["ids"], b["searches"][0]["ids"], "{name}: ids differ");
        assert_eq!(a["searches"][0]["scores"], b["searches"][0]["scores"], "{name}: scores differ");
        assert_eq!(a["build_params"], b["build_params"], "{name}: build_params");
        assert_eq!(a["n"], b["n"]);
        assert_eq!(b["build"]["train_s"], 0.0);
        assert_eq!(b["build"]["add_s"], 0.0);
        assert!(a["extra"]["save_s"].as_f64().is_some());
        assert_eq!(a["extra"]["file_bytes"].as_u64().unwrap(), std::fs::metadata(&file).unwrap().len());
        assert!(b["extra"]["load_s"].as_f64().is_some());
        assert_eq!(b["extra"]["loaded_from"], file_s);

        // A --build flag that conflicts with the header exits 2; a matching one runs.
        if name != "flat" {
            let bad = if name == "ivf" { "nlist=128" } else { "m=8" };
            let st = run_bench(name, &["--load", file_s, "--build", bad], &out2);
            assert_eq!(st.code(), Some(2), "{name}: conflicting --build");
        }
        // Loading with the wrong --index exits 2.
        let other = if name == "flat" { "hnsw" } else { "flat" };
        assert_eq!(run_bench(other, &["--load", file_s], &out2).code(), Some(2));
        for p in [&file, &out1, &out2] {
            std::fs::remove_file(p).ok();
        }
    }
}
