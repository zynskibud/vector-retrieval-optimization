//! CONTRACT section 9, test 3, and the flat part of test 4: exact search on the dev set.

use bench::{flat, npy, Params};
use std::collections::HashSet;
use std::path::PathBuf;

fn dev_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/processed/dev")
}

#[test]
fn flat_recall_is_one_and_top1_matches() {
    let dir = dev_dir();
    let vectors = npy::read_f32(&dir.join("vectors.npy"), None).unwrap();
    let queries = npy::read_f32(&dir.join("queries.npy"), None).unwrap();
    let gt = npy::read_i64(&dir.join("ground_truth.npy"), None).unwrap();
    let n = vectors.rows as u64;
    let index = flat::build(vectors, &Params::new(), 1, 42).unwrap();

    let mut hits = 0usize;
    for i in 0..queries.rows {
        let r = flat::search(&index, queries.row(i), 10, &Params::new());
        assert_eq!(r.distance_computations, Some(n));
        let truth: HashSet<i64> = gt.row(i)[..10].iter().copied().collect();
        hits += r.ids.iter().filter(|id| truth.contains(id)).count();
        assert_eq!(r.ids[0], gt.row(i)[0], "top-1 differs for query {i}");
        assert!(r.scores.windows(2).all(|w| w[0] >= w[1]), "not best first");
    }
    let recall = hits as f64 / (queries.rows * 10) as f64;
    assert_eq!(recall, 1.0);
}

#[test]
fn flat_pads_short_results() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(3)).unwrap();
    let q = v.row(0).to_vec();
    let index = flat::build(v, &Params::new(), 1, 42).unwrap();
    let r = flat::search(&index, &q, 5, &Params::new());
    assert_eq!(r.ids[0], 0);
    assert_eq!(&r.ids[3..], &[-1, -1]);
    assert!(r.scores[4] == f32::NEG_INFINITY);
}
