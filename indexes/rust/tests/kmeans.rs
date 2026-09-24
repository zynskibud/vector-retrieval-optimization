//! Sanity checks of the shared k-means (CONTRACT 6.2) on a small dev slice.

use bench::kmeans::{kmeans, KmeansOptions};
use bench::npy;
use std::path::PathBuf;

#[test]
fn kmeans_is_deterministic_and_normalized() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/processed/dev");
    let v = npy::read_f32(&dir.join("vectors.npy"), Some(4000)).unwrap();
    let a = kmeans(&v.data, v.cols, 16, 10, 42, KmeansOptions::default()).unwrap();
    let b = kmeans(&v.data, v.cols, 16, 10, 42, KmeansOptions::default()).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 16 * 384);
    for c in a.chunks_exact(384) {
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
    }
}
