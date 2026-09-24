//! CONTRACT section 9, tests 1 and 2.

use bench::npy;
use bench::splitmix::SplitMix64;
use std::path::PathBuf;

fn dev_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/processed/dev")
}

#[test]
fn npy_reads_dev_queries() {
    let q = npy::read_f32(&dev_dir().join("queries.npy"), None).unwrap();
    assert_eq!((q.rows, q.cols), (1000, 384));
    // NumPy prints the float32 values widened to float64; widening is exact.
    let expected = [
        -0.05522317439317703f64,
        -0.03818117082118988,
        0.01416025310754776,
    ];
    let got: Vec<f64> = q.row(0)[..3].iter().map(|&x| x as f64).collect();
    assert_eq!(got, expected);
}

#[test]
fn npy_limit_reads_first_rows() {
    let v = npy::read_f32(&dev_dir().join("vectors.npy"), Some(5)).unwrap();
    assert_eq!((v.rows, v.cols, v.data.len()), (5, 384, 5 * 384));
}

#[test]
fn npy_rejects_wrong_descr() {
    let err = npy::read_i64(&dev_dir().join("queries.npy"), None).unwrap_err();
    assert!(err.contains("descr"), "{err}");
}

#[test]
fn splitmix_reference_values() {
    assert_eq!(SplitMix64::new(42).next_u64(), 13679457532755275413);
    assert_eq!(SplitMix64::new(0).next_u64(), 16294208416658607535);
}

#[test]
fn splitmix_f64_and_below_ranges() {
    let mut r = SplitMix64::new(7);
    for _ in 0..1000 {
        let f = r.next_f64();
        assert!((0.0..1.0).contains(&f));
        assert!(r.next_below(10) < 10);
    }
}
