//! Dot product and top-k selection, including the tie rule (equal scores: lower ID first).

use bench::distance::{dot, TopK};

#[test]
fn dot_matches_naive_sum() {
    let a: Vec<f32> = (0..390).map(|i| (i as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..390).map(|i| (i as f32 * 0.11).cos()).collect();
    let naive: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    assert!((dot(&a, &b) as f64 - naive).abs() < 1e-4);
}

#[test]
fn topk_orders_best_first_and_breaks_ties_by_lower_id() {
    let mut t = TopK::new(3);
    for (id, s) in [(5, 0.5), (2, 0.9), (9, 0.9), (1, 0.1), (3, 0.9), (4, 0.7)] {
        t.push(id, s);
    }
    let (ids, scores) = t.into_sorted();
    assert_eq!(ids, [2, 3, 9]);
    assert_eq!(scores, [0.9, 0.9, 0.9]);
}
