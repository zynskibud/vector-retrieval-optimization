"""Shared k-means (CONTRACT 6.2), with the l2 / no-normalization mode for PQ codebooks (6.4.1)."""

import numpy as np

from . import splitmix


def default_train_size(n: int, k: int) -> int:
    return max(k, min(n, 256 * k))


def init_centers(train: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Pick k distinct training rows with next_below(train_n); redraw on a repeat."""
    n = len(train)
    if k > n:
        raise ValueError(f"k-means: k={k} is larger than the {n} training rows")
    rng = splitmix.new(seed)
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < k:
        i = splitmix.next_below(rng, n)
        if i not in seen:
            seen.add(i)
            chosen.append(i)
    return train[chosen].astype(np.float32, copy=True)


def assign(points: np.ndarray, centers: np.ndarray, metric: str = "ip") -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, fit) per point. fit is higher-is-better under the metric."""
    dots = points @ centers.T
    if metric == "l2":
        # -||p - c||^2 = 2 p.c - ||c||^2 - ||p||^2; the ||p||^2 term does not change the argmax.
        s = 2 * dots - np.einsum("ij,ij->i", centers, centers)[None, :]
    else:
        s = dots
    labels = np.argmax(s, axis=1)
    fit = s[np.arange(len(points)), labels]
    if metric == "l2":
        fit = fit - np.einsum("ij,ij->i", points, points)
    return labels, fit


def fix_empty(train: np.ndarray, centers: np.ndarray, labels: np.ndarray, fit: np.ndarray) -> None:
    """Move each empty center to the worst-fit point and move that point to it (in place)."""
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(fit, kind="stable")  # worst fit first
    used = 0
    for c in np.flatnonzero(counts == 0):
        # Take the next worst point whose old cluster does not become empty.
        while counts[labels[order[used]]] <= 1:
            used += 1
        p = order[used]
        used += 1
        counts[labels[p]] -= 1
        labels[p] = c
        counts[c] = 1
        centers[c] = train[p]


def update_centers(train: np.ndarray, labels: np.ndarray, k: int, normalize: bool) -> np.ndarray:
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    sums = np.zeros((k, train.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, train)
    centers = (sums / counts[:, None]).astype(np.float32)
    if normalize:
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def kmeans(
    points: np.ndarray,
    k: int,
    iters: int = 20,
    seed: int = 42,
    train_size: int | None = None,
    metric: str = "ip",
    normalize: bool = True,
) -> np.ndarray:
    """Return centers (k, d). Training set = the first train_size rows (not random)."""
    if train_size is None:
        train_size = default_train_size(len(points), k)
    train = np.ascontiguousarray(points[: max(k, train_size)], dtype=np.float32)
    centers = init_centers(train, k, seed)
    prev = None
    for _ in range(iters):
        labels, fit = assign(train, centers, metric)
        if prev is not None and np.array_equal(labels, prev):
            break
        prev = labels.copy()
        fix_empty(train, centers, labels, fit)
        centers = update_centers(train, labels, k, normalize)
    return centers
