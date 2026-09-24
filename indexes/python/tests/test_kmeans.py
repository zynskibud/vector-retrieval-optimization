import numpy as np

from indexes.python import kmeans


def test_kmeans_shapes_and_norm():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    c = kmeans.kmeans(x, 8, iters=10, seed=42)
    assert c.shape == (8, 16)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(c, kmeans.kmeans(x, 8, iters=10, seed=42))
    c2 = kmeans.kmeans(x, 8, iters=10, seed=42, metric="l2", normalize=False)
    assert c2.shape == (8, 16)
