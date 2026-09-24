import numpy as np
import pytest

from indexes.python.npy import read_npy

DEV = "data/processed/dev"


def test_queries_shape_and_values():
    q = read_npy(f"{DEV}/queries.npy")
    assert q.shape == (1000, 384)
    assert q.dtype == np.float32 and q.flags["C_CONTIGUOUS"]
    assert q[0, :3].tolist() == [-0.05522317439317703, -0.03818117082118988, 0.01416025310754776]


def test_int64_and_limit():
    gt = read_npy(f"{DEV}/ground_truth.npy", limit=5)
    assert gt.dtype == np.int64 and gt.shape == (5, 100)


def test_rejects_bad_descr(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="descr"):
        read_npy(path)
