import numpy as np

from indexes.python import flat
from indexes.python.npy import read_npy

DEV = "data/processed/dev"


def test_flat_recall_and_top1():
    vectors = read_npy(f"{DEV}/vectors.npy")
    queries = read_npy(f"{DEV}/queries.npy")
    gt = read_npy(f"{DEV}/ground_truth.npy")[:, :10]
    index = flat.build(vectors, {}, 1, 42)
    ids = np.stack([flat.search(index, q, 10, {})[0] for q in queries])
    assert index["distance_computations"] == len(vectors)
    hits = sum(len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt.tolist()))
    assert hits / gt.size == 1.0
    assert (ids[:, 0] == gt[:, 0]).all()
