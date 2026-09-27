from pathlib import Path

import numpy as np
import pytest

DEV = Path("data/processed/dev")


@pytest.fixture(scope="module")
def embedder():
    from tools.cache.embed import Embedder

    return Embedder()


def test_embed_shape_and_norm(embedder):
    v = embedder.embed(["a short text", "another one"])
    assert v.shape == (2, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not (DEV / "metadata.parquet").exists(), reason="dev data missing")
def test_model_matches_corpus(embedder):
    from tools.cache.embed import check_against_corpus

    cos = check_against_corpus(DEV, 100, embedder)
    print(f"mean cosine over 100 corpus rows: {cos:.5f}")
    assert cos >= 0.99
