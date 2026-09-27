"""The embedding model of the corpus, run on CPU (CONTRACT section 14.1).

The model files come from the hf-cache volume (downloaded once by the setup service), so
loading is offline: HF_HUB_OFFLINE=1 is set before sentence_transformers loads.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
BATCH = 32


class Embedder:
    """embed(texts) -> float32 (n, 384), L2-normalized, batch size 32, CPU."""

    def __init__(self, model_name: str = MODEL_NAME, model_version: str = "v1"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.model_name, self.model_version = model_name, model_version
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(list(texts), batch_size=BATCH, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return np.ascontiguousarray(v, dtype=np.float32).reshape(len(texts), DIM)


def check_against_corpus(data_dir, n: int = 100, embedder: Embedder | None = None) -> float:
    """Mean cosine between the embedded `text` of the first n metadata rows and their stored vectors."""
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    texts = pq.read_table(data_dir / "metadata.parquet", columns=["text"]).slice(0, n).column("text").to_pylist()
    stored = np.asarray(np.load(data_dir / "vectors.npy", mmap_mode="r")[:n], dtype=np.float32)
    got = (embedder or Embedder()).embed(texts)
    stored = stored / np.linalg.norm(stored, axis=1, keepdims=True)
    return float(np.mean(np.sum(got * stored, axis=1)))
