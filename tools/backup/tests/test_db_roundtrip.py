"""Database round trip on 20,000 rows (CONTRACT section 15.4).

The round trip needs host steps for pgvector and Milvus (scripts/backup_db.sh), so this file
checks the outputs that `make backup-test DB=<db>` writes first to
results/raw/dev/bak-test/bak-<db>-<index>.json. A database without outputs is skipped.

restore_identical must be true when the restore needs no index build (Qdrant snapshot, Milvus
cold copy). pgvector's pg_restore builds the IVFFlat or HNSW index again, and pgvector seeds
neither its k-means sample nor its HNSW levels, so the rebuilt index differs: for a rebuild the
test requires instead that recall@10 after the restore is within RECALL_TOL of recall@10 before,
both against the exact top 10 over the 20,000 loaded rows (flat has no index and must still be
identical). ids_overlap is not the check: IVF with 1,024 lists on 20,000 rows has about 20 rows
per list, so a new k-means changes many results at nprobe=8 while the recall stays the same.
Run: make db-up DB=qdrant && make backup-test DB=qdrant; make db-down DB=qdrant
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tools.bench.runner import BACKUP_INDEXES
from tools.bench.schema import validate

FOLDER = Path("results/raw/dev/bak-test")
ROWS = 20000
RECALL_TOL = 0.02


def exact_top(k: int) -> np.ndarray:
    vecs = np.load("data/processed/dev/vectors.npy", mmap_mode="r")[:ROWS]
    queries = np.load("data/processed/dev/queries.npy")
    scores = queries @ np.ascontiguousarray(vecs).T
    return np.argsort(-scores, axis=1)[:, :k]


def recall(ids, truth) -> float:
    return float(np.mean([len(set(a) & set(t.tolist())) / len(t) for a, t in zip(ids, truth)]))


@pytest.mark.parametrize("db,index", [(db, i) for db, idx in BACKUP_INDEXES.items() for i in idx])
def test_round_trip(db, index):
    path = FOLDER / f"bak-{db}-{index}.json"
    if not path.exists():
        pytest.skip(f"{path} not found: run make backup-test DB={db} first")
    doc = json.loads(path.read_text())
    assert validate(doc) == []
    ex = doc["extra"]
    assert doc["n"] == ROWS
    assert ex["rows_before"] == ex["rows_after"] == ROWS
    assert [s["search_params"].get("phase", "") for s in doc["searches"]] == ["", "after_restore"]
    assert ex["backup_bytes"] > 0 and ex["backup_s"] > 0 and ex["restore_s"] > 0
    if ex["rebuild_needed"]:
        truth = exact_top(doc["k"])
        before, after = (recall(s["ids"], truth) for s in doc["searches"])
        assert abs(after - before) <= RECALL_TOL, f"recall@10 {before:.4f} -> {after:.4f}, ids_overlap {ex['ids_overlap']:.4f}"
    else:
        assert ex["restore_identical"], f"ids_equal_fraction = {ex['ids_equal_fraction']}"
