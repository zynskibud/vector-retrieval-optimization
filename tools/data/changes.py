"""Make the Phase 5 change sets and their ground truth (CONTRACT section 13.1).

Nested random delete sets (10%, 30%, 50% of the rows) with the exact top-100
among the remaining rows, and an update set (10% of the rows) with new vectors
and the exact top-100 over the corpus with those rows replaced.

Run: uv run python -m tools.data.changes [--data data/processed] [--seed 7]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.data.ground_truth import exact_topk

DELETES = {"del10": 0.1, "del30": 0.3, "del50": 0.5}
UPDATE_FRACTION = 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    vectors = np.load(args.data / "vectors.npy", mmap_mode="r")
    queries = np.load(args.data / "queries.npy")
    n = len(vectors)
    rng = np.random.default_rng(args.seed)
    info = {"seed": args.seed, "rows": n}

    # Nested delete sets: one random order, the first f*N rows of it are deleted.
    order = rng.permutation(n)
    for name, f in DELETES.items():
        deleted = np.zeros(n, dtype=bool)
        deleted[order[: int(f * n)]] = True
        np.save(args.data / f"delete_{name}.npy", deleted)
        keep = np.flatnonzero(~deleted)
        ids, scores = exact_topk(np.asarray(vectors[keep]), queries, k=args.k, progress=False)
        np.save(args.data / f"ground_truth_{name}.npy", keep[ids].astype(np.int64))
        np.save(args.data / f"ground_truth_{name}_scores.npy", scores)
        info[name] = {"deleted_rows": int(deleted.sum())}
        print(f"{name}: {int(deleted.sum()):,} of {n:,} rows deleted")

    # Update set: new vector = normalize(0.6 * old + 0.8 * random unit vector).
    upd = np.sort(rng.choice(n, size=int(UPDATE_FRACTION * n), replace=False))
    r = rng.standard_normal((len(upd), vectors.shape[1])).astype(np.float32)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    new = 0.6 * np.asarray(vectors[upd]) + 0.8 * r
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    np.save(args.data / "update_upd10_ids.npy", upd.astype(np.int64))
    np.save(args.data / "update_upd10_vectors.npy", new.astype(np.float32))
    updated = np.array(vectors)  # one full copy: 1.8 GB on the full corpus
    updated[upd] = new
    ids, scores = exact_topk(updated, queries, k=args.k, progress=False)
    np.save(args.data / "ground_truth_upd10.npy", ids)
    np.save(args.data / "ground_truth_upd10_scores.npy", scores)
    info["upd10"] = {"updated_rows": int(len(upd))}
    print(f"upd10: {len(upd):,} rows get new vectors")
    (args.data / "changes.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
