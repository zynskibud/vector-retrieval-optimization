"""Recall, latency percentiles, and QPS for bench output documents.

Recall is computed here only, never inside an index program (CLAUDE.md).
"""

import numpy as np


def recall_at_k(ids, ground_truth, k: int) -> tuple[np.ndarray, float]:
    """Return (per-query recall, mean recall). -1 in ids means no result and never matches."""
    ids = np.asarray(ids, dtype=np.int64)[:, :k]
    gt = np.asarray(ground_truth, dtype=np.int64)[: len(ids), :k]
    hits = np.array([len(set(r[r >= 0].tolist()) & set(g.tolist())) for r, g in zip(ids, gt)])
    per_query = hits / k
    return per_query, float(per_query.mean())


def latency_stats(latency_ms) -> dict:
    """p50, p90, p99, and mean of per-query latencies, in ms."""
    lat = np.asarray(latency_ms, dtype=np.float64)
    p50, p90, p99 = np.percentile(lat, [50, 90, 99])
    return {"p50_ms": float(p50), "p90_ms": float(p90), "p99_ms": float(p99), "mean_ms": float(lat.mean())}


def qps(latency_ms) -> float:
    """Queries per second from the sum of per-query latencies (one thread, one query at a time)."""
    return 1000.0 * len(latency_ms) / float(np.sum(latency_ms))


def summarize(doc: dict, ground_truth, k: int = 10) -> list[dict]:
    """One row per search setting in doc. k is the recall cutoff (at most doc['k'])."""
    k = min(k, doc["k"])
    rows = []
    for run in doc["searches"]:
        _, recall = recall_at_k(run["ids"], ground_truth, k)
        lat = latency_stats(run["latency_ms"])
        rows.append({
            "language": doc["language"], "index": doc["index"], "n": doc["n"],
            "build_params": doc["build_params"], "search_params": run["search_params"],
            f"recall@{k}": recall, "p50_ms": lat["p50_ms"], "p90_ms": lat["p90_ms"], "p99_ms": lat["p99_ms"],
            "mean_ms": lat["mean_ms"], "qps": run["qps"], "build_s": doc["build"]["total_s"],
            "peak_rss_mb": doc["build"]["peak_rss_mb"], "index_bytes": doc["build"]["index_bytes"],
            "distance_computations": run["distance_computations"],
            # Spread of the runner's repeat runs (mean p50 over the search settings per run).
            "p50_spread": _spread(doc.get("extra", {}).get("runner", {}).get("p50_ms_runs")),
        })
    return rows


def _spread(p50_runs) -> str:
    """'min-max' of the repeat runs' p50, or '' when the runner did not repeat."""
    if not p50_runs or len(p50_runs) < 2:
        return ""
    return f"{min(p50_runs):.3g}-{max(p50_runs):.3g}"
