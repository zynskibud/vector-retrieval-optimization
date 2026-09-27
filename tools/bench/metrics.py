"""Recall, latency percentiles, and QPS for bench output documents.

Recall is computed here only, never inside an index program (CLAUDE.md).
"""

import json
from pathlib import Path

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


def filter_of(run: dict) -> str:
    """The filter name of one search run (CONTRACT section 11.2); "none" when absent."""
    return str(run["search_params"].get("filter", "none"))


CHANGE_SETS = ("del10", "del30", "del50", "upd10")  # CONTRACT section 13.1
DELETE_FRACTION = {"del10": 0.1, "del30": 0.3, "del50": 0.5}


def truth_key(run: dict) -> str:
    """The truth a search run is scored against: its search_params.deleted or .updated
    (CONTRACT section 13.2), else its filter (section 11.2)."""
    sp = run["search_params"]
    change = sp.get("deleted") or sp.get("updated")
    if change:
        if filter_of(run) != "none":
            raise ValueError(f"run combines {change} with filter={filter_of(run)}; no truth exists for that")
        return str(change)
    return filter_of(run)


def load_truths(data_dir) -> dict:
    """{name: ground-truth array} for every truth file in data_dir.

    "none" is ground_truth.npy; each name in filters.json and each change set
    (del10, del30, del50, upd10) is ground_truth_<name>.npy.
    """
    data_dir = Path(data_dir)
    truths = {"none": np.load(data_dir / "ground_truth.npy")}
    for name in [*load_filters(data_dir), *CHANGE_SETS]:
        path = data_dir / f"ground_truth_{name}.npy"
        if path.exists():
            truths[name] = np.load(path)
    return truths


def load_delete_masks(data_dir) -> dict:
    """{del10|del30|del50: bool array (N,)} for the delete sets present in data_dir."""
    data_dir = Path(data_dir)
    return {name: np.load(data_dir / f"delete_{name}.npy") for name in DELETE_FRACTION
            if (data_dir / f"delete_{name}.npy").exists()}


def deleted_returned(ids, mask) -> int:
    """Returned IDs (not -1) that the delete mask marks deleted, summed over all queries."""
    ids = np.asarray(ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(mask))]
    return int(np.asarray(mask, dtype=bool)[ids].sum())


def load_filters(data_dir) -> dict:
    """filters.json of data_dir, or {} when the data set has no filters."""
    path = Path(data_dir) / "filters.json"
    return json.loads(path.read_text()) if path.exists() else {}


def truth_for(ground_truth, filter_name: str):
    """Pick the truth for one filter. ground_truth is one array (unfiltered runs only) or a dict from load_truths."""
    if not isinstance(ground_truth, dict):
        if filter_name != "none":
            raise ValueError(f"run has filter={filter_name} but only the unfiltered truth was given")
        return ground_truth
    if filter_name not in ground_truth:
        raise ValueError(f"no ground truth for filter={filter_name}; known: {sorted(ground_truth)}")
    return ground_truth[filter_name]


def summarize(doc: dict, ground_truth, k: int = 10, delete_masks: dict | None = None) -> list[dict]:
    """One row per search setting in doc. k is the recall cutoff (at most doc['k']).

    ground_truth is one array (used for unfiltered runs) or {name: array}; each run is scored
    against the truth of its search_params["deleted"] / ["updated"] (CONTRACT section 13.2) or
    else its ["filter"] (section 11.2). delete_masks ({del30: bool array}, load_delete_masks)
    gives deleted_returned for runs with search_params.deleted; without it the count is NaN.
    """
    k = min(k, doc["k"])
    rows = []
    for run in doc["searches"]:
        filt = filter_of(run)
        _, recall = recall_at_k(run["ids"], truth_for(ground_truth, truth_key(run)), k)
        lat = latency_stats(run["latency_ms"])
        rows.append({
            "language": doc["language"], "index": doc["index"], "n": doc["n"],
            "build_params": doc["build_params"], "search_params": run["search_params"],
            "filter": filt,
            f"recall@{k}": recall, "p50_ms": lat["p50_ms"], "p90_ms": lat["p90_ms"], "p99_ms": lat["p99_ms"],
            "mean_ms": lat["mean_ms"], "qps": run["qps"], "build_s": doc["build"]["total_s"],
            "peak_rss_mb": doc["build"]["peak_rss_mb"], "index_bytes": doc["build"]["index_bytes"],
            "distance_computations": run["distance_computations"],
            # Spread of the runner's repeat runs (mean p50 over the search settings per run).
            "p50_spread": _spread(doc.get("extra", {}).get("runner", {}).get("p50_ms_runs")),
            # Phase 4 (CONTRACT section 12): load runs have extra.clients; recall is from worker 0's first pass.
            **load_fields(run),
            **change_fields(doc, run, delete_masks),
        })
    return rows


def change_fields(doc: dict, run: dict, delete_masks: dict | None) -> dict:
    """Phase 5 columns of one search run (CONTRACT section 13.2); "" / NaN when absent."""
    sp, ex = run["search_params"], doc.get("extra", {})
    nan = float("nan")
    deleted = str(sp.get("deleted", "") or "")
    returned = nan
    if deleted and delete_masks and deleted in delete_masks:
        returned = deleted_returned(run["ids"], delete_masks[deleted])
    num = lambda key: ex[key] if isinstance(ex.get(key), (int, float)) and not isinstance(ex.get(key), bool) else nan
    return {
        "deleted": deleted, "updated": str(sp.get("updated", "") or ""),
        "compacted": int(sp["compacted"]) if "compacted" in sp else nan,
        "delete_s": num("delete_s"), "update_s": num("update_s"), "compact_s": num("compact_s"),
        "index_bytes_after": num("index_bytes_after"), "disk_bytes": num("disk_bytes"),
        "disk_bytes_after": num("disk_bytes_after"), "deleted_returned": returned,
    }


def load_fields(run: dict) -> dict:
    """clients, cpu_pct, errors, insert_rate, phase of one search run; NaN / "" when absent."""
    ex = run.get("extra", {})
    nan = float("nan")
    return {
        "clients": ex.get("clients", nan), "cpu_pct": ex.get("cpu_pct", nan), "errors": ex.get("errors", nan),
        "insert_rate": ex.get("insert_rate", 0) or 0, "inserted_rows": ex.get("inserted_rows", nan),
        "phase": str(run["search_params"].get("phase", "")),
    }


def _spread(p50_runs) -> str:
    """'min-max' of the repeat runs' p50, or '' when the runner did not repeat."""
    if not p50_runs or len(p50_runs) < 2:
        return ""
    return f"{min(p50_runs):.3g}-{max(p50_runs):.3g}"
