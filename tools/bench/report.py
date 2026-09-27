"""Summarize every run under results/raw/<data-name>/ into a CSV, a markdown table, and plots.

Writes results/summary/<data-name>/: results.csv, results.md, <index>.png, and for each index
with filtered runs (CONTRACT section 11) <index>-filter.png: recall@10 and p50 latency against
the filter's selectivity, one line per system, at the default search setting; and for each index
with load runs (extra.clients, CONTRACT section 12) <index>-load.png: QPS and p99 against clients;
and for each index with delete runs (search_params.deleted, CONTRACT section 13) <index>-delete.png:
recall@10 and p50 against the deleted fraction (0 = the unchanged runs), solid before and dashed
after compaction, plus a "Updates, deletes, compaction" table in results.md; and for cache runs
(language "cache", CONTRACT section 14) hnsw-cache.png: hit rate and end-to-end p50 against
capacity for lru and redis on the zipf workload, with the none backend as a horizontal line,
plus an "Embedding cache" table in results.md; and for backup runs (extra.restore_identical,
CONTRACT section 15.3) a "Backup and restore" table in results.md, one row per file.

Run: uv run python -m tools.bench.report --data data/processed/dev
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.bench.metrics import DELETE_FRACTION, load_delete_masks, load_filters, load_truths, summarize

# The default search setting per index (CONTRACT section 6); the filter plots use only these runs.
DEFAULT_SEARCH = {"flat": {}, "ivf": {"nprobe": 8}, "hnsw": {"ef": 64}}
# Search/build params that make a separate line in the filter plot when not at their default.
FILTER_LINE_KEYS = {"quant": "none", "rescore": 0, "views_index": 0, "iterative": 0}
SWEEP_KEY = {"ivf": "nprobe", "ivf_pq": "nprobe", "pq": "rerank", "hnsw": "ef", "diskann": "l"}


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cell = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fmt_params(p: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in p.items())


def load_rows(raw: Path, gt, filters: dict | None = None, delete_masks: dict | None = None) -> pd.DataFrame:
    """gt: one truth array or {name: array} (metrics.load_truths). filters: filters.json.
    delete_masks: metrics.load_delete_masks, for deleted_returned."""
    filters = filters or {}
    rows = []
    for path in sorted(raw.glob("*.json")):
        doc = json.loads(path.read_text())
        for r in summarize(doc, gt, delete_masks=delete_masks):
            r["selectivity"] = 1.0 if r["filter"] == "none" else filters.get(r["filter"], {}).get("selectivity", np.nan)
            both = {**r["build_params"], **r["search_params"]}
            r["filter_line"] = r["language"] + "".join(
                f" {k}={both[k]}" for k, dflt in FILTER_LINE_KEYS.items() if k in both and both[k] != dflt)
            r["is_default"] = all(r["search_params"].get(k) == v for k, v in DEFAULT_SEARCH.get(r["index"], {"_": 0}).items())
            r["line"] = r["language"]
            if "metric" in r["build_params"]:
                r["line"] += f" {r['build_params']['metric']}"
            if "io" in r["search_params"]:
                r["line"] += f" {r['search_params']['io']}"
            if r["index"] in ("ivf_pq",) and "rerank" in r["search_params"]:
                r["line"] += f" rerank={r['search_params']['rerank']}"
            r["sweep"] = r["search_params"].get(SWEEP_KEY.get(r["index"], ""), "")
            r["build_params"], r["search_params"] = fmt_params(r["build_params"]), fmt_params(r["search_params"])
            r["file"] = path.name
            r["is_load"] = not pd.isna(r["clients"])
            r["load_line"] = r["language"] + (" +inserts" if r["insert_rate"] else "")
            r["is_change"] = bool(r["deleted"] or r["updated"])
            r["is_cache"] = r["language"] == "cache"
            r["is_backup"] = "restore_identical" in doc.get("extra", {})
            r["deleted_fraction"] = DELETE_FRACTION.get(r["deleted"], 0.0 if not r["updated"] else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, index: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for line, g in df.groupby("line"):
        g = g.sort_values("p50_ms")
        ax.plot(g["p50_ms"], g["recall@10"], marker="o", label=line)
        for _, r in g.iterrows():
            ax.annotate(str(r["sweep"]), (r["p50_ms"], r["recall@10"]), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency (ms, log scale)")
    ax.set_ylabel("recall@10")
    ax.set_title(f"{index}: {title}  (point labels: {SWEEP_KEY.get(index, '-')})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_filter(df: pd.DataFrame, index: str, title: str, out: Path) -> bool:
    """Recall@10 and p50 against selectivity (log x; "none" at 1.0), default search setting only.

    Returns False (and writes nothing) when the index has no filtered run.
    """
    g = df[df["is_default"]]
    if (g["filter"] == "none").all():
        return False
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13, 5))
    for line, h in g.groupby("filter_line"):
        h = h.groupby("selectivity", as_index=False)[["recall@10", "p50_ms"]].median().sort_values("selectivity")
        ax_r.plot(h["selectivity"], h["recall@10"], marker="o", label=line)
        ax_l.plot(h["selectivity"], h["p50_ms"], marker="o", label=line)
    ticks = sorted(g["selectivity"].dropna().unique())
    labels = {1.0: "none"} | {s: f for f, s in zip(g["filter"], g["selectivity"]) if f != "none"}
    for ax, ylab in ((ax_r, "recall@10"), (ax_l, "p50 latency (ms, log scale)")):
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{labels.get(t, '')}\n{t:.3g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("selectivity (fraction of rows that pass, log scale)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    ax_l.set_yscale("log")
    ax_r.legend(fontsize=8)
    default = fmt_params(DEFAULT_SEARCH.get(index, {})) or "exact"
    fig.suptitle(f"{index} with filter views >= t: {title}  (search {default})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def plot_load(df: pd.DataFrame, index: str, title: str, out: Path) -> bool:
    """QPS and p99 against clients (log x), one line per system; runs with inserts are their own
    line ("+inserts"). Returns False (and writes nothing) when the index has no load run."""
    g = df[df["is_load"] & (df["filter"] == "none")]
    if g.empty:
        return False
    fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(13, 5))
    for line, h in g.groupby("load_line"):
        h = h.groupby("clients", as_index=False)[["qps", "p99_ms"]].median().sort_values("clients")
        style = "--" if line.endswith("+inserts") else "-"
        ax_q.plot(h["clients"], h["qps"], style, marker="o", label=line)
        ax_p.plot(h["clients"], h["p99_ms"], style, marker="o", label=line)
    ticks = sorted(g["clients"].unique())
    for ax, ylab in ((ax_q, "queries per second (all clients)"), (ax_p, "p99 latency (ms, log scale)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("concurrent clients (log scale)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    ax_p.set_yscale("log")
    ax_q.legend(fontsize=8)
    fig.suptitle(f"{index} under load: {title}  (default search setting; databases include the client round trip)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def plot_delete(df: pd.DataFrame, index: str, title: str, out: Path) -> bool:
    """Recall@10 and p50 against the deleted fraction (0, 0.1, 0.3, 0.5) at the default search
    setting, one color per system; solid = tombstones (compacted=0), dashed = after compaction.
    The point at 0 is the median of the system's unchanged runs. Returns False when the index
    has no delete run."""
    g = df[df["is_default"] & (df["filter"] == "none") & ~df["is_load"] & (df["phase"] == "") & (df["updated"] == "")]
    dels = g[g["deleted"] != ""]
    if dels.empty:
        return False
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for n, (line, h) in enumerate(g[g["filter_line"].isin(dels["filter_line"].unique())].groupby("filter_line")):
        base0 = h[h["deleted"] == ""]
        for comp, style in ((0, "-"), (1, "--")):
            part = pd.concat([base0, h[(h["deleted"] != "") & (h["compacted"] == comp)]])
            if (part["deleted"] != "").sum() == 0:
                continue
            m = part.groupby("deleted_fraction", as_index=False)[["recall@10", "p50_ms"]].median().sort_values("deleted_fraction")
            label = f"{line} {'compacted' if comp else 'tombstones'}"
            ax_r.plot(m["deleted_fraction"], m["recall@10"], style, marker="o", color=colors[n % len(colors)], label=label)
            ax_l.plot(m["deleted_fraction"], m["p50_ms"], style, marker="o", color=colors[n % len(colors)], label=label)
    for ax, ylab in ((ax_r, "recall@10 (against the remaining-rows truth)"), (ax_l, "p50 latency (ms, log scale)")):
        ax.set_xticks([0, 0.1, 0.3, 0.5])
        ax.set_xlabel("deleted fraction of rows")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    ax_l.set_yscale("log")
    ax_r.legend(fontsize=7)
    default = fmt_params(DEFAULT_SEARCH.get(index, {})) or "exact"
    fig.suptitle(f"{index} after deletes: {title}  (search {default}; solid = tombstones, dashed = compacted)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


CACHE_COLS = ["backend", "capacity", "workload", "invalidate_at", "hit_rate", "hit_rate_after_invalidate", "embed_p50_ms",
              "search_p50_ms", "e2e_p50_ms", "e2e_p99_ms", "entries", "cache_bytes", "evictions", "recall@10",
              "recall_on_queries", "file"]


def cache_table(df: pd.DataFrame) -> str:
    g = df[df["is_cache"]].sort_values(["workload", "backend", "capacity", "invalidate_at"])
    return md_table(g[CACHE_COLS]) if not g.empty else ""


def plot_cache(df: pd.DataFrame, title: str, out: Path) -> bool:
    """Hit rate (left y) and e2e p50 (right y) against capacity, lru and redis on zipf without
    invalidation; the none backend's e2e p50 as a horizontal line. False when there is no cache run."""
    g = df[df["is_cache"] & (df["workload"] == "zipf") & df["invalidate_at"].isna()]
    if g.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    colors = {"lru": "C0", "redis": "C1"}
    for backend in ("lru", "redis"):
        h = g[g["backend"] == backend].groupby("capacity", as_index=False)[["hit_rate", "e2e_p50_ms"]].median().sort_values("capacity")
        if h.empty:
            continue
        ax.plot(h["capacity"], h["hit_rate"], "-", marker="o", color=colors[backend], label=f"{backend} hit rate")
        ax2.plot(h["capacity"], h["e2e_p50_ms"], "--", marker="s", color=colors[backend], label=f"{backend} e2e p50")
    none = g[g["backend"] == "none"]
    if not none.empty:
        ax2.axhline(float(none["e2e_p50_ms"].median()), color="gray", linestyle=":", label="none e2e p50")
    ax.set_xscale("log")
    ax.set_xlabel("capacity (entries; redis: label only, the limit is maxmemory 512 MB)")
    ax.set_ylabel("hit rate (solid)")
    ax2.set_ylabel("end-to-end p50 (ms, dashed)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="center right")
    ax.set_title(f"hnsw with an embedding cache, zipf workload: {title}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


CHANGE_COLS = ["index", "language", "deleted", "updated", "compacted", "recall@10", "deleted_returned", "p50_ms",
               "delete_s", "update_s", "compact_s", "index_bytes", "index_bytes_after", "disk_bytes", "disk_bytes_after", "file"]


def change_table(df: pd.DataFrame) -> str:
    """One row per change run at the default search setting: cost of the change and of the compaction."""
    g = df[df["is_change"] & df["is_default"]].sort_values(["index", "language", "deleted", "updated", "compacted"])
    return md_table(g[CHANGE_COLS]) if not g.empty else ""


BACKUP_EXTRA = ["backup_s", "backup_bytes", "restore_s", "rebuild_needed", "cold", "rows_before", "rows_after",
                "restore_identical", "ids_equal_fraction"]


def backup_table(raw: Path, df: pd.DataFrame) -> str:
    """One row per backup file: the extra fields, and recall@10 / p50 of run 1 and run 2 (after_restore)."""
    g = df[df["is_backup"]]
    rows = []
    for f, h in g.groupby("file"):
        ex = json.loads((raw / f).read_text())["extra"]
        before, after = h[h["phase"] == ""], h[h["phase"] == "after_restore"]
        rows.append({"index": h["index"].iloc[0], "language": h["language"].iloc[0], **{c: ex.get(c) for c in BACKUP_EXTRA},
                     "recall@10": before["recall@10"].iloc[0], "recall@10_after": after["recall@10"].iloc[0],
                     "p50_ms": before["p50_ms"].iloc[0], "p50_ms_after": after["p50_ms"].iloc[0], "file": f})
    return md_table(pd.DataFrame(rows).sort_values(["index", "language"])) if rows else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    args = ap.parse_args()
    name = args.data.name
    raw, out = Path("results/raw") / name, Path("results/summary") / name
    df = load_rows(raw, load_truths(args.data), load_filters(args.data), load_delete_masks(args.data))
    if df.empty:
        raise SystemExit(f"no JSON files in {raw}")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["index", "language", "build_params", "search_params", "filter", "selectivity", "recall@10", "p50_ms", "p50_spread", "p90_ms",
            "p99_ms", "qps", "clients", "insert_rate", "phase", "cpu_pct", "errors", "build_s", "peak_rss_mb", "index_bytes", "distance_computations",
            "deleted", "updated", "compacted", "compact_s", "disk_bytes", "disk_bytes_after", "deleted_returned"]
    df = df.sort_values(["index", "recall@10", "p50_ms"], ascending=[True, False, True])
    df[cols + ["n", "file"]].to_csv(out / "results.csv", index=False)

    md = [f"# Results: {name} (n = {df['n'].iloc[0]:,})", ""]
    for index, g in df.groupby("index"):
        md += [f"## {index}", "", md_table(g[cols[1:]]), ""]
        # The latency-recall plot keeps the unfiltered runs only, as before Phase 3.
        static = g[~g["is_load"] & (g["phase"] == "") & ~g["is_change"] & ~g["is_cache"] & ~g["is_backup"]]
        if not static.empty:
            plot(static[static["filter"] == "none"], index, f"{name}, n = {g['n'].iloc[0]:,}",
                 out / f"{index}.png")
        plot_filter(static, index, f"{name}, n = {g['n'].iloc[0]:,}", out / f"{index}-filter.png")
        plot_load(g, index, f"{name}, n = {g['n'].iloc[0]:,}", out / f"{index}-load.png")
        plot_delete(g, index, f"{name}, n = {g['n'].iloc[0]:,}", out / f"{index}-delete.png")
    plot_cache(df, f"{name}, n = {df['n'].iloc[0]:,}", out / "hnsw-cache.png")
    ctable = cache_table(df)
    if ctable:
        md += ["## Embedding cache (CONTRACT section 14)", "",
               "FAISS HNSW (m=16, ef_construct=100, ef=64) behind the embedding model. Times in ms; embed_p50 over misses only; "
               "e2e = cache lookup + embed on a miss + search. recall@10 is over the first 1,000 requests whose text is a query.",
               "", ctable, ""]
    btable = backup_table(raw, df)
    if btable:
        md += ["## Backup and restore (CONTRACT section 15)", "",
               "Default build and search setting. backup_s / restore_s in seconds; backup_bytes = the snapshot, dump, or tar file. "
               "restore_s includes any index build (rebuild_needed) and, for Milvus, the restart and the collection load (cold = true). "
               "restore_identical = the IDs of all 1,000 queries are equal before and after, and rows_before = rows_after.",
               "", btable, ""]
    table = change_table(df)
    if table:
        md += ["## Updates, deletes, compaction (CONTRACT section 13)", "",
               "Default search setting. delete_s / update_s / compact_s in seconds; *_after = when the searches "
               "started (after compaction if compacted = 1). Bytes are what each system reports (tools/db/README.md).",
               "", table, ""]
    (out / "results.md").write_text("\n".join(md))
    for f in sorted(out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
