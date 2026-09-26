"""Summarize every run under results/raw/<data-name>/ into a CSV, a markdown table, and plots.

Writes results/summary/<data-name>/: results.csv, results.md, <index>.png, and for each index
with filtered runs (CONTRACT section 11) <index>-filter.png: recall@10 and p50 latency against
the filter's selectivity, one line per system, at the default search setting.

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

from tools.bench.metrics import load_filters, load_truths, summarize

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


def load_rows(raw: Path, gt, filters: dict | None = None) -> pd.DataFrame:
    """gt: one truth array or {filter name: array} (metrics.load_truths). filters: filters.json."""
    filters = filters or {}
    rows = []
    for path in sorted(raw.glob("*.json")):
        doc = json.loads(path.read_text())
        for r in summarize(doc, gt):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    args = ap.parse_args()
    name = args.data.name
    raw, out = Path("results/raw") / name, Path("results/summary") / name
    df = load_rows(raw, load_truths(args.data), load_filters(args.data))
    if df.empty:
        raise SystemExit(f"no JSON files in {raw}")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["index", "language", "build_params", "search_params", "filter", "selectivity", "recall@10", "p50_ms", "p50_spread", "p90_ms",
            "p99_ms", "qps", "build_s", "peak_rss_mb", "index_bytes", "distance_computations"]
    df = df.sort_values(["index", "recall@10", "p50_ms"], ascending=[True, False, True])
    df[cols + ["n", "file"]].to_csv(out / "results.csv", index=False)

    md = [f"# Results: {name} (n = {df['n'].iloc[0]:,})", ""]
    for index, g in df.groupby("index"):
        md += [f"## {index}", "", md_table(g[cols[1:]]), ""]
        # The latency-recall plot keeps the unfiltered runs only, as before Phase 3.
        plot(g[g["filter"] == "none"], index, f"{name}, n = {g['n'].iloc[0]:,}", out / f"{index}.png")
        plot_filter(g, index, f"{name}, n = {g['n'].iloc[0]:,}", out / f"{index}-filter.png")
    (out / "results.md").write_text("\n".join(md))
    for f in sorted(out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
