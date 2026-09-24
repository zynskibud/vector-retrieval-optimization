"""Summarize every run under results/raw/<data-name>/ into a CSV, a markdown table, and plots.

Writes results/summary/<data-name>/: results.csv, results.md, <index>.png

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

from tools.bench.metrics import summarize

SWEEP_KEY = {"ivf": "nprobe", "ivf_pq": "nprobe", "pq": "rerank", "hnsw": "ef", "diskann": "l"}


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cell = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fmt_params(p: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in p.items())


def load_rows(raw: Path, gt: np.ndarray) -> pd.DataFrame:
    rows = []
    for path in sorted(raw.glob("*.json")):
        doc = json.loads(path.read_text())
        for r in summarize(doc, gt):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev"))
    args = ap.parse_args()
    name = args.data.name
    raw, out = Path("results/raw") / name, Path("results/summary") / name
    gt = np.load(args.data / "ground_truth.npy")
    df = load_rows(raw, gt)
    if df.empty:
        raise SystemExit(f"no JSON files in {raw}")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["index", "language", "build_params", "search_params", "recall@10", "p50_ms", "p50_spread", "p90_ms",
            "p99_ms", "qps", "build_s", "peak_rss_mb", "index_bytes", "distance_computations"]
    df = df.sort_values(["index", "recall@10", "p50_ms"], ascending=[True, False, True])
    df[cols + ["n", "file"]].to_csv(out / "results.csv", index=False)

    md = [f"# Results: {name} (n = {df['n'].iloc[0]:,})", ""]
    for index, g in df.groupby("index"):
        md += [f"## {index}", "", md_table(g[cols[1:]]), ""]
        plot(g, index, f"{name}, n = {g['n'].iloc[0]:,}", out / f"{index}.png")
    (out / "results.md").write_text("\n".join(md))
    for f in sorted(out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
