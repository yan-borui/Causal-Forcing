"""Draw mean scores in the three contiguous 10-second windows."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    out = parser.parse_args().result_dir
    with (out / "means.csv").open() as f:
        rows = list(csv.DictReader(f))
    methods = (
        ("causal_official_long", "Causal Forcing (official long)", "#172b42", "o"),
        ("attention", "Deep Forcing", "#3975a9", "s"),
        ("mixture", "Frame diversity (additive)", "#9e7758", "^"),
        ("capped", "Frame diversity + cap", "#a53552", "D"),
    )
    if any(row["method"] == "causal_sliding" for row in rows):
        methods = (
            methods[0],
            ("causal_sliding", "Causal Forcing (plain window)", "#6d5082", "X"),
            *methods[1:],
        )
    metrics = (
        ("subject_pct", "Subject consistency", "DINO score (%)"),
        ("musiq", "Imaging quality", "MUSIQ score"),
        ("flow_top5_px", "Motion magnitude", "RAFT displacement (pixels)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (metric, title, unit) in zip(axes, metrics):
        for key, label, color, symbol in methods:
            data = sorted(
                (r for r in rows if r["method"] == key), key=lambda r: int(r["start"])
            )
            ax.plot(
                [5, 15, 25],
                [float(r[metric]) for r in data],
                label=label,
                color=color,
                marker=symbol,
                lw=1.8,
                markersize=5,
            )
        ax.set(
            xticks=(5, 15, 25),
            xticklabels=("0–10", "10–20", "20–30"),
            title=title,
            xlabel="Time (s)",
            ylabel=unit,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(methods),
        frameon=False,
        fontsize=8.5,
    )
    fig.savefig(out / "segment-scores.png", dpi=160)


if __name__ == "__main__":
    main()
