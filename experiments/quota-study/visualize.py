"""Plot paired effects and render prespecified peacock and basketball comparisons."""

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare import load


NAMES = {
    "attention": "Attention",
    "mixture": "Additive",
    "capped": "Frame cap",
    "two_stage": "Two-stage",
}
COLORS = {
    "attention": "#515c64",
    "mixture": "#cc682c",
    "capped": "#248163",
    "two_stage": "#3a64a8",
}


@lru_cache()
def font(size, bold=False):
    folder = Path(matplotlib.get_data_path()) / "fonts/ttf"
    return ImageFont.truetype(
        str(folder / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
    )


def effects(folder, summary, rows, figures):
    arms = summary["methods"][1:]
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.5), layout="constrained")
    fig.suptitle(
        "Changes from attention-only selection, 20–30 s", fontsize=16, weight="bold"
    )
    for ax, (key, title, unit) in zip(
        axes.flat,
        [
            ("subject", "Subject consistency", "percentage points"),
            ("effective", "Effective source frames", "frames"),
            ("musiq", "MUSIQ quality", "score points"),
            ("flow_top5_px", "RAFT motion", "pixels per frame pair"),
        ],
    ):

        def vals(method):
            return {
                int(r["id"]): float(r[key])
                for r in rows
                if r["method"] == method and r["window"] == "2"
            }

        ref = vals("attention")
        ds = np.array([[vals(m)[p] - ref[p] for m in arms] for p in ref])
        for values in ds:
            ax.plot(
                range(len(arms)), values, color="#c4ced0", alpha=0.8, lw=0.9, zorder=1
            )
        for j, m in enumerate(arms):
            ci = summary["contrasts"][m + "-attention"][2][key]
            mid = ci["mean"]
            ax.errorbar(
                j,
                mid,
                yerr=[[mid - ci["low"]], [ci["high"] - mid]],
                fmt="o",
                color=COLORS[m],
                capsize=4,
                ms=6,
                lw=2,
                zorder=3,
            )
        ax.axhline(0, ls="--", lw=0.85, color="#60727a")
        ax.set_xticks(range(len(arms)), [NAMES[m] for m in arms])
        ax.set_xlim(-0.30, len(arms) - 0.70)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.5, color="#e5e8e8")
    fig.supxlabel(
        "Grey lines follow individual prompt seeds. Dots and whiskers show paired means and 95% bootstrap intervals.",
        fontsize=9,
    )
    fig.savefig(figures / "paired-effects.png", dpi=155)
    plt.close(fig)


def cases(ident, label, methods, sources, rows, dest):
    scores = {
        r["method"]: float(r["subject"])
        for r in rows
        if int(r["id"]) == ident and r["window"] == "2"
    }
    tw, th = 416, 240
    ncol = 2 if len(methods) == 4 else 3
    nrow = (len(methods) + ncol - 1) // ncol
    width, height = ncol * tw, nrow * (th + 28) + 28
    secs = (8, 18, 23, 28)
    thumbs = {}
    streams = [
        av.open(str(sources[m] / "videos" / f"{ident:03d}-{m}.mp4")) for m in methods
    ]
    try:
        with av.open(str(dest / f"{ident:03d}-late.mp4"), "w") as video:
            writer = video.add_stream("libx264", rate=16)
            writer.width, writer.height, writer.pix_fmt = width, height, "yuv420p"
            writer.options = {"crf": "19", "preset": "fast"}
            count = 0
            for index, frames in enumerate(zip(*(s.decode(video=0) for s in streams))):
                if index >= 480:
                    break
                if index not in [s * 16 for s in secs] and index < 320:
                    continue
                decoded = [
                    f.to_image().resize((tw, th), Image.Resampling.BILINEAR)
                    for f in frames
                ]
                if index in [s * 16 for s in secs]:
                    for m, im in zip(methods, decoded):
                        thumbs[m, index // 16] = im.resize(
                            (240, 138), Image.Resampling.LANCZOS
                        )
                if index < 320:
                    continue
                page = Image.new("RGB", (width, height), "#14262d")
                draw = ImageDraw.Draw(page)
                for pos, (method, img) in enumerate(zip(methods, decoded)):
                    x = (pos % ncol) * tw
                    y = (pos // ncol) * (th + 28)
                    draw.text(
                        (x + 9, y + 4), NAMES[method], font=font(16, True), fill="white"
                    )
                    page.paste(img, (x, y + 28))
                draw.text(
                    (10, height - 23),
                    f"#{ident:03d}  {label}   |   {index / 16:.2f} s",
                    font=font(15),
                    fill="white",
                )
                for pkt in writer.encode(av.VideoFrame.from_image(page)):
                    video.mux(pkt)
                count += 1
            for pkt in writer.encode():
                video.mux(pkt)
            if count != 160:
                raise ValueError("Incomplete comparison clip")
    finally:
        for stream in streams:
            stream.close()
    left, top, pitch = 155, 61, 145
    sheet = Image.new(
        "RGB", (left + 4 * 246 + 10, top + pitch * len(methods) + 12), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 8), f"#{ident:03d}  {label}", font=font(21, True), fill="#1b303a")
    for j, t in enumerate(secs):
        draw.text((left + 246 * j, 37), f"{t} s", font=font(15, True), fill="#526c75")
    for i, m in enumerate(methods):
        y = top + i * pitch
        draw.text((15, y + 46), NAMES[m], font=font(16, True), fill=COLORS[m])
        draw.text(
            (15, y + 72), f"Subject  {scores[m]:.2f}", font=font(13), fill="#526c75"
        )
        for j, t in enumerate(secs):
            sheet.paste(thumbs[m, t], (left + 246 * j, y))
    sheet.save(dest / f"{ident:03d}-frames.jpg", quality=91)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("."))
    parser.add_argument("--pilot", type=Path, default=Path("../segment-pilot-20261124"))
    args = parser.parse_args()
    folder = args.run / "output"
    summary = load(folder / "summary.json")
    with (folder / "per_prompt.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    sources = {m: folder for m in summary["methods"]}
    if summary["stage"] == "paired":
        sources.update(attention=args.pilot / "output", mixture=args.pilot / "output")
    effects(folder, summary, rows, figures)
    for ident, label in ((86, "Peacock"), (3, "Basketball jump")):
        cases(ident, label, summary["methods"], sources, rows, figures)
    print("Rendered paired charts and fixed example videos")


if __name__ == "__main__":
    main()
