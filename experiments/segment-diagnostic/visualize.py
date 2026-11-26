"""Draw paired metric changes and show the source frames that supplied the retained tokens."""
import csv
from functools import lru_cache
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from analyze import OUT, METHODS, pixel_bounds, read_json, read_trace


FIG = OUT / "figures"
METHOD_NAMES = {"attention": "Attention", "mixture": "Attention + diversity"}
BLUE = "#226789"
GOLD = "#d88936"
INK = "#182d39"
MUTED = "#566a73"


@lru_cache()
def font(size, bold=False):
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    return ImageFont.truetype(str(font_dir / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def chart_defaults():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#aebdc1", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "savefig.facecolor": "white"})


def draw_effects(summary):
    with (OUT / "paired_segments.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = sorted({int(r["id"]) for r in rows})
    panels = [("subject", "Segment subject consistency", "percentage points"),
              ("global_anchor", "Similarity to video frame 0", "percentage points"),
              ("musiq", "Image quality (MUSIQ)", "score points"),
              ("flow_top5_px", "Motion (RAFT)", "pixels per frame pair")]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.7), layout="constrained")
    fig.suptitle("Effect of adding frame diversity to attention", fontsize=18, weight="bold")
    for ax, (measure, title, unit) in zip(axes.flat, panels):
        grid = np.array([[float(r[f"{measure}_delta"]) for r in rows if int(r["id"]) == i] for i in ids])
        for line in grid:
            ax.plot(range(3), line, color="#b8c6ca", alpha=0.65, linewidth=1, zorder=1)
        avg = np.array([s[measure]["delta"]["mean"] for s in summary["windows"]])
        lo = np.array([s[measure]["delta"]["low"] for s in summary["windows"]])
        hi = np.array([s[measure]["delta"]["high"] for s in summary["windows"]])
        ax.axhline(0, color="#70838a", linestyle="--", linewidth=0.9)
        ax.errorbar(range(3), avg, yerr=np.vstack([avg-lo, hi-avg]), color=BLUE, linewidth=2,
                    fmt="o-", markersize=5, capsize=4, zorder=4)
        ax.set_title(title, loc="left", weight="bold", fontsize=12)
        ax.set_ylabel(unit)
        ax.set_xticks(range(3), ["0–10 s", "10–20 s", "20–30 s"])
        ax.set_xlim(-0.17, 2.17)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", color="#e5ebed", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.supxlabel("Difference = mixture minus attention. Grey: paired prompts. Blue: mean and paired 95% bootstrap interval.", fontsize=10)
    fig.savefig(FIG / "paired-effects.png", dpi=165)
    plt.close(fig)


def cache_grid(rows):
    queries = sorted({r["query"] for r in rows})
    ix = {q: i for i, q in enumerate(queries)}
    actual = np.zeros((len(queries), 31))
    native = np.zeros_like(actual)
    for r in rows:
        i = ix[r["query"]]
        for source, _, m, a in r["frames"]:
            start, end = pixel_bounds(source)
            second = int((start + end - 1) / 32)
            actual[i, second] += m
            native[i, second] += a
    denom = actual.sum(axis=1, keepdims=True)
    return np.array([pixel_bounds(q)[0]/16 for q in queries]), actual/denom, native/denom


def draw_heatmaps(traces, examples, labels):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), layout="constrained")
    fig.suptitle("Which seconds supplied the selected historical tokens?", weight="bold", fontsize=17)
    panels = []
    for ident in examples:
        t, a, _ = cache_grid(traces[ident]["attention"])
        _, m, native = cache_grid(traces[ident]["mixture"])
        panels.append((t, a*100, m*100, (m-native)*100))
    regular_limit = max(p[1:3][i].max() for p in panels for i in range(2))
    diverging_limit = max(np.abs(p[3]).max() for p in panels)
    for row, (ident, (t, a, m, diff)) in enumerate(zip(examples, panels)):
        y_edges = np.r_[t[0]-(t[1]-t[0])/2, (t[1:] + t[:-1])/2, t[-1]+(t[-1]-t[-2])/2]
        for col, values in enumerate((a, m, diff)):
            ax = axes[row, col]
            opts = dict(cmap="YlGnBu", vmin=0, vmax=regular_limit) if col < 2 else dict(
                cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-diverging_limit, vmax=diverging_limit))
            mesh = ax.pcolormesh(np.arange(32), y_edges, values, shading="flat", **opts)
            ax.set_xlim(0, 30)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax.set_xlabel("Source time (s)")
            if col == 0:
                ax.set_ylabel(labels[ident] + "\nEviction time (s)", fontweight="bold")
            if row == 0:
                ax.set_title(["Attention", "Attention + diversity", "Mixture minus its same-cache\nattention selection"][col], weight="bold", fontsize=11)
            if col == 2:
                fig.colorbar(mesh, ax=ax, label="Change (pp)", shrink=0.82)
            elif col == 1:
                fig.colorbar(mesh, ax=axes[row,:2], label="Share (%)", shrink=0.82)
    fig.savefig(FIG / "retention-map.png", dpi=150)
    plt.close(fig)


def best_sources(events, q):
    counts = {}
    chosen = [r for r in events if r["query"] == q]
    for row in chosen:
        for frame, _, selected, _ in row["frames"]:
            counts[frame] = counts.get(frame, 0) + selected
    total = sum(counts.values())
    return [(f, n / total) for f, n in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]]


def picture_indices(video_file, indices, size):
    result = {}
    with av.open(str(video_file)) as stream:
        for i, frame in enumerate(stream.decode(video=0)):
            if i in indices:
                result[i] = ImageOps.fit(frame.to_image(), size, method=Image.Resampling.LANCZOS)
                if len(result) == len(indices):
                    break
    if len(result) != len(indices):
        raise ValueError(f"Missing preview frames for {video_file}")
    return result


def frame_mid(f):
    start, stop = pixel_bounds(f)
    return int((start + stop - 1) // 2)


def draw_saved_and_story(traces, examples, labels):
    tw, th = 240, 138
    margin, left, pitch, y_start = 20, 158, 169, 65
    story = Image.new("RGB", (left+6*(tw+6)+margin, y_start+4*pitch+30), "white")
    kept = Image.new("RGB", (left+4*(tw+6)+margin, y_start+4*pitch+30), "white")
    sd, kd = ImageDraw.Draw(story), ImageDraw.Draw(kept)
    sd.text((margin, 10), "Paired videos at fixed playback times", font=font(22, True), fill=INK)
    kd.text((margin, 10), "Historical frames contributing the most selected tokens", font=font(22, True), fill=INK)
    secs = [3, 8, 13, 18, 23, 28]
    for col, sec in enumerate(secs):
        sd.text((left+col*(tw+6), 40), f"{sec} s", font=font(15, True), fill=MUTED)
    for col, name in enumerate(("At eviction", "1st source", "2nd source", "3rd source")):
        kd.text((left+col*(tw+6), 40), name, font=font(15, True), fill=MUTED)
    for i, ident in enumerate(examples):
        for j, method in enumerate(METHODS):
            rix = 2*i+j
            y = y_start+rix*pitch
            events = traces[ident][method]
            q = min({r["query"] for r in events}, key=lambda f: abs(pixel_bounds(f)[0]/16 - 27))
            best = best_sources(events, q)
            selected = [pixel_bounds(q)[0]] + [frame_mid(f) for f,_ in best]
            targets = set(selected) | {16*s for s in secs}
            clips = picture_indices(OUT/"videos"/f"{ident:03d}-{method}.mp4", targets, (tw, th))
            for drawer in (sd, kd):
                drawer.text((margin,y+30), f"#{ident:03d}", font=font(16, True), fill=INK)
                words = labels[ident].split()
                for k, word in enumerate(words):
                    drawer.text((margin,y+54+18*k), word, font=font(14), fill=MUTED)
                drawer.text((margin,y+116), "Attention" if j == 0 else "+ diversity", font=font(14, True),
                            fill=BLUE if j == 0 else "#9b590c")
            for col, sec in enumerate(secs):
                story.paste(clips[sec*16], (left+col*(tw+6), y))
            for col, frame in enumerate(selected):
                x = left+col*(tw+6)
                kept.paste(clips[frame], (x, y))
                caption = f"{frame/16:.2f} s"
                if col:
                    caption += f"   |   {best[col-1][1]*100:.1f}%"
                kd.text((x,y+th+3), caption, font=font(14), fill=INK)
    sd.text((margin,story.height-27), "The cases have the lowest and highest final-window subject-consistency change in this pilot.", font=font(14), fill=MUTED)
    kd.text((margin,kept.height-27), "Shares refer to the 3,120 selectable historical slots, pooled over layers 0, 14 and 29. Protected slots are excluded.", font=font(13), fill=MUTED)
    story.save(FIG/"storyboard.jpg", quality=91)
    kept.save(FIG/"retained-frames.jpg", quality=91)


def late_video(ident, label):
    size = (520, 300)
    files = [OUT / "videos" / f"{ident:03d}-{method}.mp4" for method in METHODS]
    with av.open(str(files[0])) as left, av.open(str(files[1])) as right, av.open(str(FIG/f"{ident:03d}-late.mp4"), "w") as dest:
        encoder = dest.add_stream("libx264", rate=16)
        encoder.width, encoder.height, encoder.pix_fmt = 1040, 350, "yuv420p"
        encoder.options = {"crf": "19", "preset": "fast"}
        emitted = 0
        for i, pair in enumerate(zip(left.decode(video=0), right.decode(video=0))):
            if i < 320:
                continue
            if i >= 480:
                break
            canvas = Image.new("RGB", (1040,350), "#15242d")
            pen = ImageDraw.Draw(canvas)
            pen.text((11,5), "ATTENTION", font=font(15,True), fill="#d0eaf1")
            pen.text((531,5), "ATTENTION + DIVERSITY", font=font(15,True), fill="#ffe0b0")
            for col, frame in enumerate(pair):
                canvas.paste(frame.to_image().resize(size, Image.Resampling.BILINEAR), (col*520,26))
            pen.text((11,329), f"#{ident:03d}  {label}   |   {i/16:.2f} s", font=font(14), fill="white")
            for packet in encoder.encode(av.VideoFrame.from_image(canvas)):
                dest.mux(packet)
            emitted += 1
        for packet in encoder.encode():
            dest.mux(packet)
        if emitted != 160:
            raise ValueError(f"Incomplete 20–30 second preview for {ident}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    chart_defaults()
    summary = read_json(OUT/"summary.json")
    examples = [summary["examples"][p] for p in ("lowest", "highest")]
    labels = {r["id"]: r["label"] for r in read_json(Path("prompts.json"))}
    traces = {i: {m: read_trace(f"{i:03d}-{m}") for m in METHODS} for i in examples}
    draw_effects(summary)
    draw_heatmaps(traces, examples, labels)
    draw_saved_and_story(traces, examples, labels)
    for i in examples:
        late_video(i, labels[i])
    print("Rendered diagnostic figures and paired late-window videos", examples)


if __name__ == "__main__":
    main()
