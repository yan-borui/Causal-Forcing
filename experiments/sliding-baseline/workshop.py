"""Review the two workshop seeds on CPU, using existing videos and per-frame MUSIQ."""

import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUN = Path.cwd()
OUT = RUN / "output" / "workshop"
TIMES = (8, 328, 363, 392, 456)
LABELS = {
    "causal_official_long": "Official long",
    "causal_sliding": "Plain sliding",
    "attention": "Deep Forcing",
    "mixture": "Additive",
    "capped": "Capped",
}


def inputs(seed, sliding):
    archive = (
        "segment-pilot-20261124" if seed == 24001 else "quota-replication-20261124"
    )
    cap = "quota-study-20261124" if seed == 24001 else archive
    rows = [
        (
            "causal_official_long",
            RUN.parent / "causal-official-30" / "output",
            str(seed),
        )
    ]
    if sliding:
        rows.append(("causal_sliding", RUN / "output", str(seed)))
    rows += [
        (method, RUN.parent / folder / "output", f"001-{method}")
        for method, folder in [
            ("attention", archive),
            ("mixture", archive),
            ("capped", cap),
        ]
    ]
    return rows


def table(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def review(seed, sliding):
    rows = inputs(seed, sliding)
    font_file = (
        RUN.parent
        / "segment-pilot-20261124/plot-deps/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
    )
    font = ImageFont.truetype(str(font_file), 15)
    page = Image.new("RGB", (138 + len(TIMES) * 266, 40 + len(rows) * 157), "white")
    draw = ImageDraw.Draw(page)
    for col, idx in enumerate(TIMES):
        draw.text((142 + col * 266, 12), f"{idx / 16:.2f} s", font=font, fill="black")
    seconds, segments = [], []
    for row, (method, folder, stem) in enumerate(rows):
        scored = json.loads(
            (folder / "scores" / "quality" / f"{stem}.json").read_text()
        )
        quality = np.array(scored["musiq"]).reshape(30, 16).mean(1)
        if scored["method"] != method or scored["id"] != 1:
            raise ValueError("Workshop score and method differ")
        draw.text((5, 40 + row * 157 + 52), LABELS[method], font=font, fill="black")
        draw.text((5, 40 + row * 157 + 75), str(seed), font=font, fill="black")
        measures = []
        with av.open(str(folder / "videos" / f"{stem}.mp4")) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i >= 480:
                    break
                if i % 4 and i not in TIMES:
                    continue
                rgb = frame.to_image()
                if i in TIMES:
                    col = TIMES.index(i)
                    page.paste(
                        rgb.resize((262, 151), Image.Resampling.LANCZOS),
                        (138 + col * 266, 40 + row * 157),
                    )
                if i % 4 == 0:
                    level = np.asarray(
                        rgb.convert("L").resize((208, 120), Image.Resampling.BOX)
                    )
                    measures.append((level.mean(), 100 * (level < 32).mean()))
        light = np.array(measures).reshape(30, 4, 2).mean(1)
        for sec, (musiq, (mean, dark)) in enumerate(zip(quality, light)):
            seconds.append(
                dict(
                    seed=seed,
                    method=method,
                    second=sec,
                    musiq=musiq,
                    luma_0_255=mean,
                    dark_pct=dark,
                )
            )
        for part in range(3):
            region = slice(part * 10, part * 10 + 10)
            worst = int(np.argmin(quality[region])) + part * 10
            segments.append(
                dict(
                    seed=seed,
                    method=method,
                    start=part * 10,
                    musiq=quality[region].mean(),
                    luma_0_255=light[region, 0].mean(),
                    dark_pct=light[region, 1].mean(),
                    worst_sec=worst,
                    worst_sec_musiq=quality[worst],
                )
            )
    page.save(OUT / f"workshop-{seed}.jpg", quality=94)
    return seconds, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-sliding", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = [review(seed, args.with_sliding) for seed in (24001, 34001)]
    table("seconds.csv", [r for sec, _ in data for r in sec])
    table("segments.csv", [r for _, seg in data for r in seg])


if __name__ == "__main__":
    main()
