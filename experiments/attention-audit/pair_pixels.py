"""Quantify the observed archived/replayed MP4 pixel differences and show event pairs."""

import csv
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

from examples import baseline, SEEDS


def main():
    out = Path("output")
    chosen = {
        34001: (356, 363, 368),
        24017: (450, 456, 462),
        34017: (450, 456, 462),
        34079: (356, 363, 368),
    }
    plate = Image.new("RGB", (3 * 310, 4 * 2 * 109), "white")
    draw = ImageDraw.Draw(plate)
    rows = []
    for row, seed in enumerate(SEEDS):
        stem = f"{seed % 1000:03d}-attention.mp4"
        new = out / str(seed)
        with (
            av.open(str(baseline(seed) / "videos" / stem)) as a,
            av.open(str(new / "videos" / stem)) as b,
        ):
            for i, (fa, fb) in enumerate(
                zip(a.decode(video=0), b.decode(video=0), strict=True)
            ):
                if fa.pts * fa.time_base != fb.pts * fb.time_base:
                    raise ValueError("Presentation timestamps differ")
                x, y = [f.to_ndarray(format="rgb24") for f in (fa, fb)]
                delta = x.astype("float32") - y.astype("float32")
                # Box-average in fixed 16x16 patches and compare the spatial luminance pattern.
                ax, ay = [
                    cv2.resize(
                        cv2.cvtColor(z, cv2.COLOR_RGB2GRAY),
                        (52, 30),
                        interpolation=cv2.INTER_AREA,
                    ).ravel()
                    for z in (x, y)
                ]
                r = float(np.corrcoef(ax, ay)[0, 1])
                rows.append(
                    (seed, i, float(np.abs(delta).mean()), float(np.mean(delta**2)), r)
                )
                if i in chosen[seed]:
                    col = chosen[seed].index(i)
                    for j, (pic, label) in enumerate(((x, "archived"), (y, "replay"))):
                        x0, t = col * 310, row * 218 + j * 109
                        draw.text(
                            (x0 + 2, t + 3), f"{seed} {i / 16:g}s {label}", fill="black"
                        )
                        plate.paste(
                            Image.fromarray(pic).resize((176, 99)), (x0 + 128, t + 2)
                        )
        values = np.array([r[2:] for r in rows if r[0] == seed])
        result = dict(
            grid_pearson_min=float(values[:, 2].min()),
            grid_pearson_median=float(np.median(values[:, 2])),
            psnr_db=float(10 * np.log10(255**2 / np.mean(values[:, 1]))),
            grid_pearson_at_events={
                str(i): r for s, i, _, _, r in rows if s == seed and i in chosen[seed]
            },
        )
        (new / "pixel_alignment.json").write_text(json.dumps(result, indent=2))
    with (out / "pixel_alignment.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("seed", "frame", "mae_rgb", "mse_rgb", "grid_luma_pearson"))
        writer.writerows(rows)
    plate.save(out / "pair-events.jpg", quality=94)


if __name__ == "__main__":
    main()
