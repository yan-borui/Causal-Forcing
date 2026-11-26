"""Extract the four baseline cases and the workshop source frame on CPU."""

from pathlib import Path

import av
from PIL import Image, ImageDraw

RUN = Path.cwd()
OLD = {24: "segment-pilot-20261124", 34: "quota-replication-20261124"}
SEEDS = (34001, 24017, 34017, 34079)
TIMES = (0.5, 18.5, 21.5, 22.25, 22.6875, 23, 25.5, 27.75, 28.5, 29.25)


def baseline(seed):
    return RUN.parent / OLD[seed // 1000] / "output"


def main():
    dest = RUN / "output"
    dest.mkdir(exist_ok=True)
    page = Image.new("RGB", (188 * len(TIMES), 131 * 4), "white")
    draw = ImageDraw.Draw(page)
    indices = {round(t * 16): i for i, t in enumerate(TIMES)}
    for row, seed in enumerate(SEEDS):
        video = baseline(seed) / "videos" / f"{seed % 1000:03d}-attention.mp4"
        with av.open(str(video)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i not in indices:
                    continue
                col = indices[i]
                im = frame.to_image()
                if seed == 34001 and i == 363:
                    im.save(dest / "workshop-363.png")
                page.paste(im.resize((186, 107)), (188 * col, 131 * row + 22))
                draw.text(
                    (188 * col + 3, 131 * row + 4),
                    f"{seed}   {i / 16:g}s",
                    fill="black",
                )
    page.save(dest / "examples.jpg", quality=90)


if __name__ == "__main__":
    main()
