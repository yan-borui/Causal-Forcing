"""Resolve the onset of the staircase transition in the archived videos."""

import av
from PIL import Image, ImageDraw
from examples import baseline


def main():
    indices = list(range(436, 463, 2))
    page = Image.new("RGB", (7 * 218, 4 * 146), "white")
    draw = ImageDraw.Draw(page)
    for case, seed in enumerate((24017, 34017)):
        with av.open(str(baseline(seed) / "videos" / "017-attention.mp4")) as stream:
            for i, frame in enumerate(stream.decode(video=0)):
                if i not in indices:
                    continue
                row, col = divmod(indices.index(i), 7)
                left, top = col * 218, (case * 2 + row) * 146
                draw.text((left + 3, top + 3), f"{seed}   {i / 16:.3f}s", fill="black")
                page.paste(frame.to_image().resize((216, 124)), (left, top + 19))
    page.save("output/stairs.jpg", quality=92)


if __name__ == "__main__":
    main()
