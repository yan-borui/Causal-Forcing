"""Show the new arm beside its exact baseline for the three selected original-seed cases."""

from contextlib import ExitStack
from pathlib import Path

import av
from PIL import Image, ImageDraw

from baseline_review import font, read


def main():
    review = read(Path("baseline/details.json"))
    cases = [row["case"] for row in review["cases"] if row["case"]["seed"] < 30000]
    roots = (Path("../segment-pilot-20261124/output/videos"), Path("output/videos"))
    methods = ("attention", "multiplicative")
    cols = ("Deep Forcing", "Multiplicative 0.15")
    tw, th = 360, 208
    with ExitStack() as stack:
        inputs = [
            stack.enter_context(av.open(str(root / f"{c['id']:03d}-{m}.mp4")))
            for c in cases
            for root, m in zip(roots, methods)
        ]
        dest = stack.enter_context(av.open("output/paired-failures.mp4", "w"))
        stream = dest.add_stream("libx264", rate=16)
        stream.width, stream.height, stream.pix_fmt = (
            tw * 2,
            len(cases) * (th + 26) + 28,
            "yuv420p",
        )
        stream.options = {"crf": "19", "preset": "fast"}
        count = 0
        for i, frames in enumerate(zip(*(src.decode(video=0) for src in inputs))):
            if i < 320:
                continue
            if i >= 480:
                break
            sheet = Image.new("RGB", (stream.width, stream.height), "#192a32")
            draw = ImageDraw.Draw(sheet)
            for j, frame in enumerate(frames):
                case = cases[j // 2]
                x = (j % 2) * tw
                y = (j // 2) * (th + 26)
                draw.text(
                    (x + 5, y + 4),
                    f"{case['label']}  |  {cols[j % 2]}",
                    font=font(14, True),
                    fill="white",
                )
                sheet.paste(
                    frame.to_image().resize((tw, th), Image.Resampling.BILINEAR),
                    (x, y + 26),
                )
            draw.text(
                (8, stream.height - 22),
                f"Original paired seeds    {i / 16:.2f} s",
                font=font(14),
                fill="white",
            )
            for pkt in stream.encode(av.VideoFrame.from_image(sheet)):
                dest.mux(pkt)
            count += 1
        for pkt in stream.encode():
            dest.mux(pkt)
        if count != 160:
            raise ValueError("Incomplete comparison video")


if __name__ == "__main__":
    main()
