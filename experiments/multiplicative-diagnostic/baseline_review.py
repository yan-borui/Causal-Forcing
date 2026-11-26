"""CPU-only ranking, aligned source-frame statistics and visual review of existing baseline videos."""

import argparse
import csv
from functools import lru_cache
import gzip
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LAYERS = (0, 14, 29)
ATTRS = (
    "effective",
    "peak_pct",
    "selection_entropy_bits",
    "mean_age_s",
    "older_than_5s_pct",
)


def read(file):
    return json.loads(file.read_text(encoding="utf-8"))


def table(file, rows):
    with file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def source_time(frame):
    return (max(0, 4 * frame - 3) + 4 * frame) / 32


def cache_events(file, ident, seed):
    out = []
    with gzip.open(file, "rt") as handle:
        for line in handle:
            event = json.loads(line)
            if (event["sink"], event["recent"], event["selected"], event["kept"]) != (
                15600,
                6240,
                3120,
                24960,
            ):
                raise ValueError("Unexpected cache budget")
            decision = max(0, 4 * event["query"] - 3) / 16
            counts = np.array(event["frames"], dtype=int)
            if (
                not np.array_equal(counts[:, 2], counts[:, 3])
                or counts[:, 2].sum() != 3120
            ):
                raise ValueError("Trace is inconsistent with the attention baseline")
            retained = counts[counts[:, 2] > 0]
            ranked = sorted(retained, key=lambda row: (-row[2], row[0]))
            p = retained[:, 2] / 3120
            age = decision - np.array([source_time(int(f)) for f in retained[:, 0]])
            entry = dict(
                id=ident,
                seed=seed,
                layer=event["layer"],
                query=event["query"],
                decision_s=decision,
                available_source_frames=len(counts),
                retained_source_frames=len(retained),
                effective=float(1 / np.dot(p, p)),
                peak_pct=float(100 * p.max()),
                selection_entropy_bits=float(-np.dot(p, np.log2(p))),
                mean_age_s=float(np.dot(p, age)),
                older_than_5s_pct=float(100 * p[age >= 5].sum()),
            )
            for i, (fid, available, selected, _) in enumerate(ranked[:3], 1):
                entry[f"source_{i}_s"] = source_time(int(fid))
                entry[f"source_{i}_pct"] = 100 * int(selected) / 3120
            out.append(entry)
    if len(out) != 51 or len({(r["layer"], r["query"]) for r in out}) != 51:
        raise ValueError("Incomplete attention-selection trace")
    return out


def collect(original, replica):
    cases, events = [], []
    arrays, videos = {}, {}
    original_prompts = read(original / "prompts.json")
    for directory in (original, replica):
        prompts = read(directory / "prompts.json")
        if [(p["id"], p["prompt"]) for p in prompts] != [
            (p["id"], p["prompt"]) for p in original_prompts
        ]:
            raise ValueError("Unequal prompt cohorts")
        folder = directory / "output"
        for prompt in prompts:
            ident, seed = prompt["id"], prompt["seed"]
            stem = f"{ident:03d}-attention"
            metrics = {
                k: read(folder / "scores" / k / (stem + ".json"))
                for k in ("motion", "subject", "quality")
            }
            if any(
                r["method"] != "attention" or r["id"] != ident for r in metrics.values()
            ):
                raise ValueError("Baseline metric identity mismatch")
            q = np.array(metrics["quality"]["musiq"]).reshape(30, 16).mean(axis=1)
            anchor = 100 * np.array(metrics["subject"]["global_anchor"]).reshape(
                30, 16
            ).mean(axis=1)
            flows = np.array(metrics["motion"]["flow_top5_px"]).reshape(3, 79)
            by_second = []
            for second in range(30):
                end_frames = 2 * np.arange(1, 80)
                group = (end_frames // 16) == second % 10
                by_second.append(float(flows[second // 10, group].mean()))
            arrays[seed] = dict(
                musiq=q.tolist(), global_anchor=anchor.tolist(), flow=by_second
            )
            worst = 20 + int(np.argmin(q[20:]))
            row = dict(id=ident, seed=seed, label=prompt["label"])
            for w in range(3):
                row[f"subject{w}"] = 100 * metrics["subject"]["segments"][w]["subject"]
                row[f"musiq{w}"] = metrics["quality"]["segments"][w]["musiq"]
                row[f"flow{w}"] = metrics["motion"]["segments"][w]["flow_top5_px"]
                row[f"anchor{w}"] = (
                    100 * metrics["subject"]["segments"][w]["global_anchor"]
                )
            row.update(worst_quality_second=worst, worst_second_musiq=float(q[worst]))
            cases.append(row)
            events.extend(
                cache_events(folder / "traces" / (stem + ".jsonl.gz"), ident, seed)
            )
            videos[seed] = folder / "videos" / (stem + ".mp4")
    for k in ("subject", "musiq"):
        for rank, row in enumerate(
            sorted(cases, key=lambda r: (r[k + "2"], r["id"], r["seed"])), 1
        ):
            row[k + "_rank"] = rank
    ordered = sorted(
        cases,
        key=lambda r: (
            r["subject_rank"] + r["musiq_rank"],
            max(r["subject_rank"], r["musiq_rank"]),
            r["id"],
            r["seed"],
        ),
    )
    selected = [r for r in ordered if min(r["subject_rank"], r["musiq_rank"]) <= 2]
    for row in ordered:
        if len(selected) == 5:
            break
        if row not in selected:
            selected.append(row)
    return ordered, selected, events, arrays, videos


def late_layers(events, seeds):
    result = {}
    for layer in LAYERS:
        case_means = []
        for seed in seeds:
            selected = [
                r
                for r in events
                if r["seed"] == seed
                and r["layer"] == layer
                and 20 <= r["decision_s"] < 30
            ]
            case_means.append([np.mean([r[k] for r in selected]) for k in ATTRS])
        result[layer] = dict(zip(ATTRS, np.mean(case_means, axis=0).tolist()))
    return result


@lru_cache()
def font(size, bold=False):
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("plot-deps/matplotlib/mpl-data/fonts/ttf") / family), size
    )


def pictures(video, indices, size):
    images = {}
    with av.open(str(video)) as stream:
        for i, frame in enumerate(stream.decode(video=0)):
            if i in indices:
                images[i] = frame.to_image().resize(size, Image.Resampling.LANCZOS)
                if len(images) == len(indices):
                    break
    if len(images) != len(indices):
        raise ValueError(f"Missing video frames: {video}")
    return images


def contact(case, aligned, video, destination):
    seed = case["seed"]
    times = [0, 4, 10, 15, 18] + [20 + k / 2 for k in range(20)]
    indices = [int(t * 16) for t in times]
    tw, th = 235, 136
    cols = 5
    pics = pictures(video, set(indices), (tw, th))
    sheet = Image.new("RGB", (cols * (tw + 6) + 10, 52 + 5 * (th + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"{case['label']}   |   seed {seed}   |   original attention baseline",
        font=font(18, True),
        fill="#213a44",
    )
    for pos, (i, t) in enumerate(zip(indices, times)):
        x = 6 + (pos % cols) * (tw + 6)
        y = 40 + (pos // cols) * (th + 24)
        sheet.paste(pics[i], (x, y))
        draw.text((x, y + th + 2), f"{t:g} s", font=font(13), fill="#253942")
    sheet.save(destination / f"{seed}-sequence.jpg", quality=93)
    rows = sorted(aligned, key=lambda r: r["layer"])
    worst = case["worst_quality_second"]
    event = rows[0]["decision_s"]
    grid = [
        [int(event * 16)] + [round(r[f"source_{n}_s"] * 16) for n in (1, 2, 3)]
        for r in rows
    ]
    indices = {i for group in grid for i in group} | {16 * worst + 8}
    pics = pictures(video, indices, (tw, th))
    sheet = Image.new("RGB", (5 * (tw + 6) + 12, 63 + 3 * (th + 38)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 7),
        f"{case['label']}  |  seed {seed}  |  lowest-MUSIQ second {worst}–{worst + 1} s",
        font=font(17, True),
        fill="#213a44",
    )
    for j, n in enumerate(
        (
            "Target +0.5 s",
            "Preceding eviction",
            "Top source 1",
            "Top source 2",
            "Top source 3",
        )
    ):
        draw.text((7 + j * (tw + 6), 35), n, font=font(14, True), fill="#45636f")
    for pos, (r, group) in enumerate(zip(rows, grid)):
        y = 59 + pos * (th + 38)
        for j, i in enumerate([16 * worst + 8] + group):
            x = 6 + j * (tw + 6)
            sheet.paste(pics[i], (x, y))
            if j == 0:
                label = f"Layer {r['layer']}  |  {i / 16:.2f} s"
            elif j == 1:
                label = f"{event:.2f} s  |  eff. {r['effective']:.1f}"
            else:
                label = f"{r[f'source_{j - 1}_s']:.2f} s  |  {r[f'source_{j - 1}_pct']:.1f}%"
            draw.text((x, y + th + 3), label, font=font(13), fill="#253942")
    sheet.save(destination / f"{seed}-sources.jpg", quality=93)


def comparison_video(cases, videos, dest):
    import contextlib

    tw, th = 352, 204
    with contextlib.ExitStack() as stack:
        inputs = [stack.enter_context(av.open(str(videos[r["seed"]]))) for r in cases]
        out = stack.enter_context(av.open(str(dest), "w"))
        enc = out.add_stream("libx264", rate=16)
        enc.width, enc.height, enc.pix_fmt = tw * 3, (th + 25) * 2 + 26, "yuv420p"
        enc.options = {"crf": "19", "preset": "fast"}
        n = 0
        for i, frames in enumerate(zip(*(s.decode(video=0) for s in inputs))):
            if i < 320:
                continue
            if i >= 480:
                break
            page = Image.new("RGB", (enc.width, enc.height), "#182a32")
            draw = ImageDraw.Draw(page)
            for j, (case, frame) in enumerate(zip(cases, frames)):
                x = tw * (j % 3) + (tw // 2 if j >= 3 else 0)
                y = (j // 3) * (th + 25)
                draw.text(
                    (x + 6, y + 3),
                    f"{case['label']}   /   {case['seed']}",
                    font=font(14, True),
                    fill="white",
                )
                page.paste(
                    frame.to_image().resize((tw, th), Image.Resampling.BILINEAR),
                    (x, y + 25),
                )
            draw.text(
                (8, enc.height - 23),
                f"Deep Forcing baseline   |   {i / 16:.2f} s",
                font=font(14),
                fill="white",
            )
            for packet in enc.encode(av.VideoFrame.from_image(page)):
                out.mux(packet)
            n += 1
        for packet in enc.encode():
            out.mux(packet)
        if n != 160:
            raise ValueError("Incomplete baseline review video")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, default=Path("../segment-pilot-20261124"))
    p.add_argument(
        "--replica", type=Path, default=Path("../quota-replication-20261124")
    )
    args = p.parse_args()
    dest = Path("baseline")
    dest.mkdir(exist_ok=True)
    ordered, selected, events, arrays, videos = collect(args.original, args.replica)
    selected_ids = [r["seed"] for r in selected]
    for r in ordered:
        r["selected"] = int(r["seed"] in selected_ids)
    table(dest / "ranking.csv", ordered)
    table(dest / "cache-events.csv", events)
    groups = {
        "selected": selected_ids,
        "other": [r["seed"] for r in ordered if not r["selected"]],
        "all": [r["seed"] for r in ordered],
    }
    details = dict(
        layer_averages={k: late_layers(events, ss) for k, ss in groups.items()},
        cases=[],
    )
    for case in selected:
        seed = case["seed"]
        t = case["worst_quality_second"] + 0.5
        when = max(
            r["decision_s"]
            for r in events
            if r["seed"] == seed and r["decision_s"] <= t
        )
        matched = [r for r in events if r["seed"] == seed and r["decision_s"] == when]
        other = next(r for r in ordered if r["id"] == case["id"] and r["seed"] != seed)
        details["cases"].append(
            dict(
                case=case,
                other_seed=other,
                source_event_s=when,
                aligned=matched,
                seconds=arrays[seed],
                late_layers=late_layers(events, [seed]),
            )
        )
        contact(case, matched, videos[seed], dest)
    (dest / "details.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    comparison_video(selected, videos, dest / "five-late.mp4")
    print(
        "Reviewed source data for",
        len(ordered),
        "clips and",
        len(events),
        "layer-eviction records. Selected:",
        selected_ids,
    )


if __name__ == "__main__":
    main()
