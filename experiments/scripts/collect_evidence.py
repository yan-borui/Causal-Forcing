import csv
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import av


root = Path(__file__).resolve().parents[1]
run_name = sys.argv[1]
destination = root / "results" / f"evidence-{run_name}"
destination.mkdir(exist_ok=True)
raw = destination / "raw"
raw.mkdir(exist_ok=True)
prompt_file = root / "data" / "text_align.txt"
prompts = [
    line.strip()
    for line in prompt_file.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
prefixes = [prompt[:100] for prompt in prompts]
if len(prompts) != 100 or len(set(prefixes)) != 100:
    raise ValueError("Unexpected prompt set")
shutil.copyfile(prompt_file, destination / "text_align.txt")

methods = ("causal-forcing", "self-forcing")
summary = {}
scores = {}
videos = {}
ref_run = root / "runs" / "reference"
ref_console = (root / "runs" / "reference_console.log").read_text()

for group in ("reference", "generated"):
    summary[group] = {}
    for method in methods:
        if group == "reference":
            result_dir = root / "results" / group / method
            run_dir = ref_run
            times = dict(
                re.findall(rf"^{method} (start|end) (\S+)", ref_console, re.MULTILINE)
            )
            timings = {
                "evaluation_seconds": int(
                    (
                        datetime.fromisoformat(times["end"])
                        - datetime.fromisoformat(times["start"])
                    ).total_seconds()
                )
            }
        else:
            result_dir = root / "results" / group / run_name / method
            run_dir = root / "runs" / "generation" / run_name / method
            lines = (run_dir / "timing.txt").read_text().splitlines()
            times = {
                k: datetime.fromisoformat(v)
                for k, v in (
                    line.split(maxsplit=1)
                    for line in lines
                    if not line.startswith("exit ")
                )
            }
            timings = {
                f"{stage}_seconds": int(
                    (times[f"{stage}_end"] - times[f"{stage}_start"]).total_seconds()
                )
                for stage in ("generation", "evaluation")
            }
            with (run_dir / "gpu.csv").open() as fh:
                readings = list(csv.reader(fh))[1:]
            timings["sampled_peak_total_device_memory_mib"] = max(
                int(row[1])
                for row in readings
                if len(row) == 3 and row[1].strip().isdigit()
            )
        if (run_dir / "exit_status").read_text().strip() != "0":
            raise ValueError(f"Run did not finish successfully: {run_dir}")

        (result_file,) = result_dir.glob("*_eval_results.json")
        fraction, individual = json.loads(result_file.read_text())["dynamic_degree"]
        by_prompt = {Path(item["video_path"]).stem: item for item in individual}
        if len(individual) != 100 or set(by_prompt) != set(prefixes):
            raise ValueError(
                f"{group}/{method}: VBench results do not cover the requested 100 prompts"
            )
        values = [by_prompt[key]["video_results"] for key in prefixes]
        if (
            any(type(value) is not bool for value in values)
            or abs(sum(values) / len(values) - fraction) > 1e-9
        ):
            raise ValueError(
                f"{group}/{method}: per-video decisions differ from the aggregate"
            )
        scores[group, method] = values
        files = [Path(by_prompt[key]["video_path"]) for key in prefixes]
        videos[group, method] = files
        profiles = Counter()
        for clip in files:
            with av.open(str(clip)) as container:
                stream = container.streams.video[0]
                profiles[
                    f"{stream.width}x{stream.height}, {stream.frames} frames, {stream.average_rate} fps"
                ] += 1
        summary[group][method] = dict(
            dynamic_degree=fraction,
            positive=sum(values),
            count=len(values),
            video_profiles=dict(profiles),
            **timings,
        )
        shutil.copyfile(result_file, raw / f"{group}-{method}.json")

    cf = scores[group, methods[0]]
    sf = scores[group, methods[1]]
    pairs = Counter(zip(cf, sf))
    summary[group]["paired"] = {
        "both_positive": pairs[True, True],
        "only_causal": pairs[True, False],
        "only_self": pairs[False, True],
        "both_negative": pairs[False, False],
    }

with (destination / "per-prompt.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as target:
    writer = csv.writer(target)
    writer.writerow(
        [
            "prompt_id",
            "prompt",
            "reference_causal_forcing",
            "reference_self_forcing",
            "generated_causal_forcing",
            "generated_self_forcing",
        ]
    )
    for index, prompt in enumerate(prompts):
        writer.writerow(
            [index + 1, prompt, *(int(values[index]) for values in scores.values())]
        )

(destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with ZipFile(destination / "generated-200.zip", "w", compression=ZIP_STORED) as archive:
    archive.write(destination / "per-prompt.csv", "per-prompt.csv")
    for method in methods:
        for index, video in enumerate(videos["generated", method], start=1):
            archive.write(video, f"{method}/{index:03d}.mp4")

for source in (
    ref_run / "environment.txt",
    ref_run / "vbench_revision.txt",
    ref_run / "exit_status",
):
    shutil.copyfile(source, raw / ("reference-" + source.name))
for method in methods:
    run_dir = root / "runs" / "generation" / run_name / method
    for name in (
        "command.txt",
        "source_revision.txt",
        "environment.txt",
        "timing.txt",
        "gpu.csv",
        "config.yaml",
        "exit_status",
        "inference.log",
        "evaluation.log",
    ):
        shutil.copyfile(run_dir / name, raw / f"generated-{method}-{name}")
    shutil.copyfile(ref_run / f"{method}.log", raw / f"reference-{method}.log")

print(json.dumps(summary, indent=2))
