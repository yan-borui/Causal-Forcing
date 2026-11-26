"""Use the frozen three-segment evaluators from the existing experiment."""

import json
from pathlib import Path
import sys
import time

import cv2
import torch
from evaluate_segments import model_for, score_images, score_motion


@torch.inference_mode()
def main():
    metric = sys.argv[1]
    metas = sorted(Path("output/meta").glob("*.json"))
    if len(metas) != 24:
        raise ValueError("Expected all 24 videos before scoring")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = model_for(metric)
    folder = Path("output/scores") / metric
    folder.mkdir(parents=True, exist_ok=True)
    for filename in metas:
        row = json.loads(filename.read_text())
        seed = row["seed"]
        target = folder / f"{seed}.json"
        if target.exists():
            continue
        video = Path("output/videos") / f"{seed}.mp4"
        start = time.monotonic()
        scores = (
            score_motion(model, video)
            if metric == "motion"
            else score_images(metric, model, video)
        )
        scores.update(
            id=row["id"],
            seed=seed,
            method=row["method"],
            seconds=time.monotonic() - start,
        )
        temp = target.with_suffix(".partial")
        temp.write_text(json.dumps(scores, allow_nan=False))
        temp.replace(target)
        print(metric, seed, round(scores["seconds"], 1), scores["segments"], flush=True)


if __name__ == "__main__":
    main()
