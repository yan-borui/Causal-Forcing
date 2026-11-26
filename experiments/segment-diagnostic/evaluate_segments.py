"""Evaluate exact 10-second segments without re-encoding their video frames."""

import argparse
import json
import os
from pathlib import Path
import time

import cv2
from decord import VideoReader
from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F
from vbench.dynamic_degree import DynamicDegree, InputPadder
from vbench.imaging_quality import transform as imaging_transform
from vbench.utils import dino_transform


OUT = Path("output")
FRAMES = 480
SEGMENT = 160
FPS = 16


def model_for(metric):
    if metric == "subject":
        model = torch.hub.load(
            str(Path("assets/dino").resolve()),
            "dino_vitb16",
            source="local",
            pretrained=False,
        )
        weights = torch.load(
            "assets/dino_vitbase16_pretrain.pth", map_location="cpu", weights_only=True
        )
        model.load_state_dict(weights, strict=True)
        return model.to("cuda").eval()
    if metric == "quality":
        from pyiqa.archs.musiq_arch import MUSIQ

        weights = "assets/musiq_spaq_ckpt-358bb6af.pth"
        return MUSIQ(pretrained_model_path=weights).to("cuda").eval()
    cfg = EasyDict(
        model="../../vbench-cache/raft_model/models/raft-things.pth",
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    return DynamicDegree(cfg, "cuda")


def score_images(metric, model, filename):
    reader = VideoReader(str(filename), num_threads=1)
    if len(reader) != 489 or abs(reader.get_avg_fps() - FPS) > 1e-4:
        raise ValueError(f"Unexpected video length/fps for {filename}")
    outputs = []
    batch_size = 16 if metric == "subject" else 4
    prep_dino = dino_transform(224)
    for start in range(0, FRAMES, batch_size):
        indices = range(start, min(start + batch_size, FRAMES))
        frames = (
            torch.from_numpy(reader.get_batch(indices).asnumpy())
            .permute(0, 3, 1, 2)
            .float()
        )
        if tuple(frames.shape[-2:]) != (480, 832):
            raise ValueError(f"Unexpected video resolution for {filename}")
        if metric == "subject":
            inputs = prep_dino(frames).to("cuda")
            outputs.append(F.normalize(model(inputs), dim=-1).cpu())
        else:
            inputs = imaging_transform(frames, "longer").to("cuda")
            outputs.append(model(inputs).reshape(-1).cpu())
    values = torch.cat(outputs, dim=0)
    if len(values) != FRAMES or not torch.isfinite(values).all():
        raise ValueError(f"Incomplete or non-finite {metric} results")
    if metric == "quality":
        return {
            "musiq": values.tolist(),
            "segments": [
                {"musiq": float(segment.mean())} for segment in values.split(SEGMENT)
            ],
        }
    anchor = (values * values[0]).sum(-1).clamp_min(0)
    out = []
    for start in range(0, FRAMES, SEGMENT):
        features = values[start : start + SEGMENT]
        preceding = (features[1:] * features[:-1]).sum(-1).clamp_min(0)
        local_first = (features[1:] * features[0]).sum(-1).clamp_min(0)
        out.append(
            dict(
                subject=float(((preceding + local_first) * 0.5).mean()),
                global_anchor=float(anchor[start : start + SEGMENT].mean()),
            )
        )
    return {"global_anchor": anchor.tolist(), "segments": out}


def score_motion(model, filename):
    capture = cv2.VideoCapture(str(filename))
    if (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 489
        or abs(capture.get(cv2.CAP_PROP_FPS) - FPS) > 1e-4
    ):
        raise ValueError(f"Unexpected RAFT video length/fps for {filename}")
    # VBench samples 8 fps and computes 20 RAFT iterations for each adjacent pair.
    stride = round(FPS / 8)
    out, all_flow = [], []
    try:
        for _ in range(3):
            scores, prev = [], None
            for i in range(SEGMENT):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Truncated video {filename}")
                if i % stride:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current = (
                    torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to("cuda")
                )
                if prev is None:
                    model.set_params(current, SEGMENT // stride)
                else:
                    padder = InputPadder(current.shape)
                    image1, image2 = padder.pad(prev, current)
                    _, flow = model.model(image1, image2, iters=20, test_mode=True)
                    scores.append(model.get_score(image1, flow))
                prev = current
            if len(scores) != 79 or not np.isfinite(scores).all():
                raise ValueError("Incomplete or non-finite optical flow")
            out.append(
                dict(
                    dynamic_degree=int(model.check_move(scores)),
                    flow_top5_px=float(np.mean(scores)),
                )
            )
            all_flow.append(scores)
    finally:
        capture.release()
    return {"flow_top5_px": all_flow, "segments": out}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=("subject", "quality", "motion"))
    metric = parser.parse_args().metric
    prompts = json.loads(Path("prompts.json").read_text(encoding="utf-8"))
    folder = OUT / "scores" / metric
    folder.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = model_for(metric)
    for row in prompts:
        for method in ("attention", "mixture"):
            stem = f"{row['id']:03d}-{method}"
            target = folder / f"{stem}.json"
            if target.exists():
                continue
            source = OUT / "videos" / f"{stem}.mp4"
            started = time.monotonic()
            scores = (
                score_motion(model, source)
                if metric == "motion"
                else score_images(metric, model, source)
            )
            scores.update(
                id=row["id"], method=method, seconds=time.monotonic() - started
            )
            partial = target.with_suffix(".partial")
            partial.write_text(json.dumps(scores, allow_nan=False), encoding="utf-8")
            os.replace(partial, target)
            print(
                metric,
                stem,
                round(scores["seconds"], 1),
                scores["segments"],
                flush=True,
            )


if __name__ == "__main__":
    main()
