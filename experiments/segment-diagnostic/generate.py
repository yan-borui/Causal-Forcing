"""Paired, single-GPU 30-second inference with the official Deep Forcing pipeline."""

import gzip
import json
import os
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch
from torchvision.io import write_video

from cache_probe import CacheProbe
from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalInferencePipeline
from utils.misc import set_seed


ROOT = Path.cwd()
OUT = ROOT / "output"
CONFIG = "configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml"
CHECKPOINT = ROOT.parent.parent / "weights/chunkwise/causal_forcing.pt"


def load_pipeline():
    set_seed(0)
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG)
    )
    cfg.model_kwargs.update(dict(is_ds_only=False, budget=16, recent=4))
    pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"))
    state = torch.load(CHECKPOINT, map_location="cpu")
    pipe.generator.load_state_dict(state["generator"], strict=True)
    del state
    pipe.to(dtype=torch.bfloat16)
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    if low_memory:
        DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    else:
        pipe.text_encoder.to(device=gpu)
    pipe.generator.to(device=gpu)
    pipe.vae.to(device=gpu)
    return pipe, low_memory


def generate(pipe, low_memory, prompt, method):
    stem = f"{prompt['id']:03d}-{method}"
    video_file = OUT / "videos" / f"{stem}.mp4"
    trace_file = OUT / "traces" / f"{stem}.jsonl.gz"
    meta_file = OUT / "meta" / f"{stem}.json"
    prefix_file = OUT / "prefix" / f"{stem}.pt"
    if all(p.is_file() for p in (video_file, trace_file, meta_file, prefix_file)):
        print("RESUME", stem, flush=True)
        return
    print("START", stem, time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    stage = {}
    start = time.perf_counter()
    set_seed(prompt["seed"])
    pipe.kv_cache1 = pipe.crossattn_cache = None
    torch.cuda.reset_peak_memory_stats()
    noise = torch.randn((1, 123, 16, 60, 104), device="cuda", dtype=torch.bfloat16)
    # Save a deterministic cross-section of the initial noise, without consuming RNG.
    noise_sample = noise.flatten()[::100003].float().cpu().tolist()
    original_decode = pipe.vae.decode_to_pixel

    def offloaded_decode(latents, *args, **kwargs):
        torch.cuda.synchronize()
        stage["predecode_seconds"] = time.perf_counter() - start
        stage["dit_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        # First eviction starts at latent 21; record the entire shared prefix.
        torch.save(latents[:, :21].detach().cpu(), prefix_file.with_suffix(".partial"))
        pipe.kv_cache1 = pipe.crossattn_cache = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pixels = original_decode(latents, *args, **kwargs).cpu()
        stage["vae_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        return pixels

    pipe.vae.decode_to_pixel = offloaded_decode
    temp_trace = trace_file.with_suffix(".partial")
    try:
        with gzip.open(temp_trace, "wt", encoding="utf-8", compresslevel=2) as stream:
            probe = CacheProbe(method, stream)
            for index, block in enumerate(pipe.generator.model.blocks):
                block.self_attn.cache_probe = probe
                block.self_attn.probe_layer = index
            pixels = pipe.inference(
                noise=noise,
                text_prompts=[prompt["prompt"]],
                return_latents=False,
                low_memory=low_memory,
            )
        pipe.vae.model.clear_cache()
        frames = pixels[0].permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
        if tuple(frames.shape) != (489, 480, 832, 3):
            raise ValueError(f"Unexpected decoded shape {tuple(frames.shape)}")
        temp_video = video_file.with_suffix(".partial.mp4")
        write_video(str(temp_video), frames, fps=16)
        os.replace(temp_video, video_file)
        os.replace(temp_trace, trace_file)
        os.replace(prefix_file.with_suffix(".partial"), prefix_file)
        meta = dict(
            id=prompt["id"],
            method=method,
            seed=prompt["seed"],
            frames=len(frames),
            elapsed_seconds=time.perf_counter() - start,
            events=probe.events,
            cache_tokens=[probe.total_kept_min, probe.total_kept_max],
            old_tokens=[probe.old_kept_min, probe.old_kept_max],
            initial_noise_sample=noise_sample,
            **stage,
        )
        temp_meta = meta_file.with_suffix(".partial")
        temp_meta.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(temp_meta, meta_file)
        print(
            "DONE",
            stem,
            round(meta["elapsed_seconds"], 1),
            "seconds",
            "cache",
            meta["cache_tokens"],
            "events",
            meta["events"],
            flush=True,
        )
    finally:
        pipe.vae.decode_to_pixel = original_decode
        pipe.kv_cache1 = pipe.crossattn_cache = None
        for block in pipe.generator.model.blocks:
            block.self_attn.cache_probe = None


def main():
    torch.set_grad_enabled(False)
    for name in ("videos", "meta", "prefix", "traces"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path("prompts.json").read_text(encoding="utf-8"))
    pipe, low_memory = load_pipeline()
    for index, prompt in enumerate(prompts):
        methods = (
            ("attention", "mixture") if index % 2 == 0 else ("mixture", "attention")
        )
        for method in methods:
            generate(pipe, low_memory, prompt, method)
        stem = f"{prompt['id']:03d}"
        a, m = [
            torch.load(
                OUT / "prefix" / f"{stem}-{x}.pt", map_location="cpu", weights_only=True
            )
            for x in ("attention", "mixture")
        ]
        records = [
            json.loads((OUT / "meta" / f"{stem}-{x}.json").read_text())
            for x in ("attention", "mixture")
        ]
        pair = dict(
            id=prompt["id"],
            exact_prefix=torch.equal(a, m),
            prefix_max_abs=(a.float() - m.float()).abs().max().item(),
            same_initial_noise_sample=records[0]["initial_noise_sample"]
            == records[1]["initial_noise_sample"],
        )
        (OUT / "meta" / f"{stem}-pair.json").write_text(json.dumps(pair))
        print("PAIR", json.dumps(pair), flush=True)


if __name__ == "__main__":
    main()
