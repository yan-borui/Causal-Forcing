"""Measure CUDA peaks at the existing Deep Forcing stage boundaries."""

import json
import os
import runpy
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from pipeline.causal_inference import CausalInferencePipeline
from utils.wan_wrapper import WanVAEWrapper

stage = "load_and_text"
started = time.monotonic()


def finish_stage():
    torch.cuda.synchronize()
    print(
        "MEMORY",
        json.dumps(
            dict(
                stage=stage,
                seconds=time.monotonic() - started,
                allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        ),
        flush=True,
    )


def begin(name):
    global stage, started
    finish_stage()
    torch.cuda.reset_peak_memory_stats()
    stage, started = name, time.monotonic()
    print("BEGIN", stage, flush=True)


init_cache = CausalInferencePipeline._initialize_kv_cache
decode = WanVAEWrapper.decode_to_pixel


def measured_init(self, *args, **kwargs):
    begin("dit_and_cache")
    return init_cache(self, *args, **kwargs)


def measured_decode(self, latent, *args, **kwargs):
    begin("vae")
    result = decode(self, latent, *args, **kwargs)
    begin("output")
    return result


CausalInferencePipeline._initialize_kv_cache = measured_init
WanVAEWrapper.decode_to_pixel = measured_decode
try:
    runpy.run_path("inference.py", run_name="__main__")
finally:
    finish_stage()
