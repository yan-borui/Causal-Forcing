"""Generate only the requested arms and audit them against the actual baseline latents."""
import json
import os
from pathlib import Path

import torch

from generate import generate, load_pipeline


HERE = Path.cwd()
PILOT = HERE.parent / "segment-pilot-20261124"
OUT = HERE / "output"


def read(filename):
    return json.loads(filename.read_text())


def check_pair(prompt, methods, reuse_pilot):
    sources = {m: OUT for m in methods}
    if reuse_pilot:
        sources.update(attention=PILOT / "output", mixture=PILOT / "output")
    prefix0, noise0 = None, None
    checked = {}
    for method in ("attention", *(m for m in sources if m != "attention")):
        stem = f"{prompt['id']:03d}-{method}"
        folder = sources[method]
        meta = read(folder / "meta" / f"{stem}.json")
        prefix = torch.load(folder / "prefix" / f"{stem}.pt", map_location="cpu", weights_only=True)
        if prefix0 is None:
            prefix0, noise0 = prefix, meta["initial_noise_sample"]
        checked[method] = dict(exact_prefix=torch.equal(prefix, prefix0),
                               same_noise_sample=meta["initial_noise_sample"] == noise0,
                               seed=meta["seed"], events=meta["events"])
        if meta["seed"] != prompt["seed"] or not checked[method]["exact_prefix"] or not checked[method]["same_noise_sample"]:
            raise ValueError(f"Paired initial conditions diverged: {stem}")
        if meta["events"] != 510 or meta["cache_tokens"] != [24960, 24960] or meta["old_tokens"] != [3120, 3120]:
            raise ValueError(f"Unexpected compression budget: {stem}")
    doc = dict(id=prompt["id"], seed=prompt["seed"], checked=checked)
    target = OUT / "meta" / f"{prompt['id']:03d}-pair.json"
    temp = target.with_suffix(".partial")
    temp.write_text(json.dumps(doc))
    os.replace(temp, target)
    print("PAIRED", prompt["id"], list(checked), flush=True)


def main():
    torch.set_grad_enabled(False)
    for name in ("videos", "meta", "traces", "prefix"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    methods = read(HERE / "methods.json")
    prompts = read(HERE / "prompts.json")
    pipe, low_memory = load_pipeline()
    for i, prompt in enumerate(prompts):
        start = i % len(methods)
        for method in methods[start:] + methods[:start]:
            generate(pipe, low_memory, prompt, method)
        check_pair(prompt, methods, "attention" not in methods)


if __name__ == "__main__":
    main()
