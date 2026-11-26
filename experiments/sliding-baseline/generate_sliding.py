"""Extrapolate the five-second Causal Forcing checkpoint with its native FIFO KV window."""
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch
from torchvision.io import write_video

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from generate_long import cases
from pipeline import CausalInferencePipeline
from utils.misc import set_seed


RUN = Path.cwd()
OUT = RUN / 'output'
WEIGHTS = RUN.parent.parent / 'weights/chunkwise/causal_forcing.pt'
WINDOW = 21
CONFIG = 'configs/causal_forcing_dmd_chunkwise.yaml'


def load_pipeline():
    set_seed(0)
    cfg = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'), OmegaConf.load(CONFIG))
    cfg.model_kwargs.update({'local_attn_size': WINDOW, 'sink_size': 0})
    pipe = CausalInferencePipeline(cfg, device=torch.device('cuda'))
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True, mmap=True)
    pipe.generator.load_state_dict(state['generator'], strict=True)
    del state
    pipe.to(dtype=torch.bfloat16)
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    if low_memory:
        DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    else:
        pipe.text_encoder.to(device=gpu)
    pipe.generator.to(device=gpu)
    pipe.vae.to(device=gpu)
    layers = pipe.generator.model.blocks
    if len(layers) != 30 or any((b.self_attn.local_attn_size, b.self_attn.sink_size) != (WINDOW, 0) for b in layers):
        raise ValueError('Native model did not adopt the plain sliding window')
    (OUT/'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return pipe, low_memory


def generate(pipe, low_memory, row):
    seed = row['seed']
    video = OUT/'videos'/f'{seed}.mp4'
    meta = OUT/'meta'/f'{seed}.json'
    if video.is_file() and meta.is_file():
        print('RESUME', seed, flush=True)
        return
    print('START', seed, time.strftime('%FT%T'), flush=True)
    started = time.monotonic()
    set_seed(seed)
    pipe.kv_cache1 = pipe.crossattn_cache = None
    torch.cuda.reset_peak_memory_stats()
    noise = torch.randn((1,123,16,60,104), device='cuda', dtype=torch.bfloat16)
    if noise.flatten()[::100003].float().cpu().tolist() != row['initial_noise_sample']:
        raise ValueError(f'Initial noise differs for seed {seed}')
    cache = {}
    stage = {}
    original_decode = pipe.vae.decode_to_pixel

    def record_cache(module, args, kwargs, output):
        start = int(kwargs['current_start'])
        if start in cache:
            return
        kv = kwargs['kv_cache']
        # Native indices just after the first denoising call in each new block.
        cache[start] = [[int(kv[i]['global_end_index']), int(kv[i]['local_end_index']),
                         int(kv[i]['k'].shape[1])] for i in (0,29)]

    def decode(latents, *args, **kwargs):
        torch.cuda.synchronize()
        stage['dit_seconds'] = time.monotonic() - started
        stage['dit_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        archive = RUN.parent/row['archive']/'output'/'prefix'/f"{row['id']:03d}-attention.pt"
        reference = torch.load(archive, map_location='cpu', weights_only=True)
        prefix = latents[:, :WINDOW].detach().cpu()
        diff = (prefix.float()-reference.float()).abs()
        stage['prefix_exact'] = torch.equal(prefix, reference)
        stage['prefix_max_abs'] = float(diff.max())
        stage['prefix_mean_abs'] = float(diff.mean())
        pipe.kv_cache1 = pipe.crossattn_cache = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pixels = original_decode(latents,*args,**kwargs).cpu()
        stage['vae_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        return pixels

    hook = pipe.generator.register_forward_hook(record_cache, with_kwargs=True)
    pipe.vae.decode_to_pixel = decode
    try:
        pixels = pipe.inference(noise=noise, text_prompts=[row['prompt']], low_memory=low_memory)
    finally:
        hook.remove()
        pipe.vae.decode_to_pixel = original_decode
        pipe.kv_cache1 = pipe.crossattn_cache = None
    pipe.vae.model.clear_cache()
    frames = pixels[0].permute(0,2,3,1).mul_(255).to(torch.uint8)
    if frames.shape != (489,480,832,3) or len(cache) != 41:
        raise ValueError('Video shape or number of native block updates differs')
    slots = WINDOW * pipe.frame_seq_length
    for start, boundaries in cache.items():
        end = start + 3*pipe.frame_seq_length
        if boundaries != [[end,min(end,slots),slots]] * 2:
            raise ValueError(f'Unexpected native window boundaries at token {start}')
    temp = video.with_suffix('.partial.mp4')
    write_video(str(temp), frames, fps=16)
    temp.replace(video)
    record = {**row,'method':'causal_sliding','weight_key':'generator','frames':489,'fps':16,
              'window_latents':WINDOW,'sink_latents':0,'noise_cross_section_matches':True,
              'cache_by_block':cache,'elapsed_seconds':time.monotonic()-started,**stage}
    temp = meta.with_suffix('.partial')
    temp.write_text(json.dumps(record))
    temp.replace(meta)
    print('DONE',seed,round(record['elapsed_seconds'],1),'seconds, exact prefix:',
          stage['prefix_exact'],flush=True)


def main():
    torch.set_grad_enabled(False)
    rows = cases()
    for name in ('videos','meta'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    pipe,low_memory = load_pipeline()
    for row in rows:
        generate(pipe,low_memory,row)


if __name__ == '__main__':
    main()
