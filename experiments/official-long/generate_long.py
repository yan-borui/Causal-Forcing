"""Run the author's EMA long-video model on both archived sets of paired noise."""
import gc
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.wan_wrapper import WanTextEncoder


RUN = Path.cwd()
ARCHIVES = ('segment-pilot-20261124', 'quota-replication-20261124')
OUT = RUN/'output'
WEIGHTS = RUN.parent.parent/'weights/chunkwise/longvideo.pt'


def cases():
    rows = []
    for name in ARCHIVES:
        archive = RUN.parent/name
        for row in json.loads((archive/'prompts.json').read_text()):
            reference = json.loads((archive/'output'/'meta'/f"{row['id']:03d}-attention.json").read_text())
            if (row['id'],row['seed']) != (reference['id'],reference['seed']):
                raise ValueError('Prompt IDs and archived seeds differ')
            rows.append({**row, 'archive':name, 'initial_noise_sample':reference['initial_noise_sample']})
    texts = {r['id']:r['prompt'] for r in rows}
    if len(rows)!=24 or len(texts)!=12 or len({r['seed'] for r in rows})!=24:
        raise ValueError('Expected 12 prompts and 24 prompt/seed pairs')
    if any(r['prompt'] != texts[r['id']] for r in rows):
        raise ValueError('Prompt strings differ between seed groups')
    return rows


class TextContexts(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, text_prompts):
        prompt, = text_prompts
        return {'prompt_embeds': self.values[prompt].to('cuda')}


def encode(prompts):
    encoder = WanTextEncoder().to(device='cuda',dtype=torch.bfloat16).eval()
    values = {text:encoder(text_prompts=[text])['prompt_embeds'].cpu() for text in prompts}
    return TextContexts(values)


def main():
    torch.set_grad_enabled(False)
    set_seed(0)
    rows = cases()
    for folder in ('videos','meta'):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    print('ENCODE', len(rows)//2, 'distinct prompts', flush=True)
    contexts = encode(dict.fromkeys(r['prompt'] for r in rows))
    gc.collect()
    torch.cuda.empty_cache()
    cfg = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'),
                          OmegaConf.load('configs/rolling_forcing_dmd.yaml'))
    pipe = CausalInferencePipeline(cfg, device=torch.device('cuda'), text_encoder=contexts)
    state = torch.load(WEIGHTS,map_location='cpu',weights_only=True,mmap=True)
    ema = {k.replace('_fsdp_wrapped_module.',''):v for k,v in state['generator_ema'].items()}
    pipe.generator.load_state_dict(ema,strict=True)
    del ema,state
    pipe.to(device='cuda',dtype=torch.bfloat16)
    original_decode = pipe.vae.decode_to_pixel
    stage = {}

    def decode(latents,*args,**kwargs):
        torch.cuda.synchronize()
        stage['dit_seconds'] = time.monotonic() - stage['start']
        stage['dit_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        pipe.kv_cache_clean = pipe.crossattn_cache = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pixels = original_decode(latents,*args,**kwargs).cpu()
        stage['vae_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        return pixels

    pipe.vae.decode_to_pixel = decode
    for row in rows:
        seed = row['seed']
        video = OUT/'videos'/f'{seed}.mp4'
        meta = OUT/'meta'/f'{seed}.json'
        if video.exists() and meta.exists():
            print('RESUME',seed,flush=True)
            continue
        print('START',seed,time.strftime('%FT%T'),flush=True)
        stage.clear()
        stage['start'] = time.monotonic()
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        noise = torch.randn((1,123,16,60,104),device='cuda',dtype=torch.bfloat16)
        sample = noise.flatten()[::100003].float().cpu().tolist()
        if sample != row['initial_noise_sample']:
            raise ValueError(f'Initial noise cross-section differs from the paired run for {seed}')
        pixels = pipe.inference_rolling_forcing(noise=noise,text_prompts=[row['prompt']],
                                                return_latents=False,initial_latent=None)
        pipe.vae.model.clear_cache()
        frames = pixels[0].permute(0,2,3,1).mul_(255).to(torch.uint8)
        if frames.shape != (489,480,832,3):
            raise ValueError(f'Unexpected output shape: {frames.shape}')
        temporary = video.with_suffix('.partial.mp4')
        write_video(str(temporary),frames,fps=16)
        temporary.replace(video)
        del frames,pixels,noise
        elapsed = time.monotonic() - stage.pop('start')
        record = {**row, 'method':'causal_official_long', 'weight_key':'generator_ema',
                  'noise_cross_section_matches':True, 'frames':489, 'fps':16,
                  'elapsed_seconds':elapsed, **stage}
        temporary = meta.with_suffix('.partial')
        temporary.write_text(json.dumps(record))
        temporary.replace(meta)
        print('DONE',seed,round(elapsed,1),'seconds',flush=True)


if __name__ == '__main__':
    main()
