"""Paired three-window comparison with both seeds kept together in each prompt."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE=Path(__file__).resolve().parent
OUT=BASE/'output'
PROJECT=BASE.parent
PILOT=PROJECT/'segment-diagnostic'
QUOTA=PROJECT/'quota-study'
METRICS=('subject_pct','musiq','flow_top5_px')
METHODS=('causal_official_long','attention','mixture','capped')
SLIDING=PROJECT/'sliding-baseline'/'output'


def read_one(folder, filename, method, row):
    meta=json.loads((folder/'meta'/f'{filename}.json').read_text())
    if (row['seed'],row['id'],method) != (meta['seed'],meta['id'],meta['method']):
        raise ValueError('Video identity mismatch')
    values={}
    for key in ('subject','quality','motion'):
        values[key]=json.loads((folder/'scores'/key/f'{filename}.json').read_text())
        if (values[key]['id'] != row['id'] or values[key]['method'] != method or
                len(values[key]['segments'])!=3):
            raise ValueError('Segment identity mismatch')
    if method in (METHODS[0],'causal_sliding') and (
            not meta['noise_cross_section_matches'] or meta['prompt']!=row['prompt']):
        raise ValueError('Conditioning or initial noise cross-section mismatch')
    if row['initial_noise_sample']!=meta['initial_noise_sample'] or meta['frames']!=489:
        raise ValueError('Reference noise or video length differs')
    results=[]
    for index in range(3):
        results.append(dict(id=row['id'],seed=row['seed'],label=row['label'],method=method,
                            start=index*10,end=(index+1)*10,
                            subject_pct=100*values['subject']['segments'][index]['subject'],
                            musiq=values['quality']['segments'][index]['musiq'],
                            flow_top5_px=values['motion']['segments'][index]['flow_top5_px']))
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-sliding',action='store_true',
                        help='Include the 5-second checkpoint with a plain window and write the five-method table beside that run.')
    sliding=parser.parse_args().with_sliding
    methods=(METHODS[0],'causal_sliding',*METHODS[1:]) if sliding else METHODS
    destination=SLIDING if sliding else OUT
    rows=[]
    for group, source in enumerate((PILOT,QUOTA/'replicate')):
        prompts=json.loads((source/'prompts.json').read_text(encoding='utf-8'))
        for row in prompts:
            ident,seed=row['id'],row['seed']
            reference=json.loads((source/'output'/'meta'/f'{ident:03d}-attention.json').read_text())
            row={**row,'initial_noise_sample':reference['initial_noise_sample']}
            for method in methods:
                folder=source/'output'
                filename=f'{ident:03d}-{method}'
                if method==METHODS[0]:
                    folder,filename=OUT,str(seed)
                elif method=='causal_sliding':
                    folder,filename=SLIDING,str(seed)
                elif method=='capped' and group==0:
                    folder=QUOTA/'paired'/'output'
                rows.extend(read_one(folder,filename,method,row))
    raw=pd.DataFrame(rows)
    if (len(raw)!=24*3*len(methods) or raw.duplicated(['id','seed','start','method']).any() or
            not np.isfinite(raw[list(METRICS)].to_numpy()).all()):
        raise ValueError('Incomplete case/method/window pairing')
    raw.to_csv(destination/'paired-segments.csv',index=False)
    pooled=raw.groupby(['start','end','method'])[list(METRICS)].mean().reset_index()
    pooled.to_csv(destination/'means.csv',index=False)
    by_seed=raw.assign(seed_group=raw.seed//1000).groupby(['seed_group','start','method'])[list(METRICS)].mean()
    by_seed.to_csv(destination/'means-by-seed.csv')
    wide=raw.set_index(['id','seed','label','start','method'])[list(METRICS)].unstack('method')
    comparisons=[(m,METHODS[0]) for m in methods[1:]]
    if sliding:
        comparisons.extend((m,'causal_sliding') for m in METHODS[1:])
    comparisons.extend([('capped','attention'),('mixture','attention'),('capped','mixture')])
    boot=np.random.default_rng(20261124).integers(0,12,size=(20000,12))
    differences=[]
    by_prompt=[]
    for method,reference in comparisons:
        tail=wide.xs(20,level='start')
        paired_tail=(tail.xs(method,axis=1,level='method')-tail.xs(reference,axis=1,level='method'))
        prompt_tail=paired_tail.groupby(level=['id','label']).mean().reset_index()
        by_prompt.append(prompt_tail.assign(method=method,reference=reference))
        for metric in METRICS:
            delta=wide[metric,method]-wide[metric,reference]
            for t0 in (0,10,20):
                paired=delta.xs(t0,level='start')
                prompt_means=paired.groupby(level='id').mean().sort_index()
                if len(paired)!=24 or len(prompt_means)!=12:
                    raise ValueError('Paired sample incomplete')
                low,high=np.quantile(prompt_means.to_numpy()[boot].mean(1),[0.025,0.975])
                differences.append(dict(method=method,reference=reference,start=t0,end=t0+10,
                                        metric=metric,difference=float(paired.mean()),ci_low=low,ci_high=high,
                                        higher_pairs=int((paired>0).sum()),higher_prompts=int((prompt_means>0).sum()),
                                        n_pairs=len(paired),n_prompts=len(prompt_means)))
    pd.DataFrame(differences).to_csv(destination/'paired-differences.csv',index=False)
    pd.concat(by_prompt,ignore_index=True).to_csv(destination/'tail-differences-by-prompt.csv',index=False)
    meta=[json.loads(f.read_text()) for f in sorted((destination/'meta').glob('*.json'))]
    score_seconds={}
    for kind in ('subject','quality','motion'):
        score_seconds[kind]=sum(json.loads(f.read_text())['seconds'] for f in (destination/'scores'/kind).glob('*.json'))
    summary=dict(cases=len(meta),prompt_count=raw.id.nunique(),all_noise_cross_sections_match=all(r['noise_cross_section_matches'] for r in meta),
                 generation_seconds=sum(r['elapsed_seconds'] for r in meta),score_seconds=score_seconds,
                 max_dit_allocated_gib=max(r['dit_peak_allocated_mib'] for r in meta)/1024,
                 max_vae_allocated_gib=max(r['vae_peak_allocated_mib'] for r in meta)/1024,
                 tail_means=json.loads(pooled[pooled.start==20].to_json(orient='records')))
    if sliding:
        summary['exact_21_latent_prefixes']=sum(r['prefix_exact'] for r in meta)
        summary['worst_prefix_max_abs']=max(r['prefix_max_abs'] for r in meta)
        summary['native_windows'] = sorted({(r['window_latents'],r['sink_latents']) for r in meta})
        summary['cache_evidence'] = {'native_blocks_per_clip':sorted({len(r['cache_by_block']) for r in meta}),
                                     'frames_per_clip':sorted({r['frames'] for r in meta})}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
