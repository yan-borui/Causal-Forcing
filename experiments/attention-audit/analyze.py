"""Summarize full-layer observation logs without treating layers as independent videos."""
import gzip
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE/'output'
SEEDS = (34001,24017,34017,34079)
NAMES = ('Workshop 34001','Stairs 24017','Stairs 34017','Cyclist 34079')
FEATURES = ('first_cos','text_cos','roi_first_cos','roi_text_cos','subject_cos','roi_subject_cos')


def begin(chunk):
    return max(0, 4*chunk-3)/16


def end(chunk):
    return 4*(chunk+2)/16


def read_jsonl(file):
    with gzip.open(file,'rt',encoding='utf-8') as f:
        return list(map(json.loads,f))


def tie_rank(values, value):
    return float(100*((values < value).sum()+0.5*(values == value).sum())/len(values))


def load(seed):
    folder = OUT/str(seed)
    case = json.loads((folder/'case.json').read_text())
    pairing = json.loads((folder/'comparison.json').read_text())
    if (pairing['frames'] != 489 or
            not pairing['metadata_match'] or not pairing['prefix_exact'] or
            pairing['archived_selections'] != pairing['matched_selections'] or
            pairing['all_layer_selections'] != 510):
        raise ValueError(f'Original/replay mismatch: {pairing}')
    obs = pd.read_csv(folder/'attention.csv')
    if len(obs)!=1230 or (obs.mass_check-1).abs().max() > 0.001:
        raise ValueError(f'Incomplete attention probabilities for {seed}')
    obs['seed'] = seed
    obs['t0_s'] = obs.chunk.map(begin)
    obs['t1_s'] = obs.chunk.map(end)
    obs['sink_above_slot_pp'] = 100*(obs.sink_mass-obs.sink_slot_share)
    features = pd.read_csv(folder/'features.csv').set_index(['layer','source'])
    trace = read_jsonl(folder/'traces'/f"{case['id']:03d}-attention.jsonl.gz")
    selection, anchor, history = [], [], []
    marginal = {(r['layer'],r['chunk']):r['mass'] for r in read_jsonl(folder/'source-attention.jsonl.gz')}
    for r in trace:
        q, layer = r['query'], r['layer']
        if (r['sink'],r['recent'],r['selected'],r['kept']) != (15600,6240,3120,24960):
            raise ValueError('Budget differs from the paired experiment')
        frames,available,selected,reference = np.array(r['frames']).T
        if not np.array_equal(selected,reference) or selected.sum()!=3120:
            raise ValueError('Native selection mismatch')
        prob = selected/3120
        selection.append(dict(seed=seed,layer=layer,chunk=q,t0_s=begin(q),
                              available_frames=len(frames),selected_effective=1/sum(prob**2),
                              peak_pct=100*max(prob)))
        # The most recent old frame is q-2 in every native eviction. Compare like ages.
        target = q-2
        block = features.loc[layer].loc[frames]
        counts = dict(zip(frames,selected))
        row = dict(seed=seed,layer=layer,chunk=q,source=target,
                   candidate_frames=len(frames),selected_share=100*counts[target]/3120)
        for col in FEATURES:
            if col in block:
                row[col] = block.loc[target,col]
                row[col+'_pct'] = tie_rank(block[col],row[col])
        anchor.append(row)
        if seed==34001 and q>=93:
            masses = marginal[layer,q]
            history.append(dict(seed=seed,layer=layer,chunk=q,t0_s=begin(q),
                                selected=int(counts.get(91,0)),read_pct=100*masses[91]))
    return obs,pd.DataFrame(selection),pd.DataFrame(anchor),pd.DataFrame(history)


def records(df):
    return json.loads(df.to_json(orient='records',double_precision=8))


def main():
    chunks = [load(seed) for seed in SEEDS]
    obs,ev,rank,trail = (pd.concat([c[i] for c in chunks],ignore_index=True) for i in range(4))
    obs.to_csv(OUT/'time-by-layer.csv',index=False)
    ev.to_csv(OUT/'selection-by-layer.csv',index=False)
    rank.to_csv(OUT/'anchor-ranks.csv',index=False)
    late = obs[obs.t0_s.between(20,30)].groupby(['seed','layer'])[
        ['entropy_normalized','effective_attended_frames','sink_mass']].mean()
    coverage = ev[ev.t0_s.between(20,30)].groupby(['seed','layer']).selected_effective.mean()
    profile = late.join(coverage).reset_index()
    profile.to_csv(OUT/'late-layer-profile.csv',index=False)
    profile['band'] = profile.layer // 10
    bands = profile.groupby(['seed','band']).mean(numeric_only=True).reset_index()
    correlation = {}
    for col in ('entropy_normalized','selected_effective'):
        aligned = profile.pivot(index='layer',columns='seed',values=col).rank()
        corr = aligned.corr()
        correlation[col] = [float(corr.loc[i,j]) for i,j in combinations(SEEDS,2)]
    stairs = obs[obs.seed.isin((24017,34017))].groupby(['seed','chunk']).agg(
        sink_pct=('sink_mass',lambda x:100*x.mean()),slot_pct=('sink_slot_share',lambda x:100*x.mean()),
        t0_s=('t0_s','first'),t1_s=('t1_s','first')).reset_index()
    stairs['change_same_phase_pp'] = stairs.groupby('seed').sink_pct.diff(2)
    stairs['change_one_chunk_pp'] = stairs.groupby('seed').sink_pct.diff()
    stairs.to_csv(OUT/'staircase-sink.csv',index=False)
    contrasts = stairs.pivot(index='chunk',columns='seed',values='change_same_phase_pp')
    contrasts['failure_minus_control_pp'] = contrasts[24017]-contrasts[34017]
    contrasts.to_csv(OUT/'staircase-paired-changes.csv')
    roi = pd.read_csv(OUT/'34001'/'workshop-region.csv')
    trail = trail.merge(roi,on=['layer','chunk'],validate='one_to_one')
    trail['chosen_pct'] = 100*trail.selected/3120
    trail.to_csv(OUT/'workshop-survival.csv',index=False)
    event=trail[trail.chunk==93].copy()
    event['roi_rate'] = event.roi_kept / event.roi_candidate
    event['outside_rate'] = (event.selected-event.roi_kept) / (event.candidate-event.roi_candidate)
    event['roi_over_outside'] = event.roi_rate/event.outside_rate
    runs = []
    for q,df in trail.groupby('chunk'):
        r = df[df.layer==0].iloc[0]
        runs.append(dict(chunk=int(q),t0_s=begin(q),layers_present=int((df.selected>0).sum()),
                         sum_selected=int(df.selected.sum()),l0_selected=int(r.selected),
                         l0_roi=int(r.roi_kept),l0_read_pct=float(r.read_pct),
                         layers_with_roi=int((df.roi_kept>0).sum())))
    metrics = [col+'_pct' for col in FEATURES]
    workshop = rank[rank.seed==34001]
    event_ranks = workshop[workshop.chunk==93]
    early = workshop[workshop.chunk<=69].groupby('chunk')[metrics].median()
    focus = event_ranks[metrics].median()
    paired = (event_ranks.set_index('layer')[metrics]-
              workshop[workshop.chunk<=69].groupby('layer')[metrics].median()).median()
    examples = []
    for label, df in [('event',event_ranks),('all_layers_early_same_age',early.reset_index())]:
        examples.append(dict(set=label, **{k:float(v) for k,v in df[metrics].median().items()}))
    summary = dict(bands=records(bands),layer_rank_correlations=correlation,
                   layer_all_cases=records(profile.groupby('layer').mean(numeric_only=True).reset_index()),
                   stairs=records(stairs[stairs.chunk>=99]),
                   pre_stairs_diff_history=records(contrasts.loc[27:108].reset_index()),
                   ranks_q93_by_case=records(rank[rank.chunk==93].groupby('seed')[metrics].median().reset_index()),
                   workshop_q93_ranks_L0=records(event_ranks[event_ranks.layer==0]),
                   workshop_q93_below10={k:int((event_ranks[k]<10).sum()) for k in metrics},
                   workshop_same_age=examples,
                   workshop_early_percentile_range={k:[float(early[k].min()),float(early[k].max())] for k in metrics},
                   workshop_median_paired_rank_difference=paired.to_dict(),
                   workshop_survival=runs,workshop_ROI_L0=records(event[event.layer==0]),
                   workshop_ROI_median=event[['roi_rate','outside_rate','roi_over_outside']].median().to_dict(),
                   workshop_ROI_enriched_layers=int((event.roi_over_outside>1).sum()),
                   full_cases=len(SEEDS), pair_checks=[json.loads((OUT/str(s)/'comparison.json').read_text())
                   | json.loads((OUT/str(s)/'pixel_alignment.json').read_text()) for s in SEEDS])
    (OUT/'findings.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
