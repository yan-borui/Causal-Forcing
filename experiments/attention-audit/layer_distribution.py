"""Reduce existing all-layer observations, giving each clip the same weight."""
from pathlib import Path
import pandas as pd


OUT = Path(__file__).resolve().parent/'output'


def main():
    attention = pd.read_csv(OUT/'time-by-layer.csv')
    selected = pd.read_csv(OUT/'selection-by-layer.csv')
    summaries = []
    for window, start in [('full',0),('20-30s',20)]:
        keys = ['seed','layer']
        a = attention[attention.t0_s >= start]
        s = selected[selected.t0_s >= start]
        same_chunks = a.merge(s[keys+['chunk']],on=keys+['chunk'],validate='one_to_one')
        features = a.groupby(keys).agg(entropy_norm=('entropy_normalized','mean'),
                                        entropy_nats=('entropy_nats','mean'),
                                        read_chunks=('chunk','count'))
        features['eviction_entropy_norm'] = same_chunks.groupby(keys).entropy_normalized.mean()
        features = features.join(s.groupby(keys).agg(effective_frames=('selected_effective','mean'),
                                                     evictions=('chunk','count')))
        if len(features)!=120 or features.isna().any().any():
            raise ValueError('Expected a complete four-video by 30-layer panel')
        stats = features.groupby(level='layer').agg(
            clips=('read_chunks','count'),read_chunks_per_clip=('read_chunks','first'),
            evictions_per_clip=('evictions','first'),
            entropy_norm=('entropy_norm','mean'),entropy_norm_min=('entropy_norm','min'),
            entropy_norm_max=('entropy_norm','max'),entropy_nats=('entropy_nats','mean'),
            eviction_entropy_norm=('eviction_entropy_norm','mean'),
            effective_frames=('effective_frames','mean'),effective_min=('effective_frames','min'),
            effective_max=('effective_frames','max')).reset_index()
        if len(stats)!=30 or stats.clips.nunique()!=1 or stats.clips.iloc[0]!=4:
            raise ValueError('Unequal clip coverage')
        summaries.append(stats.assign(window=window))
    pd.concat(summaries,ignore_index=True).to_csv(OUT/'layer-distribution.csv',index=False)


if __name__=='__main__':
    main()
