"""Pair the archived and instrumented baseline trajectories on their full outputs."""
import gzip
import json
from pathlib import Path

import av
import numpy as np
import torch
from examples import baseline, SEEDS


def events(file):
    with gzip.open(file, 'rt') as stream:
        return {(r['layer'], r['query']): r for r in map(json.loads, stream)}


def compare(seed):
    old = baseline(seed)
    new = Path('output') / str(seed)
    stem = f'{seed%1000:03d}-attention'
    metas = [json.loads((run/'meta'/f'{stem}.json').read_text()) for run in (old, new)]
    meta_keys = ('id','seed','frames','events','cache_tokens','old_tokens','initial_noise_sample')
    meta_match = all(metas[0][k] == metas[1][k] for k in meta_keys)
    prefix = [torch.load(run/'prefix'/f'{stem}.pt',map_location='cpu', weights_only=True)
              for run in (old, new)]
    old_events, new_events = [events(run/'traces'/f'{stem}.jsonl.gz') for run in (old,new)]
    old_match = sum(v == new_events.get(k) for k, v in old_events.items())
    same = count = max_abs = absolute_error = 0
    first_mismatch = None
    with av.open(str(old/'videos'/f'{stem}.mp4')) as a, av.open(str(new/'videos'/f'{stem}.mp4')) as b:
        for fa,fb in zip(a.decode(video=0),b.decode(video=0),strict=True):
            x,y = [frame.to_ndarray(format='rgb24') for frame in (fa,fb)]
            if np.array_equal(x,y):
                same += 1
            else:
                delta = np.abs(x.astype('int16') - y.astype('int16'))
                max_abs = max(max_abs, int(delta.max()))
                absolute_error += int(delta.sum())
                if first_mismatch is None:
                    first_mismatch = count
            count += 1
    result = dict(seed=seed,frames=count,identical_frames=same,first_mismatch=first_mismatch,
                  max_abs_pixel=max_abs,mean_abs_pixel=absolute_error/(count*480*832*3),
                  metadata_match=meta_match,prefix_exact=torch.equal(*prefix),
                  archived_selections=len(old_events),matched_selections=old_match,
                  all_layer_selections=len(new_events))
    (new/'comparison.json').write_text(json.dumps(result, indent=2))
    print('PAIR', json.dumps(result), flush=True)


if __name__ == '__main__':
    for case in SEEDS:
        compare(case)
