"""Paired original-seed scores and the prespecified three-metric screening decision."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


METRICS={"subject":("subject",100),"musiq":("quality",1),"flow_top5_px":("motion",1),
         "dynamic_degree":("motion",100),"global_anchor":("subject",100)}
PRIMARY=("subject","musiq","flow_top5_px")
METHOD="multiplicative"


def load(file):
    return json.loads(file.read_text(encoding="utf-8"))


def interval(values,draws):
    values=np.array(values,dtype=float)
    low,high=np.quantile(values[draws].mean(axis=1),[0.025,0.975])
    return dict(mean=float(values.mean()),low=float(low),high=float(high),
                wins=int((values>1e-8).sum()),losses=int((values< -1e-8).sum()))


def read_cache(file):
    with gzip.open(file,"rt") as handle:
        events=[json.loads(line) for line in handle]
    if len(events)!=51 or len({(r["layer"],r["query"]) for r in events})!=51:
        raise ValueError("Incomplete trace")
    groups=[[],[],[]]
    for r in events:
        win=max(0,4*r["query"]-3)//160
        if win>=3: continue
        if (r["kept"],r["selected"],r["native"])!=(24960,3120,3120):
            raise ValueError("Wrong retention budget")
        shares=np.array(r["frames"])[:,2]/3120
        groups[win].append([100*(1-r["overlap"]/3120),1/(shares**2).sum(),100*shares.max()])
    return [dict(zip(("changed_from_native_pct","effective","peak_share"),np.mean(g,axis=0).tolist())) for g in groups]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=Path("."))
    p.add_argument("--pilot",type=Path,default=Path("../segment-pilot-20261124"))
    p.add_argument("--quota",type=Path,default=Path("../quota-study-20261124"))
    a=p.parse_args()
    out=a.run/"output"
    prompts=load(a.run/"prompts.json")
    if prompts!=load(a.pilot/"prompts.json") or prompts!=load(a.quota/"prompts.json") or len(prompts)!=12:
        raise ValueError("Unexpected original-seed comparison cohort")
    sources={"attention":a.pilot/"output","mixture":a.pilot/"output","capped":a.quota/"output",METHOD:out}
    grouped={(m,w):[] for m in sources for w in range(3)}
    docs,rows=[],[]
    for prompt in prompts:
        ident,seed=prompt["id"],prompt["seed"]
        check=load(out/"meta"/f"{ident:03d}-pair.json")["checked"]
        for method in ("attention",METHOD):
            if not check[method]["exact_prefix"] or not check[method]["same_noise_sample"] or check[method]["seed"]!=seed:
                raise ValueError("Paired state or seed mismatch")
        for m,src in sources.items():
            stem=f"{ident:03d}-{m}"
            meta=load(src/"meta"/(stem+".json"))
            if (meta["seed"],meta["frames"],meta["events"],meta["cache_tokens"],meta["old_tokens"])!=(seed,489,510,[24960,24960],[3120,3120]):
                raise ValueError(f"Cache or frame count mismatch {stem}")
            results={k:load(src/"scores"/k/(stem+".json")) for k in ("motion","subject","quality")}
            if any(r["method"]!=m or r["id"]!=ident or len(r["segments"])!=3 for r in results.values()):
                raise ValueError(f"Score identity mismatch {stem}")
            cache=read_cache(src/"traces"/(stem+".jsonl.gz"))
            if m==METHOD: docs.append(meta)
            for w in range(3):
                vals={k:s*results[kind]["segments"][w][k] for k,(kind,s) in METRICS.items()}
                vals.update(cache[w])
                if not np.isfinite(list(vals.values())).all(): raise ValueError("Nonfinite scores")
                row=dict(id=ident,seed=seed,label=prompt["label"],window=w,method=m,**vals)
                rows.append(row);grouped[m,w].append(row)
    with (out/"paired_scores.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    keys=(*METRICS,"changed_from_native_pct","effective","peak_share")
    means={m:[{k:float(np.mean([r[k] for r in grouped[m,w]])) for k in keys} for w in range(3)] for m in sources}
    samples=np.random.default_rng(20261124).integers(0,12,size=(20000,12))
    differences={}
    for control in sources.keys()-{METHOD}:
        by_window=[]
        for w in range(3):
            paired=list(zip(grouped[METHOD,w],grouped[control,w]))
            if any(a["id"]!=b["id"] or a["seed"]!=b["seed"] for a,b in paired):
                raise ValueError("Mismatched comparison")
            by_window.append({k:interval([a[k]-b[k] for a,b in paired],samples) for k in keys})
        differences[control]=by_window
    late=differences["attention"][2]
    clear=[k for k in PRIMARY if late[k]["low"]>0]
    declines=[k for k in PRIMARY if late[k]["mean"]<0]
    times={name:float(t) for t,name in (line.split() for line in (a.run/"phases.tsv").read_text().splitlines())}
    with (out/"gpu.csv").open() as stream:
        gpu=csv.reader(stream);next(gpu)
        board=max(int(r[1]) for r in gpu if len(r)==3)
    logs=[m["modulation"] for m in docs]
    total=sum(d["pre_reuse_candidates"] for d in logs)
    summary=dict(n=12,means=means,contrasts=differences,gate=dict(pass_screen=bool(clear) and not declines,
                 clear_gains=clear,declining_means=declines),execution=dict(
                 gpu_hours=(times["evaluated"]-times["generating"])/3600,
                 generation_seconds=times["generated"]-times["generating"],
                 evaluation_seconds=times["evaluated"]-times["generated"],peak_board_mib=board,
                 mean_clip_seconds=float(np.mean([m["elapsed_seconds"] for m in docs])),
                 verified_prefixes=len(docs)),modulation=dict(weight=0.15,
                 minimum_factor=min(d["minimum_factor"] for d in logs),maximum_factor=max(d["maximum_factor"] for d in logs),
                 nonpositive_factors=sum(d["nonpositive_factors"] for d in logs),
                 negative_native_pct=100*sum(d["negative_native_scores"] for d in logs)/total))
    (out/"trial_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print("Original-seed paired comparisons:",len(rows),"rows. Screening decision:",summary["gate"])


if __name__=="__main__":
    main()
