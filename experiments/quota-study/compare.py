"""Compute frame-source coverage and paired VBench contrasts from actual generated clips."""
import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np


SCORES = {"subject": ("subject", 100), "global_anchor": ("subject", 100),
          "musiq": ("quality", 1), "flow_top5_px": ("motion", 1), "dynamic_degree": ("motion", 100)}
MEASURES = (*SCORES, "effective", "peak_share", "n_sources")
SPECIAL = ("capped", "two_stage")
METHODS = ("attention", "mixture", *SPECIAL)


def load(file):
    return json.loads(file.read_text(encoding="utf-8"))


def save_csv(file, rows):
    with file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def distribution(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.sum() != 3120:
        raise ValueError("Source histogram does not sum to the selection budget")
    p = counts / counts.sum()
    return np.array([1 / np.sum(p**2), 100 * p.max(), float((counts > 0).sum())])


def trace_scores(filename, method):
    with gzip.open(filename, "rt", encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    if len(events) != 51 or len({(r["layer"],r["query"]) for r in events}) != 51:
        raise ValueError(f"Incomplete trace {filename}")
    records = []
    for r in events:
        window = max(0,4*r["query"]-3) // 160
        if window >= 3:
            continue
        if (r["selected"], r["native"], r["kept"], r["sink"], r["recent"]) != (3120,3120,24960,15600,6240):
            raise ValueError(f"Unequal budget {filename}")
        actual = distribution([v[2] for v in r["frames"]])
        native = distribution([v[3] for v in r["frames"]])
        naive = distribution([v[1] for v in r["additive_frames"]]) if method in SPECIAL else None
        if method == "capped" and r["actual_max"] > r["native_cap"]:
            raise ValueError(f"Observed a cap violation {filename}")
        record = dict(window=window, effective=actual[0], peak_share=actual[1], n_sources=actual[2],
                      samecache_native_effective=native[0], samecache_native_peak=native[1])
        if naive is not None:
            record.update(samecache_additive_effective=naive[0], samecache_additive_peak=naive[1])
        records.append(record)
    per_window = []
    for w in range(3):
        selected = [r for r in records if r["window"] == w]
        per_window.append({k: float(np.mean([r[k] for r in selected])) for k in selected[0] if k != "window"})
    return per_window


def interval(d, samples):
    d = np.asarray(d, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("Missing observations in paired result")
    low, high = np.quantile(d[samples].mean(axis=1), [0.025,0.975])
    return dict(mean=float(d.mean()), low=float(low), high=float(high),
                median=float(np.median(d)), wins=int((d>1e-8).sum()), losses=int((d< -1e-8).sum()))


def resources(directory):
    lines = [s.split() for s in (directory / "phases.tsv").read_text().splitlines()]
    times = {name:float(t) for t,name in lines}
    with (directory/"output/gpu.csv").open() as stream:
        reader = csv.reader(stream)
        next(reader)
        peak = max(int(r[1]) for r in reader if len(r) == 3)
    return dict(generation_s=times["generated"]-times["generating"],
                scoring_s=times["evaluated"]-times["generated"],
                total_h=(times["evaluated"]-times["generating"])/3600, peak_board_mib=peak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("."))
    parser.add_argument("--pilot", type=Path, default=Path("../segment-pilot-20261124"))
    args = parser.parse_args()
    folder = args.run / "output"
    prompts = load(args.run / "prompts.json")
    requested = load(args.run/"methods.json")
    discovery = "attention" not in requested
    sources = {m:folder for m in requested}
    if discovery:
        sources.update(attention=args.pilot/"output", mixture=args.pilot/"output")
    methods = [m for m in METHODS if m in sources]
    if discovery and prompts != load(args.pilot/"prompts.json"):
        raise ValueError("Pilot prompts/seeds do not match")
    pooled, meta, paired = [], {m:[] for m in methods}, []
    for p in prompts:
        audit = load(folder/"meta"/f"{p['id']:03d}-pair.json")
        if set(audit["checked"]) != set(methods) or any(not a["same_noise_sample"] or not a["exact_prefix"]
                                                          for a in audit["checked"].values()):
            raise ValueError("Cross-arm audit failed")
        paired.append(audit)
        for method in methods:
            src = sources[method]
            stem = f"{p['id']:03d}-{method}"
            info = load(src/"meta"/(stem+".json"))
            if (info["id"],info["seed"],info["method"],info["frames"],info["events"]) != (p["id"],p["seed"],method,489,510):
                raise ValueError(f"Unexpected clip identity or shape: {stem}")
            if info["cache_tokens"] != [24960,24960] or info["old_tokens"] != [3120,3120]:
                raise ValueError(f"Budget mismatch: {stem}")
            if method in SPECIAL:
                pol = info["policy"]
                if pol["decisions"] != 510 or pol["cap_violations"] or pol["allocated_frames_missing"]:
                    raise ValueError(f"Failed allocation invariants: {stem}")
            meta[method].append(info)
            cache = trace_scores(src/"traces"/(stem+".jsonl.gz"), method)
            score = {k:load(src/"scores"/k/(stem+".json")) for k in ("motion","subject","quality")}
            if any(x["id"] != p["id"] or x["method"] != method or len(x["segments"]) != 3 for x in score.values()):
                raise ValueError(f"Metric cohort mismatch: {stem}")
            for w in range(3):
                row = dict(id=p["id"],label=p["label"],seed=p["seed"],window=w,method=method)
                row.update({k:scale*score[kind]["segments"][w][k] for k,(kind,scale) in SCORES.items()})
                row.update(cache[w])
                for k in ("samecache_additive_effective","samecache_additive_peak"):
                    row.setdefault(k, "")
                pooled.append(row)
    save_csv(folder/"per_prompt.csv", pooled)
    grouped = {(m,w):[r for r in pooled if r["method"] == m and r["window"] == w]
               for m,w in itertools.product(methods,range(3))}
    samples = np.random.default_rng(20261124).integers(0, len(prompts), size=(20000,len(prompts)))
    means, changes, contrast_table = {}, {}, []
    for m in methods:
        means[m] = [{k:float(np.mean([r[k] for r in grouped[m,w]])) for k in MEASURES} for w in range(3)]
    for i,m in enumerate(methods):
        for other in methods[:i]:
            name = m+"-"+other
            changes[name] = []
            for w in range(3):
                pairs = list(zip(grouped[m,w], grouped[other,w]))
                if any(a["id"]!=b["id"] or a["seed"]!=b["seed"] for a,b in pairs):
                    raise ValueError("Misaligned pairs")
                out = {k:interval([a[k]-b[k] for a,b in pairs], samples) for k in MEASURES}
                changes[name].append(out)
                contrast_table.extend(dict(method=m,reference=other,window=w,metric=k,**d) for k,d in out.items())
    save_csv(folder/"contrasts.csv", contrast_table)
    candidates = [m for m in methods if m in SPECIAL
                  and changes[m+"-mixture"][2]["effective"]["mean"] > 0
                  and changes[m+"-mixture"][2]["subject"]["mean"] > 0]
    choice = max(candidates, key=lambda m:means[m][2]["subject"]) if discovery and candidates else None
    mechanisms = {}
    for m in methods:
        if m not in SPECIAL:
            continue
        g = grouped[m,2]
        mechanisms[m] = dict(
            late_effective_minus_samecache_additive=interval([r["effective"]-r["samecache_additive_effective"] for r in g],samples),
            late_peak_minus_samecache_additive=interval([r["peak_share"]-r["samecache_additive_peak"] for r in g],samples),
            **{k:sum(r["policy"][k] for r in meta[m]) for k in meta[m][0]["policy"]})
    result = dict(stage="paired" if discovery else "replicate", methods=methods, n=len(prompts), paired=len(paired),
                  means=means, contrasts=changes, mechanism=mechanisms, replicate_choice=choice, resources=resources(args.run),
                  seconds_per_clip={m:float(np.mean([r["elapsed_seconds"] for r in meta[m]])) for m in methods})
    (folder/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("COMPARED", methods, len(pooled), "rows; independent-seed selection:", choice)


if __name__ == "__main__":
    main()
