"""Compare 15% and 50% capped fusion on the same development prompts and seeds."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parent
OUT = RUN / "output"
QUOTA = RUN.parent
PROJECT = QUOTA.parent
sys.path.insert(0, str(QUOTA))
sys.path.insert(0, str(PROJECT / "multiplicative-diagnostic"))
from compare import interval, load, resources, trace_scores
from score_trial import read_cache


METHOD = "capped_015"
CONTROLS = ("causal_sliding", "attention", "mixture", "capped")
PRIMARY = ("subject_pct", "musiq", "flow_top5_px")
CACHE = (
    "effective",
    "peak_share",
    "samecache_additive_effective",
    "samecache_native_effective",
)
KEY = ["id", "seed", "start"]


def old_results(prompts):
    paired = pd.read_csv(PROJECT / "sliding-baseline/output/paired-segments.csv")
    cohort = pd.DataFrame(prompts)[["id", "seed", "label"]]
    paired = paired[paired.method.isin(CONTROLS)].merge(
        cohort, on=["id", "seed", "label"], validate="many_to_one"
    )
    historical_cache = pd.read_csv(QUOTA / "paired/output/per_prompt.csv")
    historical_cache["start"] = 10 * historical_cache.window
    return paired.merge(
        historical_cache[[*KEY, "method", *CACHE]],
        on=[*KEY, "method"],
        how="left",
        validate="one_to_one",
    )


def new_results(prompts):
    rows, policies = [], []
    for prompt in prompts:
        ident, seed = prompt["id"], prompt["seed"]
        stem = f"{ident:03d}-{METHOD}"
        meta = load(OUT / "meta" / f"{stem}.json")
        previous = load(QUOTA / "paired/output/meta" / f"{ident:03d}-capped.json")
        sliding = load(PROJECT / "sliding-baseline/output/meta" / f"{seed}.json")
        audit = load(OUT / "meta" / f"{ident:03d}-pair.json")["checked"]
        earlier = load(QUOTA / "paired/output/meta" / f"{ident:03d}-pair.json")[
            "checked"
        ]
        for record in [audit[METHOD], audit["attention"], earlier["capped"]]:
            if record["seed"] != seed or not (
                record["exact_prefix"] and record["same_noise_sample"]
            ):
                raise ValueError(f"Mismatch before the first eviction: {seed}")
        if not sliding["prefix_exact"] or not (
            meta["initial_noise_sample"]
            == previous["initial_noise_sample"]
            == sliding["initial_noise_sample"]
        ):
            raise ValueError(f"Original conditions differ: {seed}")
        if (
            (meta["id"], meta["seed"], meta["method"], meta["frames"])
            != (ident, seed, METHOD, 489)
            or meta["diversity_weight"] != 0.15
            or meta["cache_tokens"] != [24960, 24960]
            or meta["old_tokens"] != [3120, 3120]
            or meta["events"] != 510
        ):
            raise ValueError(f"Wrong arm, coefficient or cache budget: {seed}")
        policy = meta["policy"]
        if policy["decisions"] != 510 or policy["cap_violations"]:
            raise ValueError(f"Incomplete frame-cap decisions: {seed}")
        policies.append(policy)
        scores = {
            kind: load(OUT / "scores" / kind / f"{stem}.json")
            for kind in ("subject", "quality", "motion")
        }
        if any(
            s["method"] != METHOD or s["id"] != ident or len(s["segments"]) != 3
            for s in scores.values()
        ):
            raise ValueError(f"Missing or mismatched scoring segments: {seed}")
        # The archived parser uses "capped" to read the common cap and source-histogram fields.
        trace = trace_scores(OUT / "traces" / f"{stem}.jsonl.gz", "capped")
        for w, cached in enumerate(trace):
            rows.append(
                dict(
                    id=ident,
                    seed=seed,
                    label=prompt["label"],
                    method=METHOD,
                    start=10 * w,
                    end=10 * w + 10,
                    subject_pct=100 * scores["subject"]["segments"][w]["subject"],
                    musiq=scores["quality"]["segments"][w]["musiq"],
                    flow_top5_px=scores["motion"]["segments"][w]["flow_top5_px"],
                    **{metric: cached[metric] for metric in CACHE},
                )
            )
    return pd.DataFrame(rows), policies


def attention_overlap(prompts):
    baseline = PROJECT / "segment-diagnostic/output"
    folders = {
        "attention": baseline,
        "mixture": baseline,
        "capped": QUOTA / "paired/output",
        METHOD: OUT,
    }
    rows = []
    for prompt in prompts:
        for method, folder in folders.items():
            trace = folder / "traces" / f"{prompt['id']:03d}-{method}.jsonl.gz"
            for window, values in enumerate(read_cache(trace)):
                rows.append(
                    dict(
                        id=prompt["id"],
                        seed=prompt["seed"],
                        method=method,
                        start=10 * window,
                        changed_from_samecache_attention_pct=values[
                            "changed_from_native_pct"
                        ],
                    )
                )
    return pd.DataFrame(rows)


def pair_differences(table, n_prompts):
    samples = np.random.default_rng(20261124).integers(
        0, n_prompts, size=(20000, n_prompts)
    )
    new = table[table.method == METHOD].set_index(KEY).sort_index()
    contrasts = []
    for control in CONTROLS:
        reference = table[table.method == control].set_index(KEY).sort_index()
        if not reference.index.equals(new.index):
            raise ValueError(f"Unpaired method {control}")
        metrics = list(PRIMARY)
        if control != "causal_sliding":
            metrics.extend(
                ("effective", "peak_share", "changed_from_samecache_attention_pct")
            )
        change = new[metrics] - reference[metrics]
        for start, window in change.groupby(level="start"):
            for metric in metrics:
                contrasts.append(
                    dict(
                        method=METHOD,
                        reference=control,
                        start=start,
                        metric=metric,
                        **interval(window[metric], samples),
                    )
                )
    return pd.DataFrame(contrasts)


def main():
    prompts = load(RUN / "prompts.json")
    if len(prompts) != 12 or prompts != load(QUOTA / "paired/prompts.json"):
        raise ValueError("The original development cohort has changed")
    new, policies = new_results(prompts)
    records = pd.concat([old_results(prompts), new], ignore_index=True)
    if (
        len(records) != len(prompts) * 3 * (1 + len(CONTROLS))
        or records.duplicated([*KEY, "method"]).any()
        or not np.isfinite(records[list(PRIMARY)].to_numpy()).all()
    ):
        raise ValueError("Incomplete scores or duplicate prompt/seed/segment rows")
    records = records.merge(
        attention_overlap(prompts),
        on=[*KEY, "method"],
        how="left",
        validate="one_to_one",
    )
    records.sort_values([*KEY, "method"]).to_csv(
        OUT / "paired-segments.csv", index=False
    )
    means = records.groupby(["start", "end", "method"])[
        [*PRIMARY, *CACHE, "changed_from_samecache_attention_pct"]
    ].mean()
    means.to_csv(OUT / "means.csv")
    pair_differences(records, len(prompts)).to_csv(
        OUT / "paired-differences.csv", index=False
    )

    atimes = {
        name: float(t)
        for t, name in (
            line.split() for line in (RUN / "phases.tsv").read_text().splitlines()
        )
    }
    prior_policy = load(QUOTA / "paired/output/summary.json")["mechanism"]["capped"]
    counts = (
        "decisions",
        "changes_to_additive",
        "additive_exceeds_native_max",
        "cap_violations",
    )
    summary = dict(
        cohort="development, first seed group",
        prompts=len(prompts),
        exact_prefix_matches=len(prompts),
        diversity_weight=0.15,
        comparison=CONTROLS,
        paired_segment_rows=len(records),
        policy_015={k: sum(p[k] for p in policies) for k in counts},
        policy_050={k: prior_policy[k] for k in counts},
        resources=resources(RUN),
        seconds_with_setup=atimes["evaluated"] - atimes["preparing"],
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
