"""Summarise paired outputs and counterfactual token selection in each ten-second window."""

import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path

import numpy as np


OUT = Path("output")
METHODS = ("attention", "mixture")
METRICS = {
    "subject": ("subject", 100),
    "global_anchor": ("subject", 100),
    "musiq": ("quality", 1),
    "flow_top5_px": ("motion", 1),
    "dynamic_degree": ("motion", 100),
}
TRACE_MEASURES = (
    "replaced_pct",
    "anchor_change_pp",
    "musiq_change",
    "source_time_change_s",
    "effective_sources_actual",
    "effective_sources_native",
    "effective_sources_delta",
    "dominant_source_share_delta_pp",
    "low_anchor_delta_pp",
    "low_quality_delta_pp",
    "low_both_delta_pp",
)
BOOTSTRAPS = 20000


def read_json(filename):
    return json.loads(filename.read_text(encoding="utf-8"))


def write_rows(filename, rows):
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def moments(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != draws.shape[1] or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite observations in a paired summary")
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    pos = int((values > 1e-8).sum())
    neg = int((values < -1e-8).sum())
    return dict(
        mean=float(values.mean()),
        low=float(low),
        high=float(high),
        positive=pos,
        tied=len(values) - pos - neg,
        negative=neg,
    )


def tagged(stem, metric):
    return read_json(OUT / "scores" / metric / (stem + ".json"))


def machine_record():
    with (OUT / "generation-gpu.csv").open(encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        observed = [int(r[1]) for r in reader if len(r) == 3]
    start, gen_done, eval_done = [
        Path(n).stat().st_mtime
        for n in ("generation.pid", "generation.status", "assessment.status")
    ]
    tz = timezone(timedelta(hours=8))
    return dict(
        start=datetime.fromtimestamp(start, tz).isoformat(timespec="seconds"),
        generation_finished=datetime.fromtimestamp(gen_done, tz).isoformat(
            timespec="seconds"
        ),
        evaluation_finished=datetime.fromtimestamp(eval_done, tz).isoformat(
            timespec="seconds"
        ),
        generation_elapsed_seconds=gen_done - start,
        evaluation_elapsed_seconds_including_handoff=eval_done - gen_done,
        elapsed_gpu_hours=(eval_done - start) / 3600,
        generation_board_peak_mib=max(observed),
        gpu_samples=len(observed),
    )


def read_trace(stem):
    with gzip.open(
        OUT / "traces" / (stem + ".jsonl.gz"), "rt", encoding="utf-8"
    ) as stream:
        return [json.loads(line) for line in stream]


def pixel_bounds(frame):
    return max(0, 4 * frame - 3), 4 * frame + 1


def cache_diagnostics(stem, prompt_id, pixel_scores):
    trace = read_trace(stem)
    if len(trace) != 51 or len({(r["query"], r["layer"]) for r in trace}) != 51:
        raise ValueError(f"Incomplete cache trace {stem}")
    # Only evictions within the 480 rendered evaluation frames enter this analysis.
    events = []
    frame_profiles = {}
    for event in trace:
        q0, _ = pixel_bounds(event["query"])
        window = (q0 // 16) // 10
        if window >= 3:
            continue
        slots, shared = event["selected"], event["overlap"]
        if not (
            slots == event["native"] == 3120
            and event["kept"] == 24960
            and event["sink"] == 15600
            and event["recent"] == 6240
            and 0 <= shared <= slots
        ):
            raise ValueError(f"Inconsistent token accounting {stem}")
        tallies = event["frames"]
        if (
            sum(row[2] for row in tallies) != slots
            or sum(row[3] for row in tallies) != slots
        ):
            raise ValueError(f"Incomplete source counts {stem}")
        for frame, _, _, _ in tallies:
            if frame not in frame_profiles:
                start, stop = pixel_bounds(frame)
                if stop > 480:
                    raise ValueError("A source frame falls outside the scored range")
                frame_profiles[frame] = np.array(
                    [
                        100 * np.mean(pixel_scores["anchor"][start:stop]),
                        np.mean(pixel_scores["quality"][start:stop]),
                        (start + stop - 1) / 32,
                    ]
                )
        profiles = np.stack([frame_profiles[r[0]] for r in tallies])
        selected_counts = np.array([[r[2], r[3]] for r in tallies], dtype=float)
        counts_delta = selected_counts[:, 0] - selected_counts[:, 1]
        changes = counts_delta @ profiles
        low = profiles[:, :2] <= np.quantile(profiles[:, :2], 0.25, axis=0)
        low_anchor, low_quality = counts_delta @ low
        low_both = counts_delta @ low.all(axis=1)
        shares = selected_counts / slots
        effective = 1 / (shares**2).sum(axis=0)
        peak = shares.max(axis=0) * 100
        events.append(
            dict(
                window=window,
                layer=event["layer"],
                slots=slots,
                changed=slots - shared,
                anchor=changes[0],
                quality=changes[1],
                source_time=changes[2],
                effective_actual=effective[0],
                effective_native=effective[1],
                peak_delta=peak[0] - peak[1],
                low_anchor=low_anchor,
                low_quality=low_quality,
                low_both=low_both,
            )
        )

    def aggregate(window, activity):
        slots = sum(r["slots"] for r in activity)
        changed = sum(r["changed"] for r in activity)
        if not changed or not slots:
            raise ValueError(
                f"No comparable replacements in window {window} for {stem}"
            )
        a, n = [
            float(np.mean([r["effective_" + k] for r in activity]))
            for k in ("actual", "native")
        ]
        return dict(
            id=prompt_id,
            window=window,
            events=len(activity),
            slots=slots,
            replacements=changed,
            replaced_pct=100 * changed / slots,
            anchor_change_pp=sum(r["anchor"] for r in activity) / changed,
            musiq_change=sum(r["quality"] for r in activity) / changed,
            source_time_change_s=sum(r["source_time"] for r in activity) / changed,
            effective_sources_actual=a,
            effective_sources_native=n,
            effective_sources_delta=a - n,
            dominant_source_share_delta_pp=np.mean([r["peak_delta"] for r in activity]),
            low_anchor_delta_pp=100 * sum(r["low_anchor"] for r in activity) / changed,
            low_quality_delta_pp=100
            * sum(r["low_quality"] for r in activity)
            / changed,
            low_both_delta_pp=100 * sum(r["low_both"] for r in activity) / changed,
        )

    rows, per_layer = [], []
    for window in range(3):
        activity = [r for r in events if r["window"] == window]
        rows.append(aggregate(window, activity))
        for layer in sorted({r["layer"] for r in activity}):
            per_layer.append(
                dict(
                    layer=layer,
                    **aggregate(window, [r for r in activity if r["layer"] == layer]),
                )
            )
    return rows, per_layer


def main():
    prompts = read_json(Path("prompts.json"))
    draws = np.random.default_rng(20261124).integers(
        0, len(prompts), size=(BOOTSTRAPS, len(prompts))
    )
    rows, trace_rows, layer_rows, generation, paired = [], [], [], [], []
    for prompt in prompts:
        ident, label = prompt["id"], prompt["label"]
        scores = {}
        for method in METHODS:
            stem = f"{ident:03d}-{method}"
            generation.append(read_json(OUT / "meta" / (stem + ".json")))
            scores[method] = {
                metric: tagged(stem, metric)
                for metric in ("subject", "quality", "motion")
            }
            if any(
                doc["method"] != method
                or doc["id"] != ident
                or len(doc["segments"]) != 3
                for doc in scores[method].values()
            ):
                raise ValueError(f"Unexpected metric metadata {stem}")
        paired.append(read_json(OUT / "meta" / f"{ident:03d}-pair.json"))
        for window in range(3):
            row = dict(
                id=ident,
                label=label,
                seed=prompt["seed"],
                window=window,
                start_s=10 * window,
                end_s=10 * (window + 1),
            )
            for measure, (metric, scale) in METRICS.items():
                a, m = [
                    scale * scores[method][metric]["segments"][window][measure]
                    for method in METHODS
                ]
                row.update(
                    {
                        f"{measure}_attention": a,
                        f"{measure}_mixture": m,
                        f"{measure}_delta": m - a,
                    }
                )
            rows.append(row)
        mix = scores["mixture"]
        all_layers, separate_layers = cache_diagnostics(
            f"{ident:03d}-mixture",
            ident,
            {
                "anchor": mix["subject"]["global_anchor"],
                "quality": mix["quality"]["musiq"],
            },
        )
        trace_rows.extend(all_layers)
        layer_rows.extend(separate_layers)

    for m in generation:
        if (
            m["frames"] != 489
            or m["events"] != 510
            or m["cache_tokens"] != [24960, 24960]
            or m["old_tokens"] != [3120, 3120]
        ):
            raise ValueError(f"Generation/count mismatch: {m['id']} {m['method']}")
    if not all(
        p["exact_prefix"]
        and p["same_initial_noise_sample"]
        and p["prefix_max_abs"] == 0
        for p in paired
    ):
        raise ValueError("Initial condition mismatch")
    write_rows(OUT / "paired_segments.csv", rows)
    write_rows(OUT / "retention_segments.csv", trace_rows)
    write_rows(OUT / "retention_by_layer.csv", layer_rows)
    mean_by_window, time_interaction, cache_results = [], {}, []
    for window in range(3):
        sample = [row for row in rows if row["window"] == window]
        part = {}
        for measure in METRICS:
            part[measure] = {
                k: float(np.mean([r[f"{measure}_{k}"] for r in sample]))
                for k in METHODS
            }
            part[measure]["delta"] = moments(
                [r[f"{measure}_delta"] for r in sample], draws
            )
        mean_by_window.append(part)
        cache_results.append(
            {
                m: moments([r[m] for r in trace_rows if r["window"] == window], draws)
                for m in TRACE_MEASURES
            }
        )
    for measure in METRICS:
        d = {
            w: np.array([r[f"{measure}_delta"] for r in rows if r["window"] == w])
            for w in (0, 2)
        }
        time_interaction[measure] = moments(d[2] - d[0], draws)
    final = sorted(
        (r for r in rows if r["window"] == 2), key=lambda r: r["subject_delta"]
    )
    examples = {"lowest": final[0]["id"], "highest": final[-1]["id"]}
    runtime = {}
    for method in METHODS:
        arm = [m for m in generation if m["method"] == method]
        runtime[method] = {
            "mean_seconds": float(np.mean([m["elapsed_seconds"] for m in arm])),
            "elapsed_seconds_sum": sum(m["elapsed_seconds"] for m in arm),
            "dit_peak_mib_max": max(m["dit_peak_mib"] for m in arm),
            "vae_peak_mib_max": max(m["vae_peak_mib"] for m in arm),
        }
    evaluation_seconds = {
        metric: sum(
            tagged(f"{p['id']:03d}-{m}", metric)["seconds"]
            for p in prompts
            for m in METHODS
        )
        for metric in ("motion", "quality", "subject")
    }
    summary = dict(
        n_prompts=len(prompts),
        pairs_prefix_exact=len(paired),
        windows=mean_by_window,
        late_minus_early=time_interaction,
        retention=cache_results,
        examples=examples,
        generate=runtime,
        evaluation_seconds=evaluation_seconds,
        machine=machine_record(),
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("Paired analysis complete:", len(rows), "prompt-segments;", examples)


if __name__ == "__main__":
    main()
