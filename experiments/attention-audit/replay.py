"""Read-only diagnostics on the exact four previous Deep Forcing trajectories."""

import csv
import gzip
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

import cache_probe
import generate
from examples import baseline, SEEDS
from wan.modules import causal_model, model


SPATIAL = 30 * 52
SELF_POINTS = [y * 52 + x for y in (7, 22) for x in (6, 19, 32, 45)]
TEXT_POINTS = [
    y * 52 + x for y in (2, 7, 12, 17, 22, 27) for x in (3, 9, 16, 22, 29, 35, 42, 48)
]
# Tightly encloses the additional figure at decoded frame 363; source 91 also spans frames 361–364.
ROI = [y * 52 + x for y in range(9, 18) for x in range(28, 34)]
SOURCE = 91
BLOCK = 3


def csv_out(file, fields, rows):
    with file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


class Probe(cache_probe.CacheProbe):
    def record(
        self,
        layer,
        query_frame,
        spatial_size,
        origins,
        attention,
        counts,
        reuse,
        sink,
        recent,
        top_c,
        kept,
        selected,
        frame_info,
        selector,
    ):
        super().record(
            layer,
            query_frame,
            spatial_size,
            origins,
            attention,
            counts,
            reuse,
            sink,
            recent,
            top_c,
            kept,
            selected,
            frame_info,
            selector,
        )
        if audit.seed == 34001 and query_frame >= 93:
            candidate = origins[sink : len(origins) - recent]
            inside = candidate[candidate // spatial_size == SOURCE] % spatial_size
            chosen = origins[selected]
            chosen = chosen[chosen // spatial_size == SOURCE] % spatial_size
            in_roi = lambda patches: int(audit.roi_mask[patches].sum().item())
            audit.roi_rows.append(
                [
                    layer,
                    query_frame,
                    len(inside),
                    len(chosen),
                    in_roi(inside),
                    in_roi(chosen),
                ]
            )
            if layer == 0 and query_frame == 93:
                audit.patches = dict(
                    candidate=inside.cpu().tolist(), kept=chosen.cpu().tolist()
                )


class Audit:
    def reset(self, seed, tokens, subject_indices, directory):
        self.seed = seed
        self.subject_indices = subject_indices
        self.text_indices = torch.tensor(tokens, device="cuda")
        self.text_points = torch.tensor(TEXT_POINTS + ROI, device="cuda")
        self.query_points = torch.tensor(SELF_POINTS, device="cuda")
        self.roi_indices = torch.tensor(ROI, device="cuda")
        self.roi_mask = torch.zeros(SPATIAL, device="cuda", dtype=torch.bool)
        self.roi_mask[self.roi_indices] = True
        self.directory = directory
        self.chunk_counts = {}
        self.clean = False
        self.first = {}
        self.self_features = {}
        self.cross_features = {}
        self.attention_rows = []
        self.source_rows = []
        self.roi_rows = []
        self.patches = None

    def generator_call(self, module, args, kwargs):
        self.chunk = kwargs["current_start"] // SPATIAL
        count = self.chunk_counts.get(self.chunk, 0) + 1
        self.chunk_counts[self.chunk] = count
        self.clean = count == 1 + self.denoising_steps
        if self.clean and kwargs["timestep"][0, 0].item() != self.context_noise:
            raise ValueError("Unexpected timestep on the clean cache refresh")

    def self_readout(self, q, keys, native):
        layer = native["self"].probe_layer
        start = native["current_start_frame"]
        if start != self.chunk or native["frame_seqlen"] != SPATIAL:
            raise ValueError("Sample and native self-attention coordinates differ")
        raw_keys = native["k"][0].reshape(BLOCK, SPATIAL, *q.shape[2:])
        raw_mean = raw_keys.mean(dim=1, dtype=torch.float32)
        roi_mean = raw_keys.index_select(1, self.roi_indices).mean(
            dim=1, dtype=torch.float32
        )
        pooled = F.normalize(torch.stack((raw_mean, roi_mean), 1), dim=-1)
        if start == 0:
            self.first[layer] = pooled[0].clone()
        first_cos = (pooled * self.first[layer]).sum(-1).mean(-1)
        for offset, values in enumerate(first_cos.cpu().tolist()):
            self.self_features[layer, start + offset] = values

        origins = native["kv_cache"]["source_token"][0]
        if native["rolled_condition"]:
            tail = origins[native["tail_start"] : native["tail_end"]]
            nsink = native["sink_len_tokens"]
            origins = (
                torch.cat((origins[:nsink], tail))
                if nsink and native["self"].sink_size
                else tail
            )
        else:
            origins = origins[native["window_start"] : native["local_end_index"]]
        if len(origins) != keys.shape[1] or int(origins.min()) < 0:
            raise ValueError("Provenance is missing from the native attention window")
        frames = origins // SPATIAL
        query = (
            q[0]
            .reshape(BLOCK, SPATIAL, *q.shape[2:])
            .index_select(1, self.query_points)
        )
        query = query.flatten(0, 1).transpose(0, 1).float()
        key = keys[0].permute(1, 2, 0).float()
        logits = torch.bmm(query, key) * (q.shape[-1] ** -0.5)
        logp = logits.log_softmax(-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1).mean()
        # The marginal over source frames is averaged over query positions and heads.
        marginal = torch.bincount(
            frames, weights=p.mean((0, 1)), minlength=start + BLOCK
        )
        sink = (frames < 10).sum()
        values = torch.stack(
            (
                entropy,
                entropy / math.log(len(origins)),
                marginal[:10].sum(),
                sink / len(origins),
                1 / marginal.square().sum(),
                marginal.sum(),
            )
        ).tolist()
        self.attention_rows.append([layer, start, len(origins), *values])
        self.source_rows.append(
            {"layer": layer, "chunk": start, "mass": marginal.cpu().tolist()}
        )

    def cross_readout(self, layer, q, key):
        ngrid = len(TEXT_POINTS)
        visual = (
            q[0].reshape(BLOCK, SPATIAL, *q.shape[2:]).index_select(1, self.text_points)
        )
        visual = F.normalize(visual.float(), dim=-1).permute(2, 0, 1, 3).flatten(1, 2)
        language = F.normalize(
            key[0].index_select(0, self.text_indices).float(), dim=-1
        )
        similarity = torch.bmm(visual, language.permute(1, 2, 0))
        similarity = similarity.view(q.shape[2], BLOCK, len(self.text_points), -1)
        best = similarity.amax(-1)
        data = [best[:, :, :ngrid].mean((0, 2)), best[:, :, ngrid:].mean((0, 2))]
        if self.subject_indices:
            subj = similarity[..., self.subject_indices].amax(-1)
            data.extend(
                (subj[:, :, :ngrid].mean((0, 2)), subj[:, :, ngrid:].mean((0, 2)))
            )
        for offset, values in enumerate(torch.stack(data, -1).cpu().tolist()):
            self.cross_features[layer, self.chunk + offset] = values

    def save(self):
        expected = {(layer, frame) for layer in range(30) for frame in range(123)}
        if (
            set(self.self_features) != expected
            or set(self.cross_features) != expected
            or len(self.attention_rows) != 30 * 41
            or len(self.chunk_counts) != 41
            or set(self.chunk_counts.values()) != {1 + self.denoising_steps}
        ):
            raise ValueError("Incomplete layer or time coverage")
        csv_out(
            self.directory / "attention.csv",
            [
                "layer",
                "chunk",
                "keys",
                "entropy_nats",
                "entropy_normalized",
                "sink_mass",
                "sink_slot_share",
                "effective_attended_frames",
                "mass_check",
            ],
            self.attention_rows,
        )
        with gzip.open(
            self.directory / "source-attention.jsonl.gz", "wt", encoding="utf-8"
        ) as stream:
            for row in self.source_rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        fields = [
            "layer",
            "source",
            "first_cos",
            "roi_first_cos",
            "text_cos",
            "roi_text_cos",
        ]
        if self.subject_indices:
            fields.extend(("subject_cos", "roi_subject_cos"))
        csv_out(
            self.directory / "features.csv",
            fields,
            (
                [*key, *self.self_features[key], *self.cross_features[key]]
                for key in sorted(expected)
            ),
        )
        if self.roi_rows:
            csv_out(
                self.directory / "workshop-region.csv",
                ["layer", "chunk", "candidate", "kept", "roi_candidate", "roi_kept"],
                self.roi_rows,
            )
            (self.directory / "patches.json").write_text(json.dumps(self.patches))


audit = Audit()


def main():
    torch.set_grad_enabled(False)
    pipe, low_memory = generate.load_pipeline()
    audit.denoising_steps = len(pipe.denoising_step_list)
    audit.context_noise = pipe.args.context_noise
    pipe.generator.register_forward_pre_hook(audit.generator_call, with_kwargs=True)
    cross_layers = {
        id(block.cross_attn): i for i, block in enumerate(pipe.generator.model.blocks)
    }
    original_self = causal_model.attention
    original_cross = model.flash_attention

    def self_attention(q, k, v, *args, **kwargs):
        if audit.clean:
            audit.self_readout(q, k, sys._getframe(1).f_locals)
        return original_self(q, k, v, *args, **kwargs)

    def cross_attention(q, k, v, *args, **kwargs):
        if audit.clean:
            layer = cross_layers[id(sys._getframe(1).f_locals["self"])]
            audit.cross_readout(layer, q, k)
        return original_cross(q, k, v, *args, **kwargs)

    causal_model.attention = self_attention
    model.flash_attention = cross_attention
    generate.CacheProbe = Probe
    cache_probe.TRACE_LAYERS = tuple(range(30))
    tokenizer = pipe.text_encoder.tokenizer

    for seed in SEEDS:
        prompts = json.loads((baseline(seed).parent / "prompts.json").read_text())
        prompt = next(row for row in prompts if row["seed"] == seed)
        generate.OUT = Path.cwd() / "output" / str(seed)
        for name in ("videos", "meta", "prefix", "traces"):
            (generate.OUT / name).mkdir(parents=True, exist_ok=True)
        ids, mask = tokenizer(
            [prompt["prompt"]], return_mask=True, add_special_tokens=True
        )
        special = set(tokenizer.tokenizer.all_special_ids)
        positions = [
            j for j in range(int(mask[0].sum())) if int(ids[0, j]) not in special
        ]
        subject = []
        if seed == 34001:
            lead = tokenizer.tokenizer.encode("A", add_special_tokens=False)
            words = tokenizer.tokenizer.encode(
                "A solitary woman", add_special_tokens=False
            )
            if ids[0, : len(words)].tolist() != words:
                raise ValueError("Subject phrase tokenization mismatch")
            subject = [positions.index(j) for j in range(len(lead), len(words))]
        audit.reset(seed, positions, subject, generate.OUT)
        run_info = {
            **prompt,
            "text_positions": positions,
            "subject_positions_in_text_subset": subject,
            "text_pieces": tokenizer.tokenizer.convert_ids_to_tokens(
                ids[0, positions].tolist()
            ),
            "self_grid": SELF_POINTS,
            "cross_grid": TEXT_POINTS,
            "roi": ROI,
            "context_noise": audit.context_noise,
            "schedule": pipe.denoising_step_list.tolist(),
        }
        (generate.OUT / "case.json").write_text(json.dumps(run_info, indent=2))
        generate.generate(pipe, low_memory, prompt, "attention")
        audit.save()
        print("AUDIT SAVED", seed, len(audit.attention_rows), flush=True)


if __name__ == "__main__":
    main()
