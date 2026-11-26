"""Deep Forcing candidate ranking and selection provenance for the paired pilot."""
import json

import torch
import torch.nn.functional as F


TRACE_LAYERS = (0, 14, 29)


def standardize(values, dim):
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, unbiased=False, keepdim=True)
    return (values - mean) / std.clamp_min(1e-6)


def frame_diversity(keys, source_tokens, spatial_size):
    """Focused Forcing eqs. 9–10 over the frame/position pairs still present.

    The 3D cache can contain only some patches of a historical frame. Compare
    keys at the same original spatial position and average over observed patches.
    No evicted key is stored or reconstructed.
    """
    frame = source_tokens // spatial_size
    spatial = source_tokens % spatial_size
    frame_ids, frame_idx, frame_counts = torch.unique(
        frame, sorted=True, return_inverse=True, return_counts=True
    )
    keys = keys.float()
    mean_key = keys.new_zeros((spatial_size, *keys.shape[1:]))
    mean_key.index_add_(0, spatial, keys)
    counts = torch.bincount(spatial, minlength=spatial_size).clamp_min(1)
    mean_key /= counts[:, None, None]
    mean_direction = F.normalize(mean_key, dim=-1)
    per_token_cos = (F.normalize(keys, dim=-1) * mean_direction[spatial]).sum(-1)
    per_frame = keys.new_zeros((len(frame_ids), keys.shape[1]))
    per_frame.index_add_(0, frame_idx, -per_token_cos)
    per_frame /= frame_counts[:, None]
    # Deep Forcing chooses one shared token set for all heads.
    diversity = standardize(per_frame, 0).mean(dim=1)
    return diversity[frame_idx], (frame_ids, diversity)


class CacheProbe:
    def __init__(self, method, stream):
        self.method = method
        self.stream = stream
        self.events = 0
        self.total_kept_min = None
        self.total_kept_max = 0
        self.old_kept_min = None
        self.old_kept_max = 0

    def rank(self, fused, keys, source_tokens, sink, recent, spatial_size):
        if self.method == "attention":
            return fused, None
        stop = len(source_tokens) - recent
        old = slice(sink, stop)
        diversity, frame_info = frame_diversity(keys[0, old], source_tokens[old], spatial_size)
        scores = fused.clone()
        scores[0, old] = 0.5 * standardize(fused[0, old], 0) + 0.5 * diversity
        return scores, frame_info

    def record(self, layer, query_frame, spatial_size, origins, attention, counts, reuse,
               sink, recent, top_c, kept, selected, frame_info, selector):
        self.events += 1
        n, c = len(kept), len(selected)
        self.total_kept_min = n if self.total_kept_min is None else min(n, self.total_kept_min)
        self.total_kept_max = max(n, self.total_kept_max)
        self.old_kept_min = c if self.old_kept_min is None else min(c, self.old_kept_min)
        self.old_kept_max = max(c, self.old_kept_max)
        if layer not in TRACE_LAYERS:
            return
        stop = len(origins) - recent
        candidate_frames = origins[sink:stop] // spatial_size
        if candidate_frames.min().item() < 0:
            raise ValueError("Uninitialised source-token provenance in the candidate region")
        if self.method == "mixture":
            native = selector(attention, len(origins), recent, sink, top_c, origins.device,
                              topc_counts=counts, topc_max_reuse=reuse)[2][0]
            # Both selections address this exact cache before it is modified.
            selected_mask = torch.zeros_like(origins, dtype=torch.bool)
            selected_mask[selected] = True
            overlap = int(selected_mask[native].sum().item())
        else:
            native, overlap = selected, len(selected)
        slots = query_frame + 3
        available = torch.bincount(candidate_frames, minlength=slots)
        actual = torch.bincount(origins[selected] // spatial_size, minlength=slots)
        reference = torch.bincount(origins[native] // spatial_size, minlength=slots)
        frame_ids = torch.where(available > 0)[0]
        hist = torch.stack([frame_ids, available[frame_ids], actual[frame_ids], reference[frame_ids]], 1)
        row = dict(layer=layer, query=query_frame, sink=sink, recent=recent,
                   kept=n, selected=c, native=len(native), overlap=overlap,
                   frames=hist.cpu().tolist())
        if frame_info is not None:
            row["diversity_z"] = frame_info[1].cpu().tolist()
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
