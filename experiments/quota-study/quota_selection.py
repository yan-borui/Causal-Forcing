"""Two finite-budget interventions on the original Deep Forcing candidates."""

import numpy as np
import torch

from cache_probe import CacheProbe, TRACE_LAYERS, standardize


def bounded_quotas(capacity, weight, budget):
    """Allocate integer seats with a coverage floor, saturation and largest remainders."""
    capacity = np.asarray(capacity, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    if budget > capacity.sum() or np.any(capacity < 1) or not np.all(weight > 0):
        raise ValueError("Invalid positive-capacity frame allocation")
    if budget < len(capacity):
        quota = np.zeros_like(capacity)
        quota[np.argsort(-weight, kind="stable")[:budget]] = 1
        return quota
    quota = np.ones_like(capacity)
    remaining = budget - len(capacity)
    while remaining:
        ids = np.flatnonzero(capacity > quota)
        room = capacity[ids] - quota[ids]
        ideal = remaining * weight[ids] / weight[ids].sum()
        saturated = ideal >= room
        if saturated.any():
            quota[ids[saturated]] += room[saturated]
            remaining -= int(room[saturated].sum())
            continue
        whole = np.floor(ideal).astype(np.int64)
        quota[ids] += whole
        left = remaining - int(whole.sum())
        order = np.argsort(-(ideal - whole), kind="stable")
        quota[ids[order[:left]]] += 1
        break
    return quota


def frame_ranks(scores, frame_index, counts):
    """Token order grouped by frame, descending score within each frame."""
    by_score = torch.argsort(scores, descending=True, stable=True)
    groups = torch.argsort(frame_index[by_score], stable=True)
    order = by_score[groups]
    first = counts.cumsum(0) - counts
    rank = torch.arange(len(order), device=order.device) - first[frame_index[order]]
    return order, rank


class QuotaProbe(CacheProbe):
    def __init__(self, method, stream):
        super().__init__(method, stream)
        self.policy_stats = dict(
            decisions=0,
            additive_exceeds_native_max=0,
            changes_to_additive=0,
            cap_violations=0,
            allocated_frames_missing=0,
        )
        self.decision_record = {}

    def adjust_selection(
        self,
        layer,
        query,
        fused,
        scores,
        origins,
        frame_info,
        sink,
        recent,
        spatial,
        top_c,
        counts,
        reuse,
        selection,
        selector,
    ):
        if self.method in ("attention", "mixture"):
            return selection
        if self.method not in ("capped", "two_stage"):
            raise ValueError(f"Unknown selection method {self.method}")
        total = len(origins)
        stop = total - recent
        device = origins.device
        options = dict(
            T=total,
            R_eff=recent,
            sink_tokens=sink,
            top_c=top_c,
            device=device,
            topc_counts=counts,
            topc_max_reuse=reuse,
        )
        raw_idx = selection[2][0]
        native_idx = selector(fused, **options)[2][0]
        nslots = len(raw_idx)
        if nslots != len(native_idx) or nslots != top_c:
            raise ValueError("Candidate set cannot fill the original selection budget")
        frame = origins // spatial
        length = query + 3
        raw_hist = torch.bincount(frame[raw_idx], minlength=length)
        native_hist = torch.bincount(frame[native_idx], minlength=length)
        cap = int(native_hist.max())
        exceeds = int(raw_hist.max()) > cap
        chosen = selection
        if self.method == "two_stage" or exceeds:
            positions = torch.arange(sink, stop, device=device)
            if counts is not None and reuse > 0:
                positions = positions[counts[0, positions] < reuse]
            ids, group, sizes = torch.unique(
                frame[positions], sorted=True, return_inverse=True, return_counts=True
            )
            if self.method == "two_stage":
                sums = torch.zeros(len(ids), device=device)
                sums.index_add_(0, group, fused[0, positions])
                frame_attention = standardize(sums / sizes, 0)
                diversity = frame_info[1][torch.searchsorted(frame_info[0], ids)]
                weights = torch.softmax(
                    (0.5 * frame_attention + 0.5 * diversity).double(), dim=0
                )
                quotas = bounded_quotas(
                    sizes.cpu().tolist(), weights.cpu().tolist(), nslots
                )
                limits = torch.as_tensor(quotas, device=device)
                ordering = fused[0, positions]
            else:
                limits = torch.full_like(sizes, cap)
                ordering = scores[0, positions]
            order, ranks = frame_ranks(ordering, group, sizes)
            allowed = positions[order[ranks < limits[group[order]]]]
            candidate = torch.full_like(scores, -torch.inf)
            candidate[0, allowed] = scores[0, allowed]
            chosen = selector(candidate, **options)
            selected = chosen[2][0]
            if len(selected) != nslots or not bool(
                torch.isfinite(candidate[0, selected]).all()
            ):
                raise ValueError("An ineligible token entered the allocation")
            if self.method == "two_stage":
                actual = torch.bincount(frame[selected], minlength=length)[ids]
                if not torch.equal(actual, limits):
                    raise ValueError(
                        "Within-frame selection diverged from the assigned quotas"
                    )
                self.policy_stats["allocated_frames_missing"] += int(
                    (actual == 0).sum()
                )

        selected = chosen[2][0]
        actual_hist = torch.bincount(frame[selected], minlength=length)
        if self.method == "capped" and int(actual_hist.max()) > cap:
            self.policy_stats["cap_violations"] += 1
            raise ValueError("Exceeded the maximum allocated by same-cache attention")
        self.policy_stats["decisions"] += 1
        self.policy_stats["additive_exceeds_native_max"] += int(exceeds)
        self.policy_stats["changes_to_additive"] += int(
            not torch.equal(selected, raw_idx)
        )
        if layer in TRACE_LAYERS:
            used = torch.where(raw_hist > 0)[0]
            self.decision_record = dict(
                native_cap=cap,
                additive_max=int(raw_hist.max()),
                actual_max=int(actual_hist.max()),
                additive_frames=torch.stack([used, raw_hist[used]], 1).cpu().tolist(),
            )
        return chosen
