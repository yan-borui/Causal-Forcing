"""Small multiplicative modulation of Deep Forcing's native, signed token score."""
import torch

from cache_probe import CacheProbe, frame_diversity


WEIGHT = 0.15


class MultiplicativeProbe(CacheProbe):
    def __init__(self, method, stream):
        super().__init__(method, stream)
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.candidates = self.negative_scores = self.nonpositive_factors = 0

    def rank(self, fused, keys, source_tokens, sink, recent, spatial_size):
        old = slice(sink, len(source_tokens) - recent)
        diversity, frame_info = frame_diversity(keys[0, old], source_tokens[old], spatial_size)
        native = fused[0, old]
        factor = 1 + WEIGHT * diversity
        scores = fused.clone()
        scores[0, old] = native * factor
        self.candidates += native.numel()
        self.negative_scores += int((native < 0).sum())
        self.nonpositive_factors += int((factor <= 0).sum())
        self.minimum = min(self.minimum, float(factor.min()))
        self.maximum = max(self.maximum, float(factor.max()))
        return scores, frame_info

    def score_statistics(self):
        return dict(weight=WEIGHT, pre_reuse_candidates=self.candidates,
                    negative_native_scores=self.negative_scores,
                    nonpositive_factors=self.nonpositive_factors,
                    minimum_factor=self.minimum, maximum_factor=self.maximum)
