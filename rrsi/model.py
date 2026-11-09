"""Dual-head descriptor with local and cross-image feature attention."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from model.Model1 import CNN_model1
from .attention import CrossAttentionStack, FeedForward, SelfAttentionStack


@dataclass
class DescriptorOutput:
    descriptor_a: torch.Tensor
    descriptor_b: torch.Tensor


class RRSIDescriptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_a = CNN_model1()
        self.encoder_b = CNN_model1()
        self.encoder_chunk_size = int(config.get("encoder_chunk_size", 128))
        self.context_window = int(config.get("context_window", 128))
        self.context_tokens = int(config.get("context_tokens", 128))
        if min(self.encoder_chunk_size, self.context_window, self.context_tokens) < 1:
            raise ValueError("Encoder batches and attention context sizes must be positive.")
        dimension = 256
        heads = int(config.get("attention_heads", 4))
        ratio = float(config.get("mlp_ratio", 1.0))
        self.self_a_cartesian = SelfAttentionStack(1, dimension, heads, ratio, 0.0)
        self.self_a_logpolar = SelfAttentionStack(1, dimension, heads, ratio, 0.0)
        self.self_b_cartesian = SelfAttentionStack(1, dimension, heads, ratio, 0.0)
        self.self_b_logpolar = SelfAttentionStack(1, dimension, heads, ratio, 0.0)
        self.intra_cross_a = CrossAttentionStack(1, dimension, heads, ratio, 0.0)
        self.intra_cross_b = CrossAttentionStack(1, dimension, heads, ratio, 0.0)
        self.fusion_a_cartesian = FeedForward(dimension, ratio, 0.0)
        self.fusion_a_logpolar = FeedForward(dimension, ratio, 0.0)
        self.fusion_b_cartesian = FeedForward(dimension, ratio, 0.0)
        self.fusion_b_logpolar = FeedForward(dimension, ratio, 0.0)
        self.inter_cross = CrossAttentionStack(1, dimension, heads, ratio, 0.0)

    def encode(self, encoder, patches):
        return torch.cat([encoder(batch) for batch in patches.split(self.encoder_chunk_size)])

    def _local_features(self, cartesian, logpolar, modality):
        self_cartesian = getattr(self, "self_{}_cartesian".format(modality))
        self_logpolar = getattr(self, "self_{}_logpolar".format(modality))
        cross = getattr(self, "intra_cross_{}".format(modality))
        fusion_cartesian = getattr(self, "fusion_{}_cartesian".format(modality))
        fusion_logpolar = getattr(self, "fusion_{}_logpolar".format(modality))
        outputs = []
        for start in range(0, cartesian.shape[1], self.context_window):
            original = cartesian[:, start:start + self.context_window]
            polar = logpolar[:, start:start + self.context_window]
            refined_c, refined_lp = cross(self_cartesian(original), self_logpolar(polar))
            outputs.append(original + fusion_cartesian(refined_c) + fusion_logpolar(refined_lp))
        return torch.cat(outputs, dim=1)

    def _context(self, features):
        count = min(features.shape[1], self.context_tokens)
        indices = torch.linspace(0, features.shape[1] - 1, count, device=features.device).long()
        return features[:, indices]

    def forward_features(self, cartesian_a, logpolar_a, cartesian_b, logpolar_b):
        """Process encoded BxNx256 patch features from both modalities."""
        features_a = self._local_features(cartesian_a, logpolar_a, "a")
        features_b = self._local_features(cartesian_b, logpolar_b, "b")
        context_a, context_b = self._context(features_a), self._context(features_b)
        refined_a = []
        for chunk in features_a.split(self.context_window, dim=1):
            updated, _ = self.inter_cross(chunk, context_b)
            refined_a.append(updated)
        refined_b = []
        for chunk in features_b.split(self.context_window, dim=1):
            _, updated = self.inter_cross(context_a, chunk)
            refined_b.append(updated)
        return DescriptorOutput(
            F.normalize(torch.cat(refined_a, dim=1), p=2, dim=-1),
            F.normalize(torch.cat(refined_b, dim=1), p=2, dim=-1),
        )

    def forward(self, cartesian_a, logpolar_a, cartesian_b, logpolar_b):
        a_c = self.encode(self.encoder_a, cartesian_a)
        a_lp = self.encode(self.encoder_a, logpolar_a)
        b_c = self.encode(self.encoder_b, cartesian_b)
        b_lp = self.encode(self.encoder_b, logpolar_b)
        output = self.forward_features(a_c[None], a_lp[None], b_c[None], b_lp[None])
        return DescriptorOutput(output.descriptor_a[0], output.descriptor_b[0])
