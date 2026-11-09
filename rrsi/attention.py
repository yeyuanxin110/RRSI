from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dimension: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(dimension * mlp_ratio))
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dimension)
        self.mlp = FeedForward(dimension, mlp_ratio, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(features)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        features = features + attended
        return features + self.mlp(self.mlp_norm(features))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm_a = nn.LayerNorm(dimension)
        self.query_norm_b = nn.LayerNorm(dimension)
        self.context_norm_a = nn.LayerNorm(dimension)
        self.context_norm_b = nn.LayerNorm(dimension)
        self.attention_a_from_b = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.attention_b_from_a = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm_a = nn.LayerNorm(dimension)
        self.mlp_norm_b = nn.LayerNorm(dimension)
        self.mlp_a = FeedForward(dimension, mlp_ratio, dropout)
        self.mlp_b = FeedForward(dimension, mlp_ratio, dropout)


    def forward(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_a = self.query_norm_a(features_a)
        query_b = self.query_norm_b(features_b)
        context_a = self.context_norm_a(features_a)
        context_b = self.context_norm_b(features_b)
        update_a, _ = self.attention_a_from_b(
            query_a, context_b, context_b, need_weights=False
        )
        update_b, _ = self.attention_b_from_a(
            query_b, context_a, context_a, need_weights=False
        )
        features_a = features_a + update_a
        features_b = features_b + update_b
        features_a = features_a + self.mlp_a(self.mlp_norm_a(features_a))
        features_b = features_b + self.mlp_b(self.mlp_norm_b(features_b))
        return features_a, features_b


class SelfAttentionStack(nn.Module):
    def __init__(
        self,
        layers: int,
        dimension: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SelfAttentionBlock(dimension, heads, mlp_ratio, dropout)
            for _ in range(layers)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            features = layer(features)
        return features


class CrossAttentionStack(nn.Module):
    def __init__(
        self,
        layers: int,
        dimension: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CrossAttentionBlock(dimension, heads, mlp_ratio, dropout)
            for _ in range(layers)
        )


    def forward(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            features_a, features_b = layer(features_a, features_b)
        return features_a, features_b
