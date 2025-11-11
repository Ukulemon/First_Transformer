"""Transformer encoder and decoder layers."""

from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attn_output, attn_weights = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn_weights


class TransformerDecoderLayer(nn.Module):
    """A single Transformer decoder block."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        self_attn_output, self_attn_weights = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(self_attn_output)
        tgt = self.norm1(tgt)

        attn_output, cross_attn_weights = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.linear(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        return tgt, (self_attn_weights, cross_attn_weights)
