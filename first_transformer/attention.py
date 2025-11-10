"""Attention related modules."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .utils import expand_mask


class MultiHeadAttention(nn.Module):
    """A simple multi-head attention implementation."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: Tensor, seq_len: int, bsz: int) -> Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, tgt_len, _ = query.size()
        src_len = key.size(1)

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self._shape(query, tgt_len, bsz)
        key = self._shape(key, src_len, bsz)
        value = self._shape(value, src_len, bsz)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = expand_mask(mask, tgt_len)
            scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)

        context = context.transpose(1, 2).contiguous().view(bsz, tgt_len, self.d_model)
        output = self.out_proj(context)
        return output, attn
