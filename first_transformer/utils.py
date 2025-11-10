"""Utility functions used across the Transformer implementation."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as introduced in the Transformer paper."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information to the input embeddings."""

        x = x + self.pe[:, : x.size(1)].clone().detach()
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None) -> Tensor:
    """Create a mask to prevent attention to subsequent positions."""

    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def expand_mask(mask: Optional[Tensor], target_len: int) -> Optional[Tensor]:
    """Expand a 2-D mask to the 4-D shape expected by the attention module."""

    if mask is None:
        return None
    if mask.dtype == torch.bool:
        mask = mask.masked_fill(mask, float("-inf")).to(dtype=torch.float32)
    elif mask.dtype != torch.float32:
        mask = mask.to(dtype=torch.float32)
    if mask.dim() == 2:
        if mask.size(0) == target_len and mask.size(1) == target_len:
            return mask.unsqueeze(0).unsqueeze(0)
        # (batch, src_len) -> (batch, 1, 1, src_len)
        return mask[:, None, None, :].expand(-1, 1, target_len, -1)
    if mask.dim() == 3:
        # (batch, target_len, src_len) -> (batch, 1, target_len, src_len)
        return mask[:, None, :, :]
    return mask
