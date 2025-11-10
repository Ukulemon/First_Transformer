"""A minimal yet expressive Transformer model implementation."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import TransformerConfig
from .layers import TransformerDecoderLayer, TransformerEncoderLayer
from .utils import PositionalEncoding, generate_square_subsequent_mask


class TransformerModel(nn.Module):
    """Transformer model composed of encoder and decoder stacks."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_seq_length
        )
        self.dropout = nn.Dropout(config.dropout)
        self.embedding_scale = math.sqrt(config.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.d_model,
                    config.num_heads,
                    config.dim_feedforward,
                    config.dropout,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    config.d_model,
                    config.num_heads,
                    config.dim_feedforward,
                    config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

        self.generator = nn.Linear(config.d_model, config.vocab_size)

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tuple[Tensor, list[Tensor]]:
        src = self.src_embedding(src) * self.embedding_scale
        src = self.positional_encoding(src)
        src = self.dropout(src)

        attentions = []
        for layer in self.encoder_layers:
            src, attn_weights = layer(src, src_mask)
            attentions.append(attn_weights)
        return src, attentions

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list[Tuple[Tensor, Tensor]]]:
        tgt = self.tgt_embedding(tgt) * self.embedding_scale
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        attentions = []
        for layer in self.decoder_layers:
            tgt, attn_weights = layer(tgt, memory, tgt_mask, memory_mask)
            attentions.append(attn_weights)
        return tgt, attentions

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict[str, list[Tensor]]]:
        memory, encoder_attn = self.encode(src, src_mask)
        output, decoder_attn = self.decode(tgt, memory, tgt_mask, memory_mask)
        logits = self.generator(output)
        return logits, {"encoder": encoder_attn, "decoder": decoder_attn}

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        start_symbol: int,
        max_len: int,
        src_mask: Optional[Tensor] = None,
        end_symbol: Optional[int] = None,
    ) -> Tensor:
        """Generate a sequence using greedy decoding."""

        device = src.device
        memory, _ = self.encode(src, src_mask)

        ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device=device)
            output, _ = self.decode(ys, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
            logits = self.generator(output[:, -1])
            next_word = torch.argmax(logits, dim=-1, keepdim=True)
            if end_symbol is not None:
                next_word = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_word, end_symbol), next_word
                )
            ys = torch.cat([ys, next_word], dim=1)
            if end_symbol is not None:
                finished = finished | (next_word.squeeze(-1) == end_symbol)
                if finished.all():
                    break
        return ys
