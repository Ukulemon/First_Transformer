"""Small demonstration of the minimal Transformer model."""

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from first_transformer import TransformerConfig, TransformerModel, generate_square_subsequent_mask


def main() -> None:
    config = TransformerConfig(
        vocab_size=32,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_seq_length=32,
    )

    model = TransformerModel(config)

    batch_size = 2
    src_len = 6
    tgt_len = 5

    src = torch.randint(0, config.vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, config.vocab_size, (batch_size, tgt_len))

    tgt_mask = generate_square_subsequent_mask(tgt_len)

    logits, attention_maps = model(src, tgt, tgt_mask=tgt_mask)
    print("Logits shape:", logits.shape)
    print(
        "Encoder attention maps:",
        [layer_attn.shape for layer_attn in attention_maps["encoder"]],
    )
    print(
        "Decoder self-attention maps:",
        [layer_attn[0].shape for layer_attn in attention_maps["decoder"]],
    )
    print(
        "Decoder cross-attention maps:",
        [layer_attn[1].shape for layer_attn in attention_maps["decoder"]],
    )


if __name__ == "__main__":
    main()
