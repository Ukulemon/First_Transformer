"""Configuration objects for the Transformer model."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Hyper-parameters required to build a Transformer model.

    Attributes
    ----------
    vocab_size:
        Size of the vocabulary used for both the source and target sequences.
    d_model:
        Dimensionality of the token embeddings.
    num_heads:
        Number of attention heads used in multi-head attention modules.
    num_encoder_layers:
        Number of encoder layers stacked in the encoder block.
    num_decoder_layers:
        Number of decoder layers stacked in the decoder block.
    dim_feedforward:
        Hidden dimensionality of the feed-forward network within each layer.
    dropout:
        Dropout probability applied at various points in the network.
    max_seq_length:
        Maximum sequence length supported by the positional encoding module.
    pad_token_id:
        Padding token used to mask inactive positions in both source and target.
    bos_token_id:
        Token that marks the beginning of target sequences.
    eos_token_id:
        Token that marks the end of target sequences.
    """

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
