"""Utilities to train and run inference with the minimal Transformer model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from first_transformer import TransformerConfig, TransformerModel, generate_square_subsequent_mask


class CharTokenizer:
    """Character-level tokenizer with special token handling."""

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self, texts: Sequence[str]) -> None:
        charset = {ch for text in texts for ch in text}
        specials = [self.pad_token, self.bos_token, self.eos_token]
        self.itos: List[str] = specials + sorted(charset - set(specials))
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Character {ch!r} not found in tokenizer vocabulary")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        for idx in ids:
            if idx in special_ids:
                continue
            tokens.append(self.itos[idx])
        return "".join(tokens)

    def save(self, path: Path) -> None:
        payload = {
            "itos": self.itos,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        tokenizer = cls.__new__(cls)
        tokenizer.itos = list(payload["itos"])
        tokenizer.stoi = {token: idx for idx, token in enumerate(tokenizer.itos)}
        tokenizer.pad_token = payload["pad_token"]
        tokenizer.bos_token = payload["bos_token"]
        tokenizer.eos_token = payload["eos_token"]
        tokenizer.pad_id = tokenizer.stoi[tokenizer.pad_token]
        tokenizer.bos_id = tokenizer.stoi[tokenizer.bos_token]
        tokenizer.eos_id = tokenizer.stoi[tokenizer.eos_token]
        return tokenizer


class PrefixDataset(Dataset[Tuple[Tensor, Tensor, Tensor]]):
    """Dataset that teaches the model to extend text prefixes."""

    def __init__(self, texts: Sequence[str], tokenizer: CharTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[List[int]] = []

        for text in texts:
            if not text:
                continue
            token_ids = tokenizer.encode(text)
            if len(token_ids) + 2 > max_length:
                # Skip sequences that exceed our modeling capacity.
                continue
            full_sequence = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]
            if len(full_sequence) < 3:
                continue
            self.samples.append(full_sequence)

        if not self.samples:
            raise ValueError("No valid training samples were created. Check your data or max_length.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        sequence = self.samples[idx]
        if len(sequence) <= 1:
            raise ValueError("Sequence too short to create prefix/suffix split")
        split_point = random.randint(1, len(sequence) - 1)
        src_tokens = sequence[:split_point]
        suffix_tokens = sequence[split_point:]

        tgt_input = [self.tokenizer.bos_id] + suffix_tokens[:-1]
        tgt_output = suffix_tokens

        return (
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long),
        )


def collate_fn(batch: Sequence[Tuple[Tensor, Tensor, Tensor]], pad_id: int) -> dict[str, Tensor]:
    src_seqs, tgt_inputs, tgt_outputs = zip(*batch)
    batch_size = len(batch)
    max_src_len = max(seq.size(0) for seq in src_seqs)
    max_tgt_len = max(seq.size(0) for seq in tgt_inputs)

    src_batch = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_in_batch = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_out_batch = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)

    for i, (src, tgt_in, tgt_out) in enumerate(batch):
        src_batch[i, : src.size(0)] = src
        tgt_in_batch[i, : tgt_in.size(0)] = tgt_in
        tgt_out_batch[i, : tgt_out.size(0)] = tgt_out

    src_mask = src_batch.eq(pad_id)
    src_mask = src_mask.masked_fill(src_mask, float("-inf")).float()
    src_mask = src_mask[:, None, None, :]

    tgt_mask = generate_square_subsequent_mask(max_tgt_len)
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
    tgt_padding = tgt_in_batch.eq(pad_id)
    tgt_mask = tgt_mask.masked_fill(tgt_padding.unsqueeze(1), float("-inf"))

    return {
        "src": src_batch,
        "tgt_in": tgt_in_batch,
        "tgt_out": tgt_out_batch,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def load_texts(path: Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    set_seed(args.seed)

    texts = load_texts(args.data)
    tokenizer = CharTokenizer(texts)
    dataset = PrefixDataset(texts, tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
    )

    config = TransformerConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        max_seq_length=args.max_length,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
    )

    model = TransformerModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    loss_history: List[Tuple[int, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                loss_history.append((global_step, avg_loss))
                print(f"Epoch {epoch} Step {global_step}: loss={avg_loss:.4f}")
                running_loss = 0.0

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }
    ckpt_path = output_dir / "transformer.pt"
    torch.save(checkpoint, ckpt_path)
    tokenizer.save(output_dir / "tokenizer.json")
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        json.dump(args_payload, fh, ensure_ascii=False, indent=2)
    if loss_history:
        with (output_dir / "loss_history.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "loss"])
            writer.writerows(loss_history)
    print(f"Training complete. Checkpoint saved to {ckpt_path}")


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = TransformerConfig(**checkpoint["config"])
    model = TransformerModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.load(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    src_tokens = [tokenizer.bos_id] + prompt_ids
    if len(src_tokens) > config.max_seq_length:
        raise ValueError("Source sequence is longer than the model's maximum length")

    src = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = src.eq(config.pad_token_id)
    src_mask = src_mask.masked_fill(src_mask, float("-inf")).float()
    src_mask = src_mask[:, None, None, :]

    max_len = min(args.max_new_tokens + 1, config.max_seq_length)
    generated = model.greedy_decode(
        src,
        start_symbol=config.bos_token_id,
        max_len=max_len,
        src_mask=src_mask,
        end_symbol=config.eos_token_id,
    )
    generated_tokens = []
    for token in generated[0].tolist()[1:]:  # drop BOS
        if token == config.eos_token_id:
            break
        generated_tokens.append(token)

    completion = tokenizer.decode(generated_tokens)
    result = args.prompt + completion
    print(result)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(result)
        print(f"Generation saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run inference with the minimal Transformer model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Transformer model")
    train_parser.add_argument("--data", type=Path, required=True, help="Path to a UTF-8 text file (one sample per line)")
    train_parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Directory to store checkpoints")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length (<= positional encoding size)")
    train_parser.add_argument("--d-model", type=int, default=256, help="Embedding dimension")
    train_parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("--num-encoder-layers", type=int, default=4, help="Number of encoder layers")
    train_parser.add_argument("--num-decoder-layers", type=int, default=4, help="Number of decoder layers")
    train_parser.add_argument("--ffn-dim", type=int, default=1024, help="Hidden dimension of the feed-forward network")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW")
    train_parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value (<=0 disables)")
    train_parser.add_argument("--log-every", type=int, default=50, help="Steps between logging training loss")
    train_parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--device", type=str, default=None, help="torch device string (default: auto)" )

    generate_parser = subparsers.add_parser("generate", help="Run inference with a trained checkpoint")
    generate_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the saved model checkpoint")
    generate_parser.add_argument("--tokenizer", type=Path, required=True, help="Path to the saved tokenizer JSON")
    generate_parser.add_argument("--prompt", type=str, required=True, help="Prompt text to continue")
    generate_parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    generate_parser.add_argument("--device", type=str, default=None, help="torch device string (default: auto)")
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the generated continuation",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
