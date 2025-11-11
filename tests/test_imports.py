"""Basic smoke tests for package-level imports and integration."""

from __future__ import annotations

import unittest

try:  # pragma: no cover - optional dependency for smoke tests
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may be unavailable in CI
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ImportRelationshipTest(unittest.TestCase):
    """Verify that the public package API wires together correctly."""

    def test_model_forward_runs_with_package_imports(self) -> None:
        from first_transformer import (
            TransformerConfig,
            TransformerModel,
            generate_square_subsequent_mask,
        )

        config = TransformerConfig(
            vocab_size=32,
            d_model=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )

        model = TransformerModel(config)
        self.assertIsNotNone(model)

        src = torch.randint(0, config.vocab_size, (2, 4))
        tgt = torch.randint(0, config.vocab_size, (2, 4))

        tgt_mask = generate_square_subsequent_mask(tgt.size(1))
        logits, attentions = model(src, tgt, tgt_mask=tgt_mask)

        self.assertEqual(logits.shape, (2, 4, config.vocab_size))
        self.assertIn("encoder", attentions)
        self.assertIn("decoder", attentions)


if __name__ == "__main__":  # pragma: no cover - allow running as a script
    unittest.main()
