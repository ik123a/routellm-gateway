"""
Router Model Definition
========================
DistilBERT-based classification model that predicts the optimal
LLM tier (CHEAP=0, MEDIUM=1, STRONG=2) for incoming queries.

Architecture:
  DistilBERT (frozen or fine-tuned) -> Mean Pooling -> Dropout -> FC(768,256)
  -> ReLU -> Dropout -> FC(256, 3) -> Softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class RouterModel(nn.Module):
    """
    Query complexity classifier built on DistilBERT.

    Input: tokenized text (input_ids, attention_mask)
    Output: logits for 3 tiers [CHEAP, MEDIUM, STRONG]
    """

    def __init__(
        self,
        backbone: str = "distilbert-base-uncased",
        num_tiers: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone)
        self.hidden_size = self.backbone.config.hidden_size  # 768 for distilbert

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tiers),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.

        Returns:
            logits: (batch_size, num_tiers) raw classification logits.
        """
        # Get DistilBERT hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, 768)

        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (B, 768)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)        # (B, 1)
        pooled = sum_hidden / count  # (B, 768)

        # Classify
        logits = self.classifier(pooled)  # (B, 3)
        return logits

    def predict_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities instead of raw logits."""
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)

    def export_onnx(self, save_path: str, seq_length: int = 256):
        """
        Export the model to ONNX format for low-latency inference.

        Args:
            save_path: Output file path (e.g., "router.onnx").
            seq_length: Fixed sequence length for the exported model.
        """
        self.eval()
        dummy_ids = torch.zeros(1, seq_length, dtype=torch.long)
        dummy_mask = torch.ones(1, seq_length, dtype=torch.long)

        torch.onnx.export(
            self,
            (dummy_ids, dummy_mask),
            save_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )
        print(f"ONNX model exported to: {save_path}")
