"""
Router Training Script
=======================
Trains the DistilBERT router classifier using a cost-penalized
cross-entropy loss.

Usage:
    python -m router.train                          # Synthetic data
    python -m router.train --data path/to/data.jsonl  # Custom data

The trained model is exported to ONNX for production inference.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from router.dataset import load_jsonl_dataset, load_synthetic_dataset
from router.model import RouterModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("routellm.train")


# Cost vector: relative cost per tier [CHEAP, MEDIUM, STRONG]
TIER_COSTS = torch.tensor([0.0, 0.15, 2.50], dtype=torch.float32)


class CostPenalizedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with a cost penalty term.

    L = CE(logits, labels) + lambda * sum(cost_i * softmax_i)

    This encourages the model to prefer cheaper tiers when
    the classification boundary is ambiguous.
    """

    def __init__(self, cost_vector: torch.Tensor, cost_lambda: float = 0.01):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.cost_vector = cost_vector
        self.cost_lambda = cost_lambda

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, labels)

        # Cost penalty: encourage routing to cheaper tiers
        probs = torch.softmax(logits, dim=-1)
        cost_vec = self.cost_vector.to(logits.device)
        cost_penalty = (probs * cost_vec.unsqueeze(0)).sum(dim=-1).mean()

        return ce_loss + self.cost_lambda * cost_penalty


def train(
    data_path: str | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    cost_lambda: float = 0.01,
    checkpoint_dir: str = "./router/checkpoints",
    use_wandb: bool = False,
):
    """Run the full training loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # ---- Load dataset ----
    if data_path and os.path.exists(data_path):
        train_ds, val_ds = load_jsonl_dataset(data_path)
    else:
        logger.info("No data file provided. Using synthetic bootstrap dataset.")
        train_ds, val_ds = load_synthetic_dataset()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ---- Initialize model ----
    model = RouterModel(
        backbone="distilbert-base-uncased",
        num_tiers=3,
        hidden_dim=256,
        dropout=0.3,
        freeze_backbone=False,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {param_count:,}")

    # ---- Loss, optimizer, scheduler ----
    criterion = CostPenalizedCrossEntropy(TIER_COSTS, cost_lambda=cost_lambda)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Optional: Weights & Biases ----
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="routellm-gateway",
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "cost_lambda": cost_lambda,
                    "param_count": param_count,
                },
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}. Continuing without.")
            use_wandb = False

    # ---- Training loop ----
    best_val_acc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.perf_counter()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = torch.tensor(batch["label"], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = torch.tensor(batch["label"], dtype=torch.long).to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        elapsed = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # ---- Save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    # ---- Export to ONNX ----
    logger.info("Exporting best model to ONNX...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt")))
    model.to("cpu")
    onnx_path = os.path.join(checkpoint_dir, "router.onnx")
    model.export_onnx(onnx_path)

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"ONNX model saved to: {onnx_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RouteLLM router classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to JSONL dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--cost-lambda", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", type=str, default="./router/checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        cost_lambda=args.cost_lambda,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
    )
