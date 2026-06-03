"""
Router Model Unit Tests
========================
Tests for the heuristic router, dataset loader, and model architecture.
"""

import pytest
import torch

from gateway.config import ModelTier
from gateway.router_client import (
    RouterClient,
    _heuristic_complexity_score,
    _heuristic_route,
)
from router.dataset import SYNTHETIC_EXAMPLES, RouterDataset, load_synthetic_dataset
from router.model import RouterModel


# ---------------------------------------------------------------------------
# Heuristic Router Tests
# ---------------------------------------------------------------------------

class TestHeuristicRouter:
    def test_simple_greeting_routes_cheap(self):
        tier, probs = _heuristic_route("Hi, how are you?")
        assert tier == ModelTier.CHEAP

    def test_simple_goodbye_routes_cheap(self):
        tier, probs = _heuristic_route("goodbye!")
        assert tier == ModelTier.CHEAP

    def test_math_question_routes_medium_or_strong(self):
        tier, probs = _heuristic_route("Solve the equation 3x + 7 = 22")
        assert tier in (ModelTier.MEDIUM, ModelTier.STRONG)

    def test_code_request_routes_strong(self):
        query = (
            "Implement a red-black tree in Python with insert, delete, "
            "and rotations. Include all edge cases."
        )
        tier, probs = _heuristic_route(query)
        assert tier == ModelTier.STRONG

    def test_probabilities_sum_to_one(self):
        _, probs = _heuristic_route("Explain backpropagation step by step")
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_complexity_score_range(self):
        score = _heuristic_complexity_score("hello")
        assert 0.0 <= score <= 1.0

        score = _heuristic_complexity_score(
            "Write a distributed consensus algorithm with log replication"
        )
        assert 0.0 <= score <= 1.0


class TestRouterClient:
    def test_client_initializes_in_heuristic_mode(self):
        client = RouterClient()
        assert client.mode == "heuristic"

    def test_route_returns_three_values(self):
        client = RouterClient()
        messages = [{"role": "user", "content": "hello"}]
        tier, probs, latency = client.route(messages)
        assert isinstance(tier, ModelTier)
        assert len(probs) == 3
        assert latency >= 0


# ---------------------------------------------------------------------------
# Dataset Tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_synthetic_examples_are_valid(self):
        for ex in SYNTHETIC_EXAMPLES:
            assert "text" in ex
            assert "tier" in ex
            assert ex["tier"] in (0, 1, 2)

    def test_synthetic_dataset_loads(self):
        train_ds, val_ds = load_synthetic_dataset()
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(train_ds) + len(val_ds) == len(SYNTHETIC_EXAMPLES)

    def test_dataset_item_has_required_keys(self):
        train_ds, _ = load_synthetic_dataset()
        item = train_ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item

    def test_dataset_item_shapes(self):
        train_ds, _ = load_synthetic_dataset()
        item = train_ds[0]
        assert item["input_ids"].shape == (256,)
        assert item["attention_mask"].shape == (256,)


# ---------------------------------------------------------------------------
# Model Architecture Tests
# ---------------------------------------------------------------------------

class TestRouterModel:
    @pytest.fixture
    def model(self):
        return RouterModel(
            backbone="distilbert-base-uncased",
            num_tiers=3,
            hidden_dim=64,  # Smaller for faster testing
            dropout=0.1,
            freeze_backbone=True,
        )

    def test_model_output_shape(self, model):
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        logits = model(input_ids, attention_mask)
        assert logits.shape == (batch_size, 3)

    def test_model_predict_probs_sums_to_one(self, model):
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        probs = model.predict_probs(input_ids, attention_mask)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_model_trainable_params(self, model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # With frozen backbone, only classifier head params should be trainable
        assert trainable > 0
        # Classifier head: 768*64 + 64 + 64*3 + 3 = ~49,411
        assert trainable < 100_000
