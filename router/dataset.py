"""
Training Dataset Loader
========================
Builds the router training dataset by merging queries from multiple
sources (conversational, math, code) and assigning tier labels.

Sources:
  - ShareGPT (conversational, simple) -> Tier 0 (CHEAP)
  - GSM8K (grade-school math)          -> Tier 1 (MEDIUM)
  - HumanEval / MBPP (code)           -> Tier 2 (STRONG)
  - Custom hard reasoning              -> Tier 2 (STRONG)
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Optional

from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger("routellm.dataset")


# ---------------------------------------------------------------------------
# Synthetic examples for bootstrapping (before real datasets are loaded)
# ---------------------------------------------------------------------------

SYNTHETIC_EXAMPLES = [
    # Tier 0: Simple / Conversational
    {"text": "Hello! How are you?", "tier": 0},
    {"text": "What's the weather like today?", "tier": 0},
    {"text": "Tell me a joke", "tier": 0},
    {"text": "Thanks for your help!", "tier": 0},
    {"text": "Good morning!", "tier": 0},
    {"text": "Who are you?", "tier": 0},
    {"text": "What time is it?", "tier": 0},
    {"text": "Translate hello to Spanish", "tier": 0},
    {"text": "Summarize this paragraph: The cat sat on the mat.", "tier": 0},
    {"text": "What is the capital of France?", "tier": 0},
    {"text": "List 5 types of fruits", "tier": 0},
    {"text": "How do I say goodbye in Japanese?", "tier": 0},
    {"text": "Give me a fun fact", "tier": 0},
    {"text": "What color is the sky?", "tier": 0},
    {"text": "Define the word 'serendipity'", "tier": 0},

    # Tier 1: Medium complexity (factual, multi-step, light math)
    {"text": "Explain the difference between TCP and UDP protocols", "tier": 1},
    {"text": "What are the pros and cons of microservices vs monoliths?", "tier": 1},
    {"text": "Calculate the compound interest on $1000 at 5% for 3 years", "tier": 1},
    {"text": "Explain how a neural network learns through backpropagation", "tier": 1},
    {"text": "Write a SQL query to find the top 5 customers by total spend", "tier": 1},
    {"text": "What is the time complexity of merge sort and why?", "tier": 1},
    {"text": "Explain the CAP theorem in distributed systems", "tier": 1},
    {"text": "Solve: If 3x + 7 = 22, what is x?", "tier": 1},
    {"text": "Compare and contrast REST and GraphQL APIs", "tier": 1},
    {"text": "Explain how HTTPS encryption works step by step", "tier": 1},
    {"text": "What is the difference between a stack and a queue?", "tier": 1},
    {"text": "Explain the P vs NP problem in simple terms", "tier": 1},
    {"text": "How does garbage collection work in Java?", "tier": 1},
    {"text": "Describe the OSI model layers and their functions", "tier": 1},
    {"text": "What are design patterns? Explain the Observer pattern", "tier": 1},

    # Tier 2: High complexity (code generation, proofs, multi-step reasoning)
    {"text": "Implement a red-black tree in Python with insert, delete, and search operations. Include rotation logic and color-fixing.", "tier": 2},
    {"text": "Write a dynamic programming solution to find the longest common subsequence of three strings. Analyze the time and space complexity.", "tier": 2},
    {"text": "Implement a concurrent hash map in C++ that supports lock-free reads and fine-grained locking for writes.", "tier": 2},
    {"text": "Prove that the halting problem is undecidable using a diagonalization argument.", "tier": 2},
    {"text": "Design a distributed consensus algorithm similar to Raft. Write pseudocode for leader election and log replication.", "tier": 2},
    {"text": "Implement a mini compiler that tokenizes, parses, and evaluates arithmetic expressions with parentheses and operator precedence.", "tier": 2},
    {"text": "Write a Python implementation of the A* pathfinding algorithm with a custom heuristic for a weighted grid. Include visualization.", "tier": 2},
    {"text": "Implement an attention mechanism from scratch in PyTorch. Include multi-head attention, positional encoding, and masking.", "tier": 2},
    {"text": "Design a rate limiter using the token bucket algorithm. It should support distributed rate limiting across multiple servers using Redis.", "tier": 2},
    {"text": "Write a garbage collector in C using mark-and-sweep. Handle circular references and memory fragmentation.", "tier": 2},
    {"text": "Implement a B+ tree with bulk loading, range queries, and node splitting. Analyze the I/O complexity for disk-based operations.", "tier": 2},
    {"text": "Build a simple neural network framework from scratch with automatic differentiation (autograd). Support add, multiply, ReLU, and cross-entropy loss.", "tier": 2},
    {"text": "Solve this optimization problem: minimize f(x,y) = x^2 + y^2 subject to x + y >= 10 using Lagrange multipliers. Show all steps.", "tier": 2},
    {"text": "Write a MapReduce implementation in Python that processes a large text corpus to compute TF-IDF scores across documents.", "tier": 2},
    {"text": "Implement Dijkstra's algorithm with a Fibonacci heap. Compare its performance against a binary heap implementation on large sparse graphs.", "tier": 2},
]


class RouterDataset(Dataset):
    """
    PyTorch dataset for training the router classifier.

    Each example contains:
      - input_ids: tokenized text
      - attention_mask: padding mask
      - label: tier index (0, 1, or 2)
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.examples = examples

        # Validate labels
        for ex in self.examples:
            assert "text" in ex and "tier" in ex, f"Invalid example: {ex}"
            assert ex["tier"] in (0, 1, 2), f"Invalid tier: {ex['tier']}"

        logger.info(
            f"RouterDataset initialized with {len(self.examples)} examples. "
            f"Distribution: {self._tier_distribution()}"
        )

    def _tier_distribution(self) -> dict[int, int]:
        counts = {0: 0, 1: 0, 2: 0}
        for ex in self.examples:
            counts[ex["tier"]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": ex["tier"],
        }


def load_synthetic_dataset(
    seed: int = 42,
    train_ratio: float = 0.8,
) -> tuple[RouterDataset, RouterDataset]:
    """
    Load the built-in synthetic examples and split into train/val.
    Use this for initial development before real datasets are available.
    """
    random.seed(seed)
    examples = SYNTHETIC_EXAMPLES.copy()
    random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    return RouterDataset(train_examples), RouterDataset(val_examples)


def load_jsonl_dataset(
    filepath: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[RouterDataset, RouterDataset]:
    """
    Load examples from a JSONL file.
    Each line should have: {"text": "...", "tier": 0|1|2}
    """
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    random.seed(seed)
    random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    logger.info(f"Loaded {len(examples)} examples from {filepath}")
    return RouterDataset(train_examples), RouterDataset(val_examples)
