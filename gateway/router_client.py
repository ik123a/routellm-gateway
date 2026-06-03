"""
Router Client
==============
Loads the trained DistilBERT router model (ONNX or PyTorch) and
predicts the optimal model tier for incoming queries.

During early development (before training), uses a heuristic
rule-based router as a fallback.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import numpy as np

from gateway.config import ModelTier, settings

logger = logging.getLogger("routellm.router")


# ---------------------------------------------------------------------------
# Heuristic complexity signals (used before ML router is trained)
# ---------------------------------------------------------------------------

# Patterns indicating high complexity (math, code, multi-step reasoning)
_CODE_PATTERNS = re.compile(
    r"(def |class |import |function |const |let |var |for\s*\(|while\s*\(|"
    r"```|write.*code|implement|algorithm|refactor|debug|optimize|tree|rotation|insert|delete)",
    re.IGNORECASE,
)

_MATH_PATTERNS = re.compile(
    r"(solve|equation|integral|derivative|matrix|probability|"
    r"calculate|compute|prove|theorem|\d+\s*[\+\-\*/\^]\s*\d+|"
    r"sum of|factorial|permutation|combination|log\s*\(|sqrt)",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"(explain.*step|analyze|compare.*contrast|evaluate|"
    r"what.*difference|why.*does|how.*work|reason.*about|"
    r"think.*through|chain.*of.*thought|let'?s think)",
    re.IGNORECASE,
)

_SIMPLE_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|"
    r"good morning|good night|bye|goodbye|how are you|"
    r"what'?s up|tell me a joke|who are you)\s*[!?.]*$",
    re.IGNORECASE,
)


def _heuristic_complexity_score(text: str) -> float:
    """
    Compute a rough complexity score in [0, 1] from text patterns.
    0 = trivially simple, 1 = highly complex.
    """
    # Simple greeting check — instant zero
    if _SIMPLE_PATTERNS.match(text.strip()):
        return 0.0

    score = 0.0

    # Baseline: any non-trivial query starts with a small score
    score += 0.05

    # Length signal (longer prompts tend to be more complex)
    word_count = len(text.split())
    if word_count > 200:
        score += 0.25
    elif word_count > 80:
        score += 0.15
    elif word_count > 30:
        score += 0.10
    elif word_count > 10:
        score += 0.05

    # Code patterns (high weight — code generation is expensive)
    code_matches = len(_CODE_PATTERNS.findall(text))
    score += min(0.5, code_matches * 0.2)

    # Math patterns
    math_matches = len(_MATH_PATTERNS.findall(text))
    score += min(0.4, math_matches * 0.2)

    # Reasoning patterns
    reasoning_matches = len(_REASONING_PATTERNS.findall(text))
    score += min(0.3, reasoning_matches * 0.15)

    return min(1.0, score)


def _heuristic_route(text: str) -> tuple[ModelTier, list[float]]:
    """
    Rule-based routing fallback.
    Returns (predicted_tier, probabilities_per_tier).
    """
    complexity = _heuristic_complexity_score(text)

    if complexity < 0.10:
        probs = [0.85, 0.10, 0.05]
        return ModelTier.CHEAP, probs
    elif complexity < 0.35:
        probs = [0.15, 0.70, 0.15]
        return ModelTier.MEDIUM, probs
    else:
        probs = [0.05, 0.20, 0.75]
        return ModelTier.STRONG, probs


# ---------------------------------------------------------------------------
# ONNX-based ML Router
# ---------------------------------------------------------------------------

class ONNXRouter:
    """Loads and runs the trained ONNX router model."""

    def __init__(self, model_path: str, tokenizer_name: str = "distilbert-base-uncased"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        logger.info(f"Loading ONNX router model from: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        logger.info(f"ONNX router loaded. Inputs: {self.input_names}")

    def predict(self, text: str) -> tuple[ModelTier, list[float]]:
        """
        Run inference on a query string.
        Returns (predicted_tier, softmax_probabilities).
        """
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = encoding["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = encoding["attention_mask"].astype(np.int64)

        logits = self.session.run(None, feed)[0]  # shape: (1, 3)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_logits / exp_logits.sum(axis=-1, keepdims=True))[0].tolist()

        predicted_class = int(np.argmax(probs))
        return ModelTier(predicted_class), probs


# ---------------------------------------------------------------------------
# Unified Router Interface
# ---------------------------------------------------------------------------

class RouterClient:
    """
    Unified routing interface. Tries to load the trained ONNX model;
    falls back to heuristic routing if the model file is missing.
    """

    def __init__(self):
        self._onnx_router: Optional[ONNXRouter] = None
        self._mode: str = "heuristic"
        self._load_model()

    def _load_model(self):
        model_path = settings.router_model_path
        if os.path.exists(model_path):
            try:
                self._onnx_router = ONNXRouter(model_path)
                self._mode = "onnx"
                logger.info("Router running in ONNX ML mode.")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}. Using heuristic.")
                self._mode = "heuristic"
        else:
            logger.info(
                f"ONNX model not found at {model_path}. "
                "Running in heuristic mode (train the router first)."
            )
            self._mode = "heuristic"

    def route(self, messages: list[dict]) -> tuple[ModelTier, list[float], float]:
        """
        Predict the best model tier for a list of chat messages.

        Returns:
            (predicted_tier, probability_vector, inference_time_ms)
        """
        # Concatenate all user messages for classification
        user_text = " ".join(
            msg.get("content", "")
            for msg in messages
            if msg.get("role") in ("user", "system")
        )

        start = time.perf_counter()

        if self._mode == "onnx" and self._onnx_router is not None:
            tier, probs = self._onnx_router.predict(user_text)
        else:
            tier, probs = _heuristic_route(user_text)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Router decision: tier={tier.name} probs={[f'{p:.3f}' for p in probs]} "
            f"mode={self._mode} latency={elapsed_ms:.2f}ms"
        )

        return tier, probs, elapsed_ms

    @property
    def mode(self) -> str:
        return self._mode
