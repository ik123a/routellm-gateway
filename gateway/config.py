"""
Gateway Configuration Module
=============================
Defines model tiers, cost tables, routing thresholds, and
application settings loaded from environment variables.
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class ModelTier(IntEnum):
    """
    Model tiers ranked by capability and cost.
    Tier 0 = cheapest / weakest, Tier 2 = most expensive / strongest.
    """
    CHEAP = 0    # Local Llama-3-8B / GPT-4o-mini equivalent
    MEDIUM = 1   # Mixtral-8x7B / GPT-4o-mini
    STRONG = 2   # GPT-4o / Claude-3.5-Sonnet


# ---------------------------------------------------------------------------
# Per-model configuration: endpoint, pricing, and metadata
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    # ---- Tier 0: Cheap / Local ----
    "llama3-8b": {
        "tier": ModelTier.CHEAP,
        "provider": "local",           # vLLM / Ollama
        "model_id": "llama3:8b",
        "cost_per_1k_input": 0.0,      # Self-hosted
        "cost_per_1k_output": 0.0,
        "max_tokens": 4096,
        "supports_streaming": True,
    },
    # ---- Tier 1: Medium ----
    "gpt-4o-mini": {
        "tier": ModelTier.MEDIUM,
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "cost_per_1k_input": 0.00015,   # $0.15 / 1M input
        "cost_per_1k_output": 0.0006,   # $0.60 / 1M output
        "max_tokens": 16384,
        "supports_streaming": True,
    },
    "llama3-70b-nvidia": {
        "tier": ModelTier.MEDIUM,
        "provider": "nvidia",
        "model_id": "meta/llama3-70b-instruct",
        "cost_per_1k_input": 0.00088,   # $0.88 / 1M input on NVIDIA NIM
        "cost_per_1k_output": 0.00088,
        "max_tokens": 8192,
        "supports_streaming": True,
    },
    # ---- Tier 2: Strong ----
    "gpt-4o": {
        "tier": ModelTier.STRONG,
        "provider": "openai",
        "model_id": "gpt-4o",
        "cost_per_1k_input": 0.0025,    # $2.50 / 1M input
        "cost_per_1k_output": 0.01,     # $10.00 / 1M output
        "max_tokens": 16384,
        "supports_streaming": True,
    },
}


def get_models_for_tier(tier: ModelTier) -> list[str]:
    """Return all model keys belonging to a given tier."""
    return [k for k, v in MODEL_REGISTRY.items() if v["tier"] == tier]


def get_cheapest_model(tier: ModelTier) -> str:
    """Return the cheapest model key in the given tier."""
    models = get_models_for_tier(tier)
    if not models:
        raise ValueError(f"No models registered for tier {tier}")
    return min(models, key=lambda k: MODEL_REGISTRY[k]["cost_per_1k_input"])


def get_fallback_tier(current_tier: ModelTier) -> Optional[ModelTier]:
    """Return the next higher tier for fallback routing, or None."""
    if current_tier < ModelTier.STRONG:
        return ModelTier(current_tier + 1)
    return None


# ---------------------------------------------------------------------------
# Application-wide settings from environment / .env file
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application settings. Values are loaded from environment variables
    or a .env file in the project root.
    """

    # --- API Keys ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    nvidia_api_key: str = Field(default="", description="NVIDIA API key for NIM endpoints")

    # --- Local Model ---
    local_model_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for local model (Ollama / vLLM)",
    )
    local_model_name: str = Field(default="llama3:8b")

    # --- Redis ---
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    # --- Gateway ---
    gateway_host: str = Field(default="0.0.0.0")
    gateway_port: int = Field(default=8000)
    log_level: str = Field(default="info")
    dev_mode: bool = Field(
        default=False,
        description="When true, routes ALL tiers to the local mock server",
    )

    # --- Router ---
    router_model_path: str = Field(
        default="./router/checkpoints/router.onnx",
        description="Path to the exported ONNX router model",
    )
    cost_sensitivity: float = Field(
        default=0.01,
        description="Lambda for cost-penalized routing (higher = more cost-averse)",
    )

    # --- Cache ---
    cache_enabled: bool = Field(default=True)
    cache_similarity_threshold: float = Field(
        default=0.96,
        description="Cosine similarity threshold for semantic cache hits",
    )
    cache_ttl_seconds: int = Field(
        default=86400 * 7,  # 7 days
        description="Time-to-live for cached entries in seconds",
    )

    # --- Fallback ---
    max_fallback_retries: int = Field(
        default=2,
        description="Max number of tier escalations on failure",
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout per LLM request in seconds",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton settings instance
settings = Settings()


# ---------------------------------------------------------------------------
# Dev mode: redirect all providers to the local mock server
# ---------------------------------------------------------------------------

def _apply_dev_mode():
    """In dev mode, override all providers to use the local endpoint."""
    if not settings.dev_mode:
        return
    import logging
    logger = logging.getLogger("routellm.config")
    logger.info("DEV_MODE enabled — all tiers redirected to local mock server.")
    for key in MODEL_REGISTRY:
        MODEL_REGISTRY[key]["provider"] = "local"

_apply_dev_mode()
