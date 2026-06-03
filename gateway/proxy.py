"""
Streaming Proxy Handler
========================
Handles forwarding of chat completion requests to upstream LLM
providers (OpenAI, Anthropic, local vLLM/Ollama) and streams
responses back to the client using Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Optional

import httpx

from gateway.config import MODEL_REGISTRY, ModelTier, get_fallback_tier, settings

logger = logging.getLogger("routellm.proxy")


# ---------------------------------------------------------------------------
# Provider-specific client factories
# ---------------------------------------------------------------------------

def _build_openai_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }


def _build_nvidia_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }


def _build_local_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "local": f"{settings.local_model_base_url}/chat/completions",
}


# ---------------------------------------------------------------------------
# Request/response types mirroring OpenAI schema
# ---------------------------------------------------------------------------

class ProxyResult:
    """Encapsulates the result of a proxied LLM call."""

    def __init__(
        self,
        model_key: str,
        tier: ModelTier,
        status: str,            # "success" | "fallback" | "error"
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float = 0.0,
    ):
        self.model_key = model_key
        self.tier = tier
        self.status = status
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.estimated_cost = estimated_cost

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "tier": int(self.tier),
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 6),
        }


# ---------------------------------------------------------------------------
# Core proxy functions
# ---------------------------------------------------------------------------

def _get_headers(provider: str) -> dict[str, str]:
    if provider == "openai":
        return _build_openai_headers()
    elif provider == "nvidia":
        return _build_nvidia_headers()
    return _build_local_headers()


def _get_endpoint(provider: str) -> str:
    endpoint = PROVIDER_ENDPOINTS.get(provider)
    if not endpoint:
        raise ValueError(f"Unknown provider: {provider}")
    return endpoint


def _estimate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost for a given request."""
    model_info = MODEL_REGISTRY.get(model_key)
    if not model_info:
        return 0.0
    input_cost = (input_tokens / 1000) * model_info["cost_per_1k_input"]
    output_cost = (output_tokens / 1000) * model_info["cost_per_1k_output"]
    return input_cost + output_cost


async def proxy_chat_completion(
    model_key: str,
    messages: list[dict],
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> tuple[AsyncIterator[str] | dict, ProxyResult]:
    """
    Send a chat completion request to the upstream provider.

    If `stream=True`, returns an async iterator of SSE-formatted chunks
    and a ProxyResult with partial metrics (tokens counted post-hoc).

    If `stream=False`, returns the full JSON response dict and ProxyResult.
    """
    model_info = MODEL_REGISTRY.get(model_key)
    if not model_info:
        raise ValueError(f"Unknown model: {model_key}")

    provider = model_info["provider"]
    endpoint = _get_endpoint(provider)
    headers = _get_headers(provider)

    payload = {
        "model": model_info["model_id"],
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    start_time = time.perf_counter()

    if stream:
        return await _stream_request(
            endpoint, headers, payload, model_key, model_info, start_time
        )
    else:
        return await _sync_request(
            endpoint, headers, payload, model_key, model_info, start_time
        )


async def _stream_request(
    endpoint: str,
    headers: dict,
    payload: dict,
    model_key: str,
    model_info: dict,
    start_time: float,
) -> tuple[AsyncIterator[str], ProxyResult]:
    """Stream SSE chunks from the upstream provider."""

    async def _chunk_iterator() -> AsyncIterator[str]:
        output_tokens = 0
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                chunk = json.loads(data)
                                # Count output tokens from delta content
                                choices = chunk.get("choices", [])
                                for choice in choices:
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        # Rough token estimate: 1 token ~ 4 chars
                                        output_tokens += max(1, len(content) // 4)
                            except json.JSONDecodeError:
                                pass
                            yield f"data: {data}\n\n"
        except httpx.TimeoutException as e:
            logger.error(f"Stream timeout midway for model {model_key}: {e}")
            error_chunk = json.dumps({"choices": [{"delta": {"content": "\n\n[Network Timeout: Partial Response Received]"}}]})
            yield f"data: {error_chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error midway for model {model_key}: {e}")
            error_chunk = json.dumps({"choices": [{"delta": {"content": f"\n\n[Stream Error: {e}]"}}]})
            yield f"data: {error_chunk}\n\n"
            yield "data: [DONE]\n\n"

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    result = ProxyResult(
        model_key=model_key,
        tier=model_info["tier"],
        status="success",
        latency_ms=elapsed_ms,
    )
    return _chunk_iterator(), result


async def _sync_request(
    endpoint: str,
    headers: dict,
    payload: dict,
    model_key: str,
    model_info: dict,
    start_time: float,
) -> tuple[dict, ProxyResult]:
    """Non-streaming synchronous request to the upstream provider."""
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Extract token usage from response
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = _estimate_cost(model_key, input_tokens, output_tokens)

    result = ProxyResult(
        model_key=model_key,
        tier=model_info["tier"],
        status="success",
        latency_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=cost,
    )
    return data, result


async def proxy_with_fallback(
    initial_tier: ModelTier,
    messages: list[dict],
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> tuple[AsyncIterator[str] | dict, ProxyResult]:
    """
    Attempt to proxy a request starting at `initial_tier`.
    If the upstream returns a 429 (rate limit) or 5xx error,
    automatically escalate to the next tier.

    Returns the response (streaming iterator or dict) and ProxyResult.
    """
    current_tier = initial_tier
    retries = 0

    while retries <= settings.max_fallback_retries:
        # Pick the cheapest model in the current tier
        from gateway.config import get_cheapest_model
        model_key = get_cheapest_model(current_tier)

        try:
            logger.info(
                f"Routing to model={model_key} tier={current_tier.name} "
                f"(attempt {retries + 1})"
            )
            response, result = await proxy_chat_completion(
                model_key=model_key,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response, result

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.warning(
                f"Model {model_key} returned HTTP {status_code}. "
                f"Attempting fallback..."
            )
            if status_code in (429, 500, 502, 503, 504):
                next_tier = get_fallback_tier(current_tier)
                if next_tier is not None:
                    current_tier = next_tier
                    retries += 1
                    continue
            # Non-retriable error or no higher tier available
            raise

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(
                f"Connection error for model {model_key}: {e}. "
                f"Attempting fallback..."
            )
            next_tier = get_fallback_tier(current_tier)
            if next_tier is not None:
                current_tier = next_tier
                retries += 1
                continue
            raise

    # Should not reach here, but just in case
    raise RuntimeError("All fallback tiers exhausted")
