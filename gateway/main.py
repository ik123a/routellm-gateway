"""
RouteLLM-Gateway: Main Application
====================================
OpenAI-compatible API gateway that intercepts chat completion requests,
routes them through the cost-aware classifier, and streams responses
back to the client with full observability.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from gateway.cache import SemanticCache
from gateway.config import MODEL_REGISTRY, ModelTier, settings
from gateway.middleware import RequestIDMiddleware, StructuredLoggingMiddleware
from gateway.proxy import proxy_with_fallback
from gateway.router_client import RouterClient

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("routellm.main")


# ---------------------------------------------------------------------------
# Application lifespan: init singletons on startup
# ---------------------------------------------------------------------------

router_client: Optional[RouterClient] = None
cache: Optional[SemanticCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_client, cache
    logger.info("=" * 60)
    logger.info("  RouteLLM-Gateway starting up...")
    logger.info("=" * 60)

    # Initialize the router classifier
    router_client = RouterClient()
    logger.info(f"Router mode: {router_client.mode}")

    # Initialize the semantic cache
    cache = SemanticCache()
    logger.info(f"Cache available: {cache.is_available}")

    logger.info(f"Registered models: {list(MODEL_REGISTRY.keys())}")
    logger.info(f"Cost sensitivity (lambda): {settings.cost_sensitivity}")
    logger.info("Gateway ready to accept requests.")
    logger.info("=" * 60)

    yield

    logger.info("Gateway shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RouteLLM-Gateway",
    description=(
        "A cost-aware LLM API gateway that dynamically routes queries "
        "to the most cost-effective model while preserving accuracy."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracing and structured logging
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="One of: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="auto",
        description="Model name or 'auto' for dynamic routing",
    )
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Observability: per-request metrics
# ---------------------------------------------------------------------------

_request_count = 0
_total_cost = 0.0
_tier_counts = {t.name: 0 for t in ModelTier}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "router_mode": router_client.mode if router_client else "unloaded",
        "cache_available": cache.is_available if cache else False,
        "models_registered": len(MODEL_REGISTRY),
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = []
    for key, info in MODEL_REGISTRY.items():
        models.append({
            "id": key,
            "object": "model",
            "owned_by": info["provider"],
            "tier": info["tier"].name,
            "cost_per_1k_input": info["cost_per_1k_input"],
            "cost_per_1k_output": info["cost_per_1k_output"],
        })
    # Include the "auto" routing model
    models.insert(0, {
        "id": "auto",
        "object": "model",
        "owned_by": "routellm",
        "tier": "DYNAMIC",
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
    })
    return {"object": "list", "data": models}


@app.get("/stats")
async def get_stats():
    """Return gateway routing statistics."""
    return {
        "total_requests": _request_count,
        "total_estimated_cost_usd": round(_total_cost, 6),
        "tier_distribution": _tier_counts,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.

    If model="auto" (default), the gateway routes dynamically.
    If a specific model key is given (e.g., "gpt-4o"), it is used directly.
    """
    global _request_count, _total_cost

    _request_count += 1
    start_time = time.perf_counter()

    messages = [msg.model_dump() for msg in request.messages]

    # ---- Extract user prompt for caching / routing ----
    user_prompt = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    )

    # ---- Check cache (non-streaming only) ----
    if not request.stream and cache and cache.is_available:
        cached = cache.get(user_prompt)
        if cached:
            logger.info("Serving response from cache.")
            # Inject routing metadata
            cached["_routellm"] = {
                "source": "cache",
                "latency_ms": round(
                    (time.perf_counter() - start_time) * 1000, 2
                ),
            }
            return JSONResponse(content=cached)

    # ---- Determine target tier ----
    if request.model == "auto":
        # Dynamic routing via classifier
        tier, probs, router_latency = router_client.route(messages)
    else:
        # Explicit model selection
        model_info = MODEL_REGISTRY.get(request.model)
        if not model_info:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. "
                       f"Available: {list(MODEL_REGISTRY.keys()) + ['auto']}",
            )
        tier = model_info["tier"]
        probs = [0.0, 0.0, 0.0]
        probs[tier] = 1.0
        router_latency = 0.0

    _tier_counts[tier.name] = _tier_counts.get(tier.name, 0) + 1

    # ---- Proxy to upstream ----
    try:
        response, result = await proxy_with_fallback(
            initial_tier=tier,
            messages=messages,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        logger.error(f"All upstream providers failed: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    _total_cost += result.estimated_cost

    # ---- Build routing metadata ----
    routing_meta = {
        "router_mode": router_client.mode,
        "predicted_tier": tier.name,
        "tier_probabilities": {
            "CHEAP": round(probs[0], 4),
            "MEDIUM": round(probs[1], 4),
            "STRONG": round(probs[2], 4),
        },
        "router_latency_ms": round(router_latency, 2),
        "proxy_latency_ms": round(result.latency_ms, 2),
        "total_latency_ms": round(
            (time.perf_counter() - start_time) * 1000, 2
        ),
        "model_used": result.model_key,
        "estimated_cost_usd": round(result.estimated_cost, 6),
        "status": result.status,
    }

    # ---- Streaming response ----
    if request.stream:
        async def _stream_with_meta():
            async for chunk in response:
                yield chunk

        logger.info(
            f"Streaming response | {json.dumps(routing_meta)}"
        )
        return StreamingResponse(
            _stream_with_meta(),
            media_type="text/event-stream",
            headers={
                "X-RouteLLM-Model": result.model_key,
                "X-RouteLLM-Tier": tier.name,
                "X-RouteLLM-Router-Latency-Ms": str(round(router_latency, 2)),
            },
        )

    # ---- Non-streaming response ----
    # Inject routing metadata into the response
    if isinstance(response, dict):
        response["_routellm"] = routing_meta

        # Cache the response
        if cache and cache.is_available:
            cache.store(user_prompt, response)

    logger.info(
        f"Request complete | {json.dumps(routing_meta)}"
    )

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level=settings.log_level,
        reload=True,
    )
