"""
Mock LLM Server
=================
A lightweight FastAPI server that mimics the OpenAI /v1/chat/completions
endpoint. Supports both streaming (SSE) and non-streaming modes.

Used for local development and integration testing without burning
real API credits.

Usage:
    python -m gateway.mock_server          # Starts on port 9000
    python -m gateway.mock_server --port 9001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mock_llm")

app = FastAPI(title="Mock LLM Server", version="1.0")


# ---------------------------------------------------------------------------
# Pre-built responses by query complexity
# ---------------------------------------------------------------------------

MOCK_RESPONSES = {
    "simple": "Hello! I'm doing great, thanks for asking. How can I help you today?",
    "medium": (
        "TCP (Transmission Control Protocol) is connection-oriented and ensures "
        "reliable, ordered delivery of data. UDP (User Datagram Protocol) is "
        "connectionless, offering faster but unreliable data transfer. TCP is "
        "used for web browsing and email, while UDP is preferred for streaming "
        "and gaming where speed matters more than reliability."
    ),
    "complex": (
        "```python\n"
        "class TrieNode:\n"
        "    def __init__(self):\n"
        "        self.children = {}\n"
        "        self.is_end = False\n\n"
        "class Trie:\n"
        "    def __init__(self):\n"
        "        self.root = TrieNode()\n\n"
        "    def insert(self, word: str) -> None:\n"
        "        node = self.root\n"
        "        for char in word:\n"
        "            if char not in node.children:\n"
        "                node.children[char] = TrieNode()\n"
        "            node = node.children[char]\n"
        "        node.is_end = True\n\n"
        "    def search(self, word: str) -> bool:\n"
        "        node = self.root\n"
        "        for char in word:\n"
        "            if char not in node.children:\n"
        "                return False\n"
        "            node = node.children[char]\n"
        "        return node.is_end\n\n"
        "    def startsWith(self, prefix: str) -> bool:\n"
        "        node = self.root\n"
        "        for char in prefix:\n"
        "            if char not in node.children:\n"
        "                return False\n"
        "            node = node.children[char]\n"
        "        return True\n"
        "```\n\n"
        "Time complexity: O(m) for all operations where m is the key length. "
        "Space complexity: O(n * m) where n is the number of keys."
    ),
}


def _classify_query(text: str) -> str:
    """Rough classification for selecting mock response."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["implement", "write", "code", "algorithm", "design"]):
        return "complex"
    if any(kw in text_lower for kw in ["explain", "compare", "difference", "how", "solve"]):
        return "medium"
    return "simple"


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return max(1, len(text.split()) * 4 // 3)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-llm"
    messages: list[ChatMessage] = []
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "server": "mock-llm"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "mock-llm", "object": "model", "owned_by": "local"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Mock OpenAI chat completions endpoint."""

    # Extract user message
    user_text = ""
    for msg in request.messages:
        if msg.role == "user":
            user_text = msg.content
            break

    # Select response based on complexity
    complexity = _classify_query(user_text)
    response_text = MOCK_RESPONSES.get(complexity, MOCK_RESPONSES["simple"])

    logger.info(f"Mock LLM | complexity={complexity} | stream={request.stream} | query={user_text[:60]}...")

    completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
    model_name = request.model or "mock-llm"

    if request.stream:
        return StreamingResponse(
            _stream_response(completion_id, model_name, response_text),
            media_type="text/event-stream",
        )

    # Non-streaming response
    prompt_tokens = _estimate_tokens(user_text)
    completion_tokens = _estimate_tokens(response_text)

    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


async def _stream_response(
    completion_id: str,
    model_name: str,
    response_text: str,
):
    """Generate SSE-formatted streaming chunks, mimicking OpenAI."""

    # Split response into word-level chunks for realistic streaming
    words = response_text.split(" ")
    chunk_size = 3  # words per chunk

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        content = " ".join(chunk_words)
        if i > 0:
            content = " " + content

        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.03)  # Simulate network latency

    # Final chunk with finish_reason
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Mock LLM Server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logger.info(f"Starting Mock LLM server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
