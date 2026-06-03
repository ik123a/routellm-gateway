"""
End-to-End Streaming Test Script
==================================
Verifies the full request lifecycle:
  1. Client sends a chat completion request with stream=True
  2. Gateway routes via heuristic router
  3. Proxy forwards to mock LLM
  4. SSE chunks stream back to client
  5. Response headers contain routing metadata

Usage:
    # Terminal 1: Start mock LLM
    python -m gateway.mock_server --port 9000

    # Terminal 2: Start gateway (configure LOCAL_MODEL_BASE_URL=http://localhost:9000/v1)
    python -m gateway.main

    # Terminal 3: Run this test
    python scripts/test_streaming.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

GATEWAY_URL = "http://localhost:8000"

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(text: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def pass_msg(test: str, detail: str = ""):
    print(f"  {GREEN}PASS{RESET}  {test} {detail}")


def fail_msg(test: str, detail: str = ""):
    print(f"  {RED}FAIL{RESET}  {test} {detail}")


def info_msg(text: str):
    print(f"  {YELLOW}INFO{RESET}  {text}")


async def test_health():
    """Test the /health endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{GATEWAY_URL}/health")
        data = resp.json()

        if resp.status_code == 200 and data.get("status") == "healthy":
            pass_msg("/health", f"router={data.get('router_mode')}")
            return True
        else:
            fail_msg("/health", str(data))
            return False


async def test_models():
    """Test the /v1/models endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{GATEWAY_URL}/v1/models")
        data = resp.json()

        model_ids = [m["id"] for m in data.get("data", [])]
        if "auto" in model_ids:
            pass_msg("/v1/models", f"models={model_ids}")
            return True
        else:
            fail_msg("/v1/models", f"'auto' not found in {model_ids}")
            return False


async def test_non_streaming():
    """Test a non-streaming chat completion."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": False,
        }
        start = time.perf_counter()
        resp = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload)
        elapsed = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            meta = data.get("_routellm", {})
            tier = meta.get("predicted_tier", "unknown")
            model = meta.get("model_used", "unknown")
            cost = meta.get("estimated_cost_usd", 0)
            pass_msg(
                "Non-streaming completion",
                f"tier={tier} model={model} cost=${cost} latency={elapsed:.0f}ms"
            )
            return True
        else:
            fail_msg("Non-streaming completion", f"status={resp.status_code}")
            return False


async def test_streaming():
    """Test a streaming (SSE) chat completion."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Explain the difference between TCP and UDP"}],
            "stream": True,
        }

        start = time.perf_counter()
        chunks_received = 0
        content_parts = []
        first_chunk_time = None

        async with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload) as resp:
            # Check response headers
            req_id = resp.headers.get("x-request-id", "none")
            model_header = resp.headers.get("x-routellm-model", "none")
            tier_header = resp.headers.get("x-routellm-tier", "none")

            info_msg(f"Response headers: request_id={req_id} model={model_header} tier={tier_header}")

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        info_msg("Received [DONE] signal")
                        break
                    try:
                        chunk = json.loads(data_str)
                        chunks_received += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                        choices = chunk.get("choices", [])
                        for choice in choices:
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                content_parts.append(content)
                    except json.JSONDecodeError:
                        pass

        elapsed = (time.perf_counter() - start) * 1000
        ttft = ((first_chunk_time - start) * 1000) if first_chunk_time else 0
        full_response = "".join(content_parts)

        if chunks_received > 0 and len(full_response) > 0:
            pass_msg(
                "Streaming completion",
                f"chunks={chunks_received} chars={len(full_response)} "
                f"TTFT={ttft:.0f}ms total={elapsed:.0f}ms"
            )
            info_msg(f"Response preview: {full_response[:100]}...")
            return True
        else:
            fail_msg("Streaming completion", f"chunks={chunks_received}")
            return False


async def test_routing_tiers():
    """Test that different queries route to different tiers."""
    test_cases = [
        ("hello!", "CHEAP"),
        ("Explain backpropagation step by step", "MEDIUM"),
        ("Implement a red-black tree in Python with insert, delete, and rotation logic", "STRONG"),
    ]

    all_passed = True
    async with httpx.AsyncClient(timeout=30.0) as client:
        for query, expected_tier in test_cases:
            payload = {
                "model": "auto",
                "messages": [{"role": "user", "content": query}],
                "stream": False,
            }
            resp = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload)

            if resp.status_code == 200:
                meta = resp.json().get("_routellm", {})
                actual_tier = meta.get("predicted_tier", "UNKNOWN")
                if actual_tier == expected_tier:
                    pass_msg(f"Routing [{expected_tier}]", f"query='{query[:40]}...'")
                else:
                    fail_msg(
                        f"Routing [{expected_tier}]",
                        f"got {actual_tier} for '{query[:40]}...'"
                    )
                    all_passed = False
            else:
                fail_msg(f"Routing [{expected_tier}]", f"HTTP {resp.status_code}")
                all_passed = False

    return all_passed


async def test_stats():
    """Test the /stats endpoint after running queries."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{GATEWAY_URL}/stats")
        data = resp.json()

        total = data.get("total_requests", 0)
        tiers = data.get("tier_distribution", {})

        if total > 0:
            pass_msg("/stats", f"requests={total} tiers={tiers}")
            return True
        else:
            fail_msg("/stats", f"total_requests={total}")
            return False


async def run_all_tests():
    """Run the complete test suite."""
    header("RouteLLM-Gateway End-to-End Test Suite")

    results = {}

    # 1. Health check
    info_msg("Testing gateway health...")
    results["health"] = await test_health()

    # 2. Models listing
    info_msg("Testing model listing...")
    results["models"] = await test_models()

    # 3. Non-streaming
    info_msg("Testing non-streaming completion...")
    results["non_streaming"] = await test_non_streaming()

    # 4. Streaming SSE
    info_msg("Testing streaming (SSE) completion...")
    results["streaming"] = await test_streaming()

    # 5. Routing tiers
    info_msg("Testing tier routing decisions...")
    results["routing"] = await test_routing_tiers()

    # 6. Stats
    info_msg("Testing stats endpoint...")
    results["stats"] = await test_stats()

    # Summary
    header("Test Results Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")

    print(f"\n  {BOLD}Total: {passed}/{total} passed{RESET}\n")

    if passed < total:
        info_msg(
            "Some tests failed. Make sure both the mock server and gateway are running:\n"
            "    Terminal 1: python -m gateway.mock_server --port 9000\n"
            "    Terminal 2: python -m gateway.main"
        )
        return 1
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
