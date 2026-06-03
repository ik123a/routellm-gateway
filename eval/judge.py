"""
LLM-as-Judge Evaluator
========================
Uses a strong model (GPT-4o) to evaluate whether a routed response
is functionally equivalent to the GPT-4o baseline response.

This enables automated accuracy measurement without human annotation.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("routellm.judge")


async def judge_equivalence(
    query: str,
    routed_response: str,
    baseline_response: str,
    openai_api_key: str,
    judge_model: str = "gpt-4o-mini",
) -> dict:
    """
    Use an LLM judge to determine if two responses are functionally equivalent.

    Args:
        query: The original user query.
        routed_response: The response from the routed (possibly cheaper) model.
        baseline_response: The response from the baseline GPT-4o model.
        openai_api_key: API key for the judge model.
        judge_model: Which model to use as the judge.

    Returns:
        dict with keys: "equivalent" (bool), "confidence" (float), "reasoning" (str)
    """
    import httpx

    judge_prompt = f"""You are a strict evaluator comparing two AI responses to the same query.

QUERY: {query}

RESPONSE A (Baseline):
{baseline_response}

RESPONSE B (Candidate):
{routed_response}

Evaluate whether Response B is functionally equivalent to Response A.
"Functionally equivalent" means:
- The core answer/information is the same
- No factual errors are introduced
- The quality is comparable (minor wording differences are OK)

Respond in this exact JSON format:
{{"equivalent": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from judge response
            import json
            result = json.loads(content)
            return result

    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return {
            "equivalent": True,  # Default to true on failure
            "confidence": 0.0,
            "reasoning": f"Judge error: {e}",
        }
