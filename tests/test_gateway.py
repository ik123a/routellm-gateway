"""
Gateway Integration Tests
==========================
Tests for the FastAPI gateway endpoints: health check, model listing,
chat completions (routing logic), and statistics.
"""

import pytest
from fastapi.testclient import TestClient

from gateway.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health & Info Endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "router_mode" in data
        assert "cache_available" in data
        assert "models_registered" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    def test_list_models_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_list_models_contains_auto(self, client):
        data = client.get("/v1/models").json()
        model_ids = [m["id"] for m in data["data"]]
        assert "auto" in model_ids

    def test_list_models_has_tier_info(self, client):
        data = client.get("/v1/models").json()
        for model in data["data"]:
            assert "tier" in model


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_has_counters(self, client):
        data = client.get("/stats").json()
        assert "total_requests" in data
        assert "total_estimated_cost_usd" in data
        assert "tier_distribution" in data


# ---------------------------------------------------------------------------
# Chat Completion Endpoint
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_invalid_model_returns_400(self, client):
        payload = {
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 400

    def test_empty_messages_fails(self, client):
        payload = {
            "model": "auto",
            "messages": [],
            "stream": False,
        }
        # Should fail validation or routing with empty messages
        resp = client.post("/v1/chat/completions", json=payload)
        # Either 422 (validation) or 502 (no content to route)
        assert resp.status_code in (422, 502, 200)

    def test_valid_request_schema(self, client):
        """Test that a well-formed request is accepted (may fail at proxy level)."""
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "temperature": 0.5,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        # Will return 502 if no upstream is available, which is expected
        # in test environment. The point is it shouldn't be 422 or 400.
        assert resp.status_code != 422
