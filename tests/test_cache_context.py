"""Exercise the actual HTTP handler with synthetic cache/provider boundaries."""
import copy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import gateway.main as main
from gateway.config import ModelTier


class MemoryCache:
    is_available = True

    def __init__(self):
        self.values = {}

    def get(self, key):
        return copy.deepcopy(self.values.get(key))

    get_exact = get

    def store(self, key, value):
        self.values[key] = copy.deepcopy(value)

    set_exact = store


@pytest.fixture
def offline_gateway(monkeypatch):
    calls = []
    cache = MemoryCache()
    router = SimpleNamespace(mode='offline-test', route=lambda messages: (ModelTier.CHEAP, [1., 0., 0.], 0.))
    monkeypatch.setattr(main, 'SemanticCache', lambda **kwargs: cache)
    monkeypatch.setattr(main, 'RouterClient', lambda: router)

    async def provider(**kwargs):
        calls.append(kwargs)
        response = {'choices': [{'message': {'role': 'assistant', 'content': 'synthetic-response-' + str(len(calls))}}]}
        result = SimpleNamespace(estimated_cost=0., latency_ms=0., model_key='synthetic-model', status='success')
        return response, result

    monkeypatch.setattr(main, 'proxy_with_fallback', provider)
    with TestClient(main.app) as client:
        yield client, calls


BASE = {'model': 'auto', 'messages': [{'role': 'user', 'content': 'same question'}], 'stream': False}


@pytest.mark.parametrize('change', [
    {'messages': [{'role': 'system', 'content': 'synthetic-private-context'}, *BASE['messages']]},
    {'messages': [{'role': 'assistant', 'content': 'different prior answer'}, *BASE['messages']]},
    {'model': 'gpt-4o'},
    {'temperature': 0.1},
    {'max_tokens': 12},
])
def test_context_and_generation_parameters_do_not_share_cached_responses(offline_gateway, change):
    client, calls = offline_gateway
    first = client.post('/v1/chat/completions', json=BASE)
    second = client.post('/v1/chat/completions', json={**BASE, **change})
    assert first.status_code == second.status_code == 200
    assert len(calls) == 2
    assert first.json()['choices'] != second.json()['choices']


def test_identical_full_request_can_use_exact_cache(offline_gateway):
    client, calls = offline_gateway
    first = client.post('/v1/chat/completions', json=BASE)
    second = client.post('/v1/chat/completions', json=BASE)
    assert first.status_code == second.status_code == 200
    assert len(calls) == 1
    assert second.json()['_routellm']['source'] == 'cache'


def test_invalid_model_cannot_reuse_an_existing_prompt_response(offline_gateway):
    client, calls = offline_gateway
    assert client.post('/v1/chat/completions', json=BASE).status_code == 200
    assert client.post('/v1/chat/completions', json={**BASE, 'model': 'unknown-model'}).status_code == 400
    assert len(calls) == 1


def test_exact_only_cache_initialization_does_not_load_embedding_models(monkeypatch):
    from gateway.cache import SemanticCache
    monkeypatch.setattr(main.settings, 'cache_enabled', True)
    monkeypatch.setattr(SemanticCache, '_connect_redis', lambda self: None)

    def unexpected_download(self):
        raise AssertionError('exact request cache must not load an embedding model')

    monkeypatch.setattr(SemanticCache, '_load_embedder', unexpected_download)
    SemanticCache(semantic_enabled=False)


def test_empty_messages_are_rejected_before_calling_a_provider(offline_gateway):
    client, calls = offline_gateway
    response = client.post('/v1/chat/completions', json={**BASE, 'messages': []})
    assert response.status_code == 422
    assert not calls


def test_untrusted_browser_origin_is_not_granted_cross_origin_api_access(offline_gateway):
    client, _ = offline_gateway
    response = client.options('/v1/chat/completions', headers={
        'Origin': 'https://synthetic-untrusted.example',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type',
    })
    assert 'access-control-allow-origin' not in response.headers


def test_upstream_failure_details_are_not_returned_to_clients(offline_gateway, monkeypatch):
    client, _ = offline_gateway

    async def failed_provider(**kwargs):
        raise RuntimeError('synthetic-internal-detail')

    monkeypatch.setattr(main, 'proxy_with_fallback', failed_provider)
    response = client.post('/v1/chat/completions', json=BASE)
    assert response.status_code == 502
    assert 'synthetic-internal-detail' not in response.text
