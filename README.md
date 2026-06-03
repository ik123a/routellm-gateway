# RouteLLM-Gateway

> A cost-aware LLM API gateway that dynamically routes queries to the most cost-effective model while preserving accuracy.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Problem

Using GPT-4o for every query costs ~$10/1M output tokens. But 40-60% of queries are simple enough for a local Llama-3-8B or GPT-4o-mini. **RouteLLM-Gateway** solves this by dynamically routing each query to the cheapest model capable of answering it correctly.

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────┐
│  FastAPI Gateway         │
│  /v1/chat/completions    │
│                          │
│  ┌───────────────────┐   │
│  │ Redis Cache       │◄──┤── Exact Hash + Semantic Embedding
│  │ (2-Layer)         │   │
│  └───────────────────┘   │
│           │ Miss         │
│           ▼              │
│  ┌───────────────────┐   │
│  │ Router Classifier  │  │   DistilBERT + MLP Head
│  │ (ONNX Runtime)    │   │   Cost-Penalized CE Loss
│  └───────────────────┘   │
│           │              │
│     ┌─────┼─────┐        │
│     ▼     ▼     ▼        │
│   Tier0 Tier1 Tier2      │   Llama-3-8B / GPT-4o-mini / GPT-4o
│   $0.00 $0.15  $2.50     │   per 1K tokens
│     │     │     │        │
│     └─────┼─────┘        │
│           ▼              │
│  ┌───────────────────┐   │
│  │ Fallback Handler  │   │   429/5xx → auto-escalate tier
│  └───────────────────┘   │
└─────────────────────────┘
    │
    ▼
Streaming SSE Response
(with X-RouteLLM-* headers)
```

## Key Features

- **OpenAI-compatible API** – Drop-in replacement for `/v1/chat/completions`
- **Cost-aware routing** – Custom loss function penalizes expensive routes
- **Dual-layer caching** – Exact hash + semantic similarity (cosine ≥ 0.96)
- **Automatic fallback** – Escalates tier on 429/5xx errors
- **Full observability** – Routing metadata in response headers and `/stats` endpoint
- **ONNX inference** – Router runs in <15ms on CPU

## Quick Start

### 1. Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/routellm-gateway.git
cd routellm-gateway

# Create environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure
copy .env.example .env
# Edit .env with your API keys
```

### 2. Run the Gateway (Heuristic Mode)

```bash
# Start without a trained model (uses rule-based routing)
python -m gateway.main
```

### 3. Test It

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'
```

### 4. Train the Router

```bash
# Train on synthetic data (bootstrap)
python -m router.train --epochs 20 --wandb

# Train on custom data
python -m router.train --data data/training_queries.jsonl --epochs 30
```

### 5. Run with Docker

```bash
docker-compose up -d
```

## Project Structure

```
routellm-gateway/
├── gateway/
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py             # Model tiers, costs, settings
│   ├── router_client.py      # Router inference (ONNX + heuristic)
│   ├── proxy.py              # Streaming proxy to LLM providers
│   └── cache.py              # Redis semantic caching
├── router/
│   ├── model.py              # PyTorch DistilBERT classifier
│   ├── dataset.py            # Dataset loaders & synthetic data
│   └── train.py              # Training loop & ONNX export
├── eval/
│   ├── benchmark.py          # Cost/accuracy evaluation harness
│   └── judge.py              # LLM-as-Judge correctness scorer
├── tests/
│   ├── test_gateway.py       # API integration tests
│   └── test_router.py        # Router & model unit tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Evaluation Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Cost Reduction | >60% | Compare total cost vs GPT-4o-only baseline |
| Accuracy Retention | >98% | LLM-as-Judge agreement with GPT-4o |
| Router Overhead | <15ms | ONNX inference profiling |
| Cache Hit Rate | >20% | Semantic similarity threshold tuning |

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT

## Author

Ishaan Kumar
