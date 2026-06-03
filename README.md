<div align="center">
  
# 🚦 RouteLLM-Gateway

**An intelligent, cost-aware AI API Gateway that cuts LLM bills by up to 80% through dynamic model routing and semantic caching.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 What is RouteLLM?

Companies using AI currently face a major dilemma:
* Routing all traffic to **cheap models** (like LLaMA-3-8B) saves money but yields poor answers for complex questions.
* Routing all traffic to **expensive models** (like GPT-4o) yields great answers but wastes massive amounts of money on simple questions (e.g., "Say hello").

**RouteLLM-Gateway** acts as a "traffic cop" sitting between the user and LLM providers. It analyzes incoming prompts in real-time and dynamically routes them to the cheapest model capable of answering correctly, reducing costs while maintaining high accuracy.

It is a **100% drop-in replacement** for the official OpenAI API, meaning you can integrate it into any existing app by just changing the base URL.

---

## ✨ Key Features

- 🧠 **Intelligent Routing Core:** Uses an embedded `DistilBERT` machine learning classifier to instantly categorize prompts into `CHEAP`, `MEDIUM`, or `STRONG` tiers based on complexity (math, coding, casual chat).
- ⚡ **Semantic Caching (Redis):** Caches responses using **Sentence Embeddings**. If a user asks "How do I reverse a string?" and another asks "Code to reverse text?", the gateway detects the semantic similarity and returns the cached answer in <10ms, saving 100% of the API cost.
- 🛡️ **Resilience & Fault Tolerance:** If a lower-tier model crashes or times out *mid-stream*, the proxy catches the error and transparently upgrades the request to a more reliable tier without breaking the user experience.
- 🎨 **Built-in UI:** Includes a stunning, glassmorphism-inspired web interface to chat, test streams, and view live routing statistics/cost-savings.
- 🐳 **1-Click Docker Setup:** The Gateway, ML models, and Redis cache are all containerized for instant deployment.

---

## 🚀 How to Use (1-Click Start)

You don't need to install any complex dependencies. Everything is containerized!

### 1. Start the System
Ensure you have Docker Desktop installed, then double-click the included `start.bat` file, OR run this in your terminal:

```bash
docker-compose up --build -d
```
*This starts the FastAPI Gateway on port 8000 and the Redis cache on port 6379.*

### 2. Open the Beautiful UI
Once the container is running, open your web browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

You can chat with the AI directly from here. The UI will show you exactly which tier was selected, the total latency, and your estimated cost savings!

---

## 💻 API Integration (For Developers)

RouteLLM is fully compatible with the standard OpenAI SDKs. To integrate it into your own scripts or apps, just change the base URL to `http://localhost:8000/v1` and set the model to `"auto"`.

### Python Example:

```python
import requests
import json

payload = {
    "model": "auto",  # Tells the gateway to dynamically route!
    "messages": [{"role": "user", "content": "Write a python script to reverse a linked list."}],
    "stream": False
}

response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
data = response.json()

print(data["choices"][0]["message"]["content"])

# See exactly how much money RouteLLM saved you:
print("Routing Stats:", json.dumps(data["_routellm"], indent=2))
```

### cURL Example (With Streaming):
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain quantum computing."}],
    "stream": true
  }'
```

---

## 🏗️ Project Architecture

1. **Frontend / UI:** Vanilla JS, CSS Glassmorphism interacting via REST.
2. **FastAPI Gateway:** Handles SSE streaming chunking, timeouts, and request parsing.
3. **RouterClient (`router_client.py`):** The brain of the operation. It uses regex heuristics and ONNX-based ML models to calculate prompt complexity probabilities.
4. **SemanticCache (`cache.py`):** Uses `sentence-transformers/all-MiniLM-L6-v2` to vectorize queries and store them in Redis.
5. **Proxy (`proxy.py`):** Maintains HTTPX Async clients to proxy traffic to OpenAI, Anthropic, or local endpoints.

---

## 🧪 Running Benchmarks & Evaluation

Want to prove the cost savings? We built a benchmarking suite that evaluates the cost-vs-accuracy tradeoff of the gateway.

To run the benchmark (requires Python environment setup):
```bash
python -m eval.benchmark
```
*Outputs a detailed report showing total requests, cost saved (%), tier distribution, and p95 latency metrics.*

---

## 📝 License
This project is open-source and available under the MIT License.
