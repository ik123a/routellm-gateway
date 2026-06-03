"""
Redis Semantic Cache
=====================
Caches routing decisions and LLM responses using both exact-match
hashing and semantic similarity (sentence embeddings).

When a new query arrives:
  1. Check exact hash -> instant hit (< 1ms).
  2. If miss, compute embedding and check cosine similarity against
     recent cached embeddings -> semantic hit (< 10ms).
  3. If miss, route normally and store result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

import numpy as np

from gateway.config import settings

logger = logging.getLogger("routellm.cache")


class SemanticCache:
    """
    Two-layer cache backed by Redis.

    Layer 1 (Exact): SHA256 hash of the prompt -> cached response.
    Layer 2 (Semantic): Sentence embedding -> nearest neighbor search
                        using cosine similarity.
    """

    def __init__(self):
        self._redis = None
        self._embedder = None
        self._enabled = settings.cache_enabled
        self._similarity_threshold = settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds

        if self._enabled:
            self._connect_redis()
            self._load_embedder()

    def _connect_redis(self):
        """Attempt to connect to Redis. Disable cache on failure."""
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=False,  # We store binary embeddings
            )
            self._redis.ping()
            logger.info(
                f"Redis connected at {settings.redis_host}:{settings.redis_port}"
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Cache disabled.")
            self._redis = None
            self._enabled = False

    def _load_embedder(self):
        """Load the sentence-transformer model for semantic embeddings."""
        if not self._enabled:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Semantic embedder loaded (all-MiniLM-L6-v2).")
        except Exception as e:
            logger.warning(f"Failed to load embedder: {e}. Semantic cache disabled.")
            self._embedder = None

    @property
    def is_available(self) -> bool:
        return self._enabled and self._redis is not None

    # ------------------------------------------------------------------
    # Exact-match cache
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_key(text: str) -> str:
        return f"route:exact:{hashlib.sha256(text.encode()).hexdigest()}"

    def get_exact(self, prompt: str) -> Optional[dict]:
        """Look up an exact match in Redis."""
        if not self.is_available:
            return None
        try:
            key = self._hash_key(prompt)
            raw = self._redis.get(key)
            if raw:
                logger.debug(f"Exact cache hit for key={key[:24]}...")
                return json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.warning(f"Exact cache lookup error: {e}")
        return None

    def set_exact(self, prompt: str, data: dict):
        """Store an exact-match cache entry."""
        if not self.is_available:
            return
        try:
            key = self._hash_key(prompt)
            self._redis.setex(key, self._ttl, json.dumps(data).encode("utf-8"))
        except Exception as e:
            logger.warning(f"Exact cache write error: {e}")

    # ------------------------------------------------------------------
    # Semantic similarity cache
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Compute a 384-dim sentence embedding."""
        if self._embedder is None:
            return None
        return self._embedder.encode(text, normalize_embeddings=True)

    def get_semantic(self, prompt: str) -> Optional[dict]:
        """
        Search cached embeddings for a semantically similar query.
        Returns cached data if cosine similarity >= threshold.
        """
        if not self.is_available or self._embedder is None:
            return None

        query_vec = self._embed(prompt)
        if query_vec is None:
            return None

        try:
            # Scan all semantic keys (in production, use a vector DB)
            keys = self._redis.keys(b"route:semantic:*")
            best_score = -1.0
            best_data = None

            for key in keys[:500]:  # Cap scan for performance
                raw = self._redis.get(key)
                if not raw:
                    continue
                entry = json.loads(raw.decode("utf-8"))
                cached_vec = np.array(entry.get("embedding", []), dtype=np.float32)
                if cached_vec.shape[0] != query_vec.shape[0]:
                    continue

                # Cosine similarity (vectors are already normalized)
                score = float(np.dot(query_vec, cached_vec))
                if score > best_score:
                    best_score = score
                    best_data = entry

            if best_score >= self._similarity_threshold and best_data:
                logger.info(
                    f"Semantic cache hit (similarity={best_score:.4f})"
                )
                return best_data.get("response_data")

        except Exception as e:
            logger.warning(f"Semantic cache lookup error: {e}")
        return None

    def set_semantic(self, prompt: str, data: dict):
        """Store a semantic cache entry with its embedding vector."""
        if not self.is_available or self._embedder is None:
            return

        vec = self._embed(prompt)
        if vec is None:
            return

        try:
            key_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            key = f"route:semantic:{key_hash}"
            entry = {
                "embedding": vec.tolist(),
                "response_data": data,
                "prompt_preview": prompt[:100],
            }
            self._redis.setex(
                key.encode(), self._ttl, json.dumps(entry).encode("utf-8")
            )
        except Exception as e:
            logger.warning(f"Semantic cache write error: {e}")

    # ------------------------------------------------------------------
    # Unified lookup
    # ------------------------------------------------------------------

    def get(self, prompt: str) -> Optional[dict]:
        """
        Try exact match first, then semantic match.
        Returns cached data dict or None.
        """
        start = time.perf_counter()

        # Layer 1: exact
        result = self.get_exact(prompt)
        if result:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Cache hit (exact) in {elapsed:.2f}ms")
            return result

        # Layer 2: semantic
        result = self.get_semantic(prompt)
        if result:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Cache hit (semantic) in {elapsed:.2f}ms")
            return result

        return None

    def store(self, prompt: str, data: dict):
        """Store in both exact and semantic caches."""
        self.set_exact(prompt, data)
        self.set_semantic(prompt, data)
