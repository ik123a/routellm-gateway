"""
Request Middleware
===================
Adds per-request tracing (unique request IDs), structured JSON
logging, and latency measurement to every gateway request.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("routellm.middleware")

# Context variable to propagate request_id across async boundaries
request_id_var: ContextVar[str] = ContextVar("request_id", default="none")


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique X-Request-ID header into every request/response.
    If the client already sends one, we reuse it. Otherwise we generate
    a new UUID4.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Check if client provided a request ID
        req_id = request.headers.get("X-Request-ID")
        if not req_id:
            req_id = str(uuid.uuid4())[:12]  # Short UUID for readability

        # Store in context for downstream access
        token = request_id_var.set(req_id)

        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)

        # Inject into response headers
        response.headers["X-Request-ID"] = req_id
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request/response as structured JSON with:
      - request_id, method, path, status_code, latency_ms, client_ip
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        req_id = request.headers.get("X-Request-ID", get_request_id())
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Log request entry
        logger.info(json.dumps({
            "event": "request_start",
            "request_id": req_id,
            "method": method,
            "path": path,
            "client_ip": client_ip,
        }))

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(json.dumps({
                "event": "request_error",
                "request_id": req_id,
                "method": method,
                "path": path,
                "error": str(e),
                "latency_ms": round(elapsed, 2),
            }))
            raise

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(json.dumps({
            "event": "request_complete",
            "request_id": req_id,
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "latency_ms": round(elapsed, 2),
        }))

        return response
