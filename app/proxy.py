"""Generic HTTP proxy helpers. Forwards requests to backend services.

The gateway never owns business logic — it either forwards a request as-is or
composes multiple backend calls into one workflow.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, Request, Response

from app.config import settings


def _client(timeout: float | None = None) -> httpx.Client:
    return httpx.Client(timeout=timeout or settings.http_timeout_seconds)


def _forward_response(r: httpx.Response) -> Response:
    """Preserve the backend's status code and body; scrub hop-by-hop headers."""
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers = {k: v for k, v in r.headers.items() if k.lower() not in excluded}
    return Response(content=r.content, status_code=r.status_code, headers=headers)


def proxy_get(base_url: str, path: str, params: dict | None = None) -> Response:
    with _client() as c:
        try:
            r = c.get(f"{base_url.rstrip('/')}{path}", params=params)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    return _forward_response(r)


def proxy_json(
    method: str, base_url: str, path: str, json_body: Any | None = None,
    params: dict | None = None,
) -> Response:
    with _client() as c:
        try:
            r = c.request(
                method, f"{base_url.rstrip('/')}{path}", json=json_body, params=params
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    return _forward_response(r)


def proxy_multipart(
    base_url: str, path: str, files: dict, data: dict | None = None,
) -> Response:
    with _client() as c:
        try:
            r = c.post(f"{base_url.rstrip('/')}{path}", files=files, data=data or {})
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    return _forward_response(r)


def check_upstream(base_url: str) -> bool:
    try:
        with _client(timeout=5.0) as c:
            r = c.get(f"{base_url.rstrip('/')}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False
