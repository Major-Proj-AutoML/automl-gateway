"""Composed multi-service workflows.

The gateway calls several backend services in sequence to answer a single
frontend request. Each step returns quickly if it hits a cache.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException

from app.config import settings


def _post(url: str, files=None, data=None, json_body=None, timeout: float = 60.0):
    with httpx.Client(timeout=timeout) as c:
        try:
            return c.post(url, files=files, data=data, json=json_body)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}")


def upload_and_extract(
    csv_bytes: bytes, filename: str, target_col: str, task_type: str | None = None,
) -> dict[str, Any]:
    """Upload CSV -> extract meta-features. Returns both dataset info and features."""
    files = {"file": (filename, csv_bytes, "text/csv")}
    data = {"target_col": target_col}
    if task_type:
        data["task_type"] = task_type
    r = _post(f"{settings.data_service_url}/datasets", files=files, data=data)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    dataset = r.json()

    r = _post(
        f"{settings.metafeatures_service_url}/meta-features/{dataset['id']}",
        timeout=120.0,
    )
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    features = r.json()

    return {"dataset": dataset, "meta_features": features}


def full_run(
    csv_bytes: bytes, filename: str, target_col: str,
    condition: str, llm_backend: str, seed: int = 42,
    max_iter: int = 3, timeout_seconds: int = 300,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Upload CSV -> extract meta-features -> enqueue a single-cell generation run."""
    result = upload_and_extract(csv_bytes, filename, target_col, task_type=task_type)
    dataset_id = result["dataset"]["id"]

    payload = {
        "dataset_id": dataset_id,
        "condition": condition,
        "llm_backend": llm_backend,
        "seed": seed,
        "max_iter": max_iter,
        "timeout_seconds": timeout_seconds,
    }
    r = _post(f"{settings.generation_service_url}/runs", json_body=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    enqueue = r.json()

    return {
        "dataset": result["dataset"],
        "meta_features": result["meta_features"],
        "run": enqueue,
    }
