"""Gateway tests. Uses httpx MockTransport to intercept outbound calls so we
never actually hit the backend services."""

from __future__ import annotations

import httpx
import pytest

from app import proxy, workflows


class FakeUpstream:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        url = str(request.url)
        if url.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and "/datasets" in url and not url.endswith("/datasets"):
            # /datasets/{id} etc
            return httpx.Response(200, json={"id": 1, "name": "ds"})
        if request.method == "GET" and url.endswith("/datasets"):
            return httpx.Response(200, json=[{"id": 1, "name": "ds"}])
        if request.method == "POST" and url.endswith("/datasets"):
            return httpx.Response(201, json={
                "id": 1, "name": "uploaded", "source": "custom",
                "target_col": "y", "task_type": "binary_classification",
                "train_path": "/x", "test_path": "/y", "created_at": "2024-01-01T00:00:00Z",
            })
        if request.method == "POST" and "/meta-features/" in url:
            return httpx.Response(201, json={
                "dataset_id": 1, "features": {"foo": 1}, "computed_at": "2024-01-01T00:00:00Z",
                "cached": False,
            })
        if request.method == "POST" and url.endswith("/runs"):
            return httpx.Response(202, json={
                "rq_job_id": "abc", "status_url": "/runs/by-rq-job/abc", "run_id": None,
            })
        if request.method == "GET" and "/analysis/summary" in url:
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"detail": "not found in stub"})


@pytest.fixture
def fake_upstream(monkeypatch):
    fake = FakeUpstream()
    transport = httpx.MockTransport(fake.handler)

    def _client(timeout=None):
        return httpx.Client(transport=transport, timeout=timeout or 10.0)

    monkeypatch.setattr(proxy, "_client", _client)
    # Also patch the workflows module's httpx.Client
    original_client = httpx.Client

    def _patched_client(*args, **kwargs):
        return original_client(transport=transport, timeout=kwargs.get("timeout", 10.0))

    monkeypatch.setattr(workflows.httpx, "Client", _patched_client)
    return fake


def test_health_reports_all_upstreams(client, fake_upstream):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["upstream"]["data"] is True
    assert body["upstream"]["metafeatures"] is True
    assert body["upstream"]["generation"] is True
    assert body["upstream"]["analysis"] is True


def test_proxies_datasets_list(client, fake_upstream):
    r = client.get("/datasets")
    assert r.status_code == 200
    assert r.json() == [{"id": 1, "name": "ds"}]


def test_proxies_create_run(client, fake_upstream):
    r = client.post("/runs", json={
        "dataset_id": 1, "condition": "b2_metafeature",
        "llm_backend": "m1", "seed": 42, "max_iter": 3, "timeout_seconds": 60,
    })
    assert r.status_code == 202, r.text
    assert r.json()["rq_job_id"] == "abc"


def test_proxies_analysis(client, fake_upstream):
    r = client.get("/analysis/summary")
    assert r.status_code == 200
    assert r.json() == []


def test_workflow_upload_and_extract(client, fake_upstream):
    csv = b"a,b,y\n1,2,0\n3,4,1\n"
    files = {"file": ("data.csv", csv, "text/csv")}
    data = {"target_col": "y"}
    r = client.post("/workflows/upload-and-extract", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["id"] == 1
    assert body["meta_features"]["dataset_id"] == 1


def test_workflow_full_run(client, fake_upstream):
    csv = b"a,b,y\n1,2,0\n3,4,1\n"
    files = {"file": ("data.csv", csv, "text/csv")}
    data = {
        "target_col": "y", "condition": "b2_metafeature",
        "llm_backend": "m1",
    }
    r = client.post("/workflows/full-run", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert body["run"]["rq_job_id"] == "abc"
