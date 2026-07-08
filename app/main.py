"""API gateway for the AutoML microservices."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.proxy import check_upstream, proxy_get, proxy_json, proxy_multipart
from app.workflows import full_run, upload_and_extract

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="automl-gateway",
    version="0.1.0",
    description="Frontend-facing gateway. Proxies to data / metafeatures / generation / "
    "analysis services. Also exposes composed workflows.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------- HEALTH -------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "upstream": {
            "data": check_upstream(settings.data_service_url),
            "metafeatures": check_upstream(settings.metafeatures_service_url),
            "generation": check_upstream(settings.generation_service_url),
            "analysis": check_upstream(settings.analysis_service_url),
        },
    }


# ------------------------------- DATASETS -------------------------------

@app.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    task_type: Optional[str] = Form(None),
    seed: int = Form(42),
    test_size: float = Form(0.2),
) -> Response:
    data = await file.read()
    files = {"file": (file.filename or "upload.csv", data, file.content_type or "text/csv")}
    form = {"target_col": target_col, "seed": str(seed), "test_size": str(test_size)}
    if task_type:
        form["task_type"] = task_type
    return proxy_multipart(settings.data_service_url, "/datasets", files, form)


@app.post("/datasets/openml")
async def import_openml(request: Request) -> Response:
    body = await request.json()
    return proxy_json("POST", settings.data_service_url, "/datasets/openml", body)


@app.get("/datasets")
def list_datasets() -> Response:
    return proxy_get(settings.data_service_url, "/datasets")


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int) -> Response:
    return proxy_get(settings.data_service_url, f"/datasets/{dataset_id}")


@app.get("/datasets/{dataset_id}/preview")
def preview(dataset_id: int, n_rows: int = Query(10), split: str = Query("train")) -> Response:
    return proxy_get(
        settings.data_service_url, f"/datasets/{dataset_id}/preview",
        params={"n_rows": n_rows, "split": split},
    )


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, wipe_files: bool = Query(False)) -> Response:
    return proxy_json(
        "DELETE", settings.data_service_url, f"/datasets/{dataset_id}",
        params={"wipe_files": str(wipe_files).lower()},
    )


@app.get("/datasets/openml/catalog")
def openml_catalog() -> Response:
    return proxy_get(settings.data_service_url, "/datasets/openml/catalog")


# ------------------------------- META-FEATURES -------------------------------

@app.post("/meta-features/{dataset_id}")
def compute_metafeatures(dataset_id: int, force: bool = Query(False)) -> Response:
    return proxy_json(
        "POST", settings.metafeatures_service_url, f"/meta-features/{dataset_id}",
        params={"force": str(force).lower()},
    )


@app.get("/meta-features/{dataset_id}")
def get_metafeatures(dataset_id: int) -> Response:
    return proxy_get(settings.metafeatures_service_url, f"/meta-features/{dataset_id}")


@app.delete("/meta-features/{dataset_id}")
def delete_metafeatures(dataset_id: int) -> Response:
    return proxy_json(
        "DELETE", settings.metafeatures_service_url, f"/meta-features/{dataset_id}"
    )


# ------------------------------- GENERATION -------------------------------

@app.post("/runs")
async def create_run(request: Request) -> Response:
    return proxy_json("POST", settings.generation_service_url, "/runs", await request.json())


@app.get("/runs")
def list_runs(
    dataset_id: Optional[int] = Query(None),
    condition: Optional[str] = Query(None),
    llm_backend: Optional[str] = Query(None),
    limit: int = Query(100),
) -> Response:
    params: dict[str, Any] = {"limit": limit}
    if dataset_id is not None:
        params["dataset_id"] = dataset_id
    if condition:
        params["condition"] = condition
    if llm_backend:
        params["llm_backend"] = llm_backend
    return proxy_get(settings.generation_service_url, "/runs", params=params)


@app.get("/runs/{run_id}")
def get_run(run_id: int) -> Response:
    return proxy_get(settings.generation_service_url, f"/runs/{run_id}")


@app.post("/sweeps")
async def create_sweep(request: Request) -> Response:
    return proxy_json("POST", settings.generation_service_url, "/sweeps", await request.json())


@app.get("/sweeps")
def list_sweeps(limit: int = Query(50)) -> Response:
    return proxy_get(settings.generation_service_url, "/sweeps", params={"limit": limit})


@app.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: int) -> Response:
    return proxy_get(settings.generation_service_url, f"/sweeps/{sweep_id}")


# ------------------------------- ANALYSIS -------------------------------

@app.get("/analysis/summary")
def analysis_summary() -> Response:
    return proxy_get(settings.analysis_service_url, "/analysis/summary")


@app.get("/analysis/errors")
def analysis_errors() -> Response:
    return proxy_get(settings.analysis_service_url, "/analysis/errors")


@app.get("/analysis/iterations")
def analysis_iterations() -> Response:
    return proxy_get(settings.analysis_service_url, "/analysis/iterations")


@app.get("/analysis/models")
def analysis_models() -> Response:
    return proxy_get(settings.analysis_service_url, "/analysis/models")


@app.get("/analysis/size-stratified")
def analysis_size() -> Response:
    return proxy_get(settings.analysis_service_url, "/analysis/size-stratified")


@app.get("/analysis/wilcoxon")
def analysis_wilcoxon(a: str, b: str) -> Response:
    return proxy_get(settings.analysis_service_url, "/analysis/wilcoxon", params={"a": a, "b": b})


# ------------------------------- WORKFLOWS -------------------------------

@app.post("/workflows/upload-and-extract")
async def workflow_upload_and_extract(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    task_type: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Compose upload + meta-feature extraction in one call."""
    data = await file.read()
    return upload_and_extract(
        data, file.filename or "upload.csv", target_col, task_type=task_type,
    )


@app.post("/workflows/full-run")
async def workflow_full_run(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    condition: str = Form(...),
    llm_backend: str = Form(...),
    seed: int = Form(42),
    max_iter: int = Form(3),
    timeout_seconds: int = Form(300),
    task_type: Optional[str] = Form(None),
) -> dict[str, Any]:
    """End-to-end: upload CSV -> meta-features -> enqueue a single-cell generation."""
    data = await file.read()
    return full_run(
        data, file.filename or "upload.csv", target_col,
        condition=condition, llm_backend=llm_backend, seed=seed,
        max_iter=max_iter, timeout_seconds=timeout_seconds, task_type=task_type,
    )
