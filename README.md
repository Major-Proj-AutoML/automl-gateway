# automl-gateway

API gateway for the AutoML microservices. Port **8000**. Single entry point for the frontend.

## Responsibilities

- **Proxy** frontend requests to the correct backend service (`data`, `metafeatures`, `generation`, `analysis`).
- **CORS**: allows the frontend origin (`http://localhost:3000` and `http://localhost:5173` by default).
- **Composed workflows** — the gateway itself calls multiple services in sequence:
  - `POST /workflows/upload-and-extract` — upload CSV → extract meta-features → return both.
  - `POST /workflows/full-run` — upload → meta-features → enqueue a generation run.

## Endpoints

Everything the four backend services expose is available under the same path here. Full list at `http://localhost:8000/docs`.

Highlights:

| Purpose | Path |
|---|---|
| Upload CSV | `POST /datasets` |
| Import OpenML | `POST /datasets/openml` |
| List datasets | `GET /datasets` |
| Extract meta-features | `POST /meta-features/{id}` |
| Enqueue single run | `POST /runs` |
| Enqueue sweep | `POST /sweeps` |
| RQ1–RQ5 analyses | `GET /analysis/*` |
| Full pipeline in one call | `POST /workflows/full-run` |

## Local run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Requires the four backend services to be running (either standalone or via docker-compose). If any are down, the gateway will return 502 on the affected routes; `GET /health` shows per-upstream status.

## Tests

Uses httpx MockTransport to intercept outbound calls — no backend services required:

```bash
pytest -v
```
