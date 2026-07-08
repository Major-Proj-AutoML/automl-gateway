FROM python:3.11-slim

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir \
        fastapi "uvicorn[standard]" pydantic-settings httpx python-multipart

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
