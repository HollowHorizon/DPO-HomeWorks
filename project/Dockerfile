FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    CONFIG_PATH=configs/config.yaml \
    ARTIFACTS_DIR=artifacts \
    USE_EMBEDDINGS=false

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .

RUN python -m src.data.prepare_dataset && \
    python -m src.data.eda && \
    python -m src.models.train && \
    python -m src.models.evaluate && \
    pytest -q

EXPOSE 8000
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
