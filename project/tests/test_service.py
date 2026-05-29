from fastapi.testclient import TestClient

from src.service.app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json={"query": "Как сделать Dockerfile?", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert "answerable" in data
        assert "chunks" in data
