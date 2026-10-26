from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


def test_predict_returns_ranked_categories(client):
    response = client.post("/predict", json={"text": "My new card still hasn't arrived", "top_k": 2})
    assert response.status_code == 200
    result = response.json()
    assert result["category"] == "card_arrival"
    assert 0 <= result["score"] <= 1
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["score"] >= result["candidates"][1]["score"]
    assert isinstance(result["review_required"], bool)


def test_validation_rejects_invalid_requests(client):
    assert client.post("/predict", json={"text": "  "}).status_code == 422
    assert client.post("/predict", json={"text": "a"}).status_code == 422
    assert client.post("/predict", json={"text": "valid question", "top_k": 99}).status_code == 422


def test_categories_and_health(client):
    assert client.get("/health").json()["model_ready"] is True
    assert "card_arrival" in client.get("/categories").json()


def test_no_model_returns_503(tmp_path):
    app = create_app(Settings(model_path=str(tmp_path / "missing.joblib"), metadata_path=str(tmp_path / "missing.json")))
    with TestClient(app) as http:
        assert http.get("/health").json() == {"status": "model_missing", "model_ready": False}
        assert http.post("/predict", json={"text": "Where is my card?"}).status_code == 503
        assert http.get("/categories").status_code == 503


def test_confidence_threshold_flags_low_score(classifier):
    app = create_app(Settings(model_path="missing.joblib", metadata_path="missing.json", review_threshold=1.0), classifier)
    with TestClient(app) as http:
        result = http.post("/predict", json={"text": "card not here yet"}).json()
        assert result["review_required"] is True
