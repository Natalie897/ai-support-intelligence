import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.inference import IntentClassifier
from ml.train import build_pipeline


@pytest.fixture(scope="session")
def classifier():
    texts = [
        "my card has not arrived", "where is my new bank card", "still waiting for card delivery",
        "track the card shipping", "how long before card arrives", "card never delivered",
        "my card was stolen", "someone stole my card", "I lost my credit card",
        "report missing debit card", "my wallet and card were stolen", "freeze a stolen card",
        "bank transfer has not arrived", "recipient never got the transfer",
        "money transfer is missing", "where is my wire transfer", "transfer not received",
        "the recipient has not received money",
    ]
    labels = ["card_arrival"] * 6 + ["lost_or_stolen_card"] * 6 + ["transfer_not_received_by_recipient"] * 6
    model = build_pipeline("logistic_regression").fit(texts, labels)
    return IntentClassifier(model)


@pytest.fixture
def client(classifier):
    app = create_app(Settings(model_path="nonexistent.joblib", metadata_path="nonexistent.json"), classifier)
    with TestClient(app) as http:
        yield http
