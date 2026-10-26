"""Model loading and probability ranking."""
from dataclasses import dataclass
from pathlib import Path
import joblib


@dataclass(frozen=True)
class IntentPrediction:
    category: str
    score: float


class IntentClassifier:
    def __init__(self, model):
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            raise TypeError("A fitted classifier supporting predict_proba is required")
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        # Load artifacts only from trusted sources: pickle/joblib can execute code.
        return cls(joblib.load(path))

    def predict(self, text: str, top_k: int = 3) -> list[IntentPrediction]:
        scores = self.model.predict_proba([text])[0]
        names = self.model.classes_
        order = scores.argsort()[::-1][:top_k]
        return [IntentPrediction(str(names[i]), float(scores[i])) for i in order]

    def categories(self) -> list[str]:
        return sorted(str(c) for c in self.model.classes_)
