"""Environment-based configuration with safe defaults."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_path: str = "artifacts/model.joblib"
    metadata_path: str = "artifacts/metrics.json"
    database_url: str | None = None
    review_threshold: float = 0.40


def from_env() -> Settings:
    threshold = float(os.getenv("REVIEW_THRESHOLD", "0.40"))
    if not 0 <= threshold <= 1:
        raise ValueError("REVIEW_THRESHOLD must be between 0 and 1")
    return Settings(
        model_path=os.getenv("MODEL_PATH", "artifacts/model.joblib"),
        metadata_path=os.getenv("MODEL_METADATA_PATH", "artifacts/metrics.json"),
        database_url=os.getenv("DATABASE_URL") or None,
        review_threshold=threshold,
    )
