"""FastAPI inference API (separates training from serving)."""
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.config import Settings, from_env
from app.inference import IntentClassifier
from app.schemas import Candidate, PredictRequest, PredictResponse, StatusResponse
from app.storage import create_session_factory, save_prediction

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, classifier: IntentClassifier | None = None) -> FastAPI:
    settings = settings or from_env()

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        api.state.classifier = classifier
        api.state.model_version = "unknown"
        api.state.session_factory = None
        api.state.db_engine = None
        if api.state.classifier is None:
            artifact = Path(settings.model_path)
            if artifact.is_file():
                api.state.classifier = IntentClassifier.load(artifact)
            else:
                logger.warning("Model missing at %s; run training before prediction", artifact)
        meta = Path(settings.metadata_path)
        if meta.is_file():
            api.state.model_version = str(json.loads(meta.read_text(encoding="utf-8")).get("model_version", "unknown"))
        if settings.database_url:
            engine, factory = create_session_factory(settings.database_url)
            api.state.db_engine, api.state.session_factory = engine, factory
        yield
        if api.state.db_engine is not None:
            api.state.db_engine.dispose()

    api = FastAPI(
        title="AI Support Intelligence",
        description="Classify English banking queries into intents with a trained TF-IDF model.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @api.get("/health", response_model=StatusResponse, tags=["service"])
    def health(request: Request):
        ready = request.app.state.classifier is not None
        return StatusResponse(status="ok" if ready else "model_missing", model_ready=ready)

    @api.get("/categories", response_model=list[str], tags=["inference"])
    def categories(request: Request):
        active = request.app.state.classifier
        if active is None:
            raise HTTPException(status_code=503, detail="Model not trained; see README")
        return active.categories()

    @api.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(payload: PredictRequest, request: Request):
        active = request.app.state.classifier
        if active is None:
            raise HTTPException(status_code=503, detail="Model not trained; see README")
        ranked = active.predict(payload.text, payload.top_k)
        best = ranked[0]
        review_required = best.score < settings.review_threshold
        if request.app.state.session_factory is not None:
            try:
                save_prediction(request.app.state.session_factory, best.category, best.score, review_required)
            except Exception:
                logger.exception("Failed to write audit record")
                raise HTTPException(status_code=503, detail="Prediction logging unavailable")
        return PredictResponse(
            category=best.category,
            score=best.score,
            review_required=review_required,
            candidates=[Candidate(category=item.category, score=item.score) for item in ranked],
            model_version=request.app.state.model_version,
        )

    return api


app = create_app()
