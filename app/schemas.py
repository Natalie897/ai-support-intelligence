"""Public API schemas."""
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000, description="English-language bank support message")
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value.strip()


class Candidate(BaseModel):
    category: str
    score: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    category: str
    score: float
    review_required: bool
    candidates: list[Candidate]
    model_version: str


class StatusResponse(BaseModel):
    status: str
    model_ready: bool
