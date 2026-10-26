"""Optional audit of aggregate prediction metadata (never stores raw text)."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    review_required: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


def create_session_factory(url: str):
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def save_prediction(session_factory, category: str, score: float, review_required: bool) -> None:
    with session_factory() as session:
        session.add(PredictionRecord(
            category=category, score=score, review_required=review_required
        ))
        session.commit()
