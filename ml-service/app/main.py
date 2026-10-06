"""
app/main.py

FastAPI entry point for the recommendation ML service (internal; the public API
is the Next.js app in ../api, which proxies to this service).

Routes:
  GET  /health           liveness + whether data/model are loaded
  POST /recommendations  relevance-sorted products for a member email
  GET  /model/info       holdout metrics of the serving model
  POST /model/train      reload data and retrain the XGBoost model
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.db import DatabaseError
from app.recommender import MemberNotFoundError, Recommender
from app.schemas import ModelInfoResponse, RecommendationRequest, RecommendationResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

recommender = Recommender()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load shop data and the model (training it if absent) before serving."""
    settings = get_settings()
    logger.info(
        "Starting recommendation service — LLM provider=%s model=%s",
        settings.llm_provider.value,
        settings.llm_model_name,
    )
    try:
        await run_in_threadpool(recommender.ensure_ready)
    except (DatabaseError, ValueError) as exc:
        # Keep serving /health so the problem is visible; requests return 503
        logger.error("Recommender not ready: %s", exc)
    yield


app = FastAPI(
    title="Recommendation ML Service",
    description="XGBoost candidate scoring + LLM reranking for shoppingwebsite members.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_ready() -> None:
    """Raise 503 until data and model are loaded."""
    if not recommender.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender is not ready (database unreachable or model not trained)",
        )


@app.get("/health", tags=["ops"])
def health_check() -> dict:
    """Return service liveness and readiness."""
    settings = get_settings()
    return {
        "status": "ok",
        "ready": recommender.is_ready,
        "llm_provider": settings.llm_provider.value,
        "llm_model": settings.llm_model_name,
    }


@app.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
def recommendations(body: RecommendationRequest) -> RecommendationResponse:
    """Return relevance-sorted product recommendations for a member email."""
    _require_ready()
    try:
        return recommender.recommend(body.email, body.limit, body.exclude_product_ids)
    except MemberNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Member not found") from exc
    except DatabaseError as exc:
        logger.error("Database error while recommending: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable") from exc


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Return holdout metrics and feature importance of the serving model."""
    _require_ready()
    return ModelInfoResponse(**recommender.metrics_dict())


@app.post("/model/train", response_model=ModelInfoResponse, tags=["model"])
def model_train() -> ModelInfoResponse:
    """Reload data from PostgreSQL and retrain the model."""
    try:
        recommender.train()
    except (DatabaseError, ValueError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return ModelInfoResponse(**recommender.metrics_dict())
