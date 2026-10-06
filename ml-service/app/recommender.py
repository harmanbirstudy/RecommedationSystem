"""
app/recommender.py

Orchestrates a recommendation request:
  1. Look up the member by email.
  2. Score every catalogue product for them with the XGBoost model.
  3. Drop cart items (and, by default, products they already bought).
  4. Send the top CANDIDATE_POOL_SIZE candidates to the LLM for reranking.
  5. Return `limit` products sorted by relevance.

Shop data is cached in memory and reloaded every DATA_REFRESH_SECONDS so new
orders are reflected without retraining. Responses are cached per
(email, limit, excluded ids) for CACHE_TTL_SECONDS because LLM calls are slow.
"""

import logging
import threading
import time
from dataclasses import asdict

import pandas as pd
from xgboost import XGBClassifier

from app.config import Settings, get_settings
from app.db import ShopData, load_shop_data
from app.features import FEATURE_COLUMNS, UserProfile, build_features, build_user_profile
from app.llm import Candidate, merge_rankings, rerank_with_llm
from app.model import TrainingMetrics, load_model, train_model
from app.schemas import RecommendationResponse, RecommendedProduct

logger = logging.getLogger(__name__)


class MemberNotFoundError(Exception):
    """Raised when no user has the requested email."""


class Recommender:
    """Thread-safe holder for shop data, the trained model and the response cache."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create an empty recommender; call ensure_ready() before use."""
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._data: ShopData | None = None
        self._data_loaded_at = 0.0
        self._model: XGBClassifier | None = None
        self._metrics: TrainingMetrics | None = None
        self._cache: dict[tuple, tuple[float, RecommendationResponse]] = {}

    # ------------------------------------------------------------------ setup
    def ensure_ready(self) -> None:
        """Load shop data and a saved model, training one if needed."""
        with self._lock:
            self._refresh_data(force=True)
            loaded = None if self.settings.retrain_on_startup else load_model(
                self.settings.model_dir
            )
            if loaded:
                self._model, self._metrics = loaded
                logger.info("Loaded saved model (trained %s)", self._metrics.trained_at)
            else:
                self.train()

    def train(self) -> TrainingMetrics:
        """Reload data, retrain the model, and clear the response cache."""
        with self._lock:
            self._refresh_data(force=True)
            assert self._data is not None
            self._model, self._metrics = train_model(
                self._data,
                self.settings.model_dir,
                k=self.settings.recommendation_count,
                exclude_purchased=self.settings.exclude_purchased_products,
            )
            self._cache.clear()
            return self._metrics

    @property
    def metrics(self) -> TrainingMetrics | None:
        """Metrics of the model currently serving."""
        return self._metrics

    @property
    def is_ready(self) -> bool:
        """True once data and a model are loaded."""
        return self._data is not None and self._model is not None

    def _refresh_data(self, force: bool = False) -> None:
        """Reload shop data if stale (or forced)."""
        age = time.monotonic() - self._data_loaded_at
        if force or self._data is None or age > self.settings.data_refresh_seconds:
            self._data = load_shop_data()
            self._data_loaded_at = time.monotonic()

    # -------------------------------------------------------------- recommend
    def recommend(
        self, email: str, limit: int | None = None, exclude_product_ids: list[str] | None = None
    ) -> RecommendationResponse:
        """
        Return relevance-sorted recommendations for a member.

        Raises:
            MemberNotFoundError: If the email does not belong to a user.
        """
        settings = self.settings
        email = email.strip().lower()
        count = min(limit or settings.recommendation_count, settings.max_recommendation_count)
        excluded = frozenset(exclude_product_ids or [])

        cache_key = (email, count, excluded)
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < settings.cache_ttl_seconds:
            return cached[1]

        with self._lock:
            self._refresh_data()
            data, model = self._data, self._model
        assert data is not None and model is not None

        match = data.users[data.users["email"] == email]
        if match.empty:
            raise MemberNotFoundError(email)
        userid = str(match.iloc[0]["userid"])

        candidates, profile = self._score_candidates(data, model, userid, excluded)
        pool = candidates[: max(settings.candidate_pool_size, count)]

        ranking = None
        if pool:
            try:
                ranking = rerank_with_llm(profile, pool, min(count, len(pool)))
            except Exception as exc:  # noqa: BLE001 - any provider failure falls back to the model
                logger.error(
                    "LLM rerank failed (%s: %s); falling back to model ranking",
                    type(exc).__name__,
                    exc,
                )

        ranked = merge_rankings(pool, ranking, count)
        response = RecommendationResponse(
            email=email,
            llm_provider=settings.llm_provider.value,
            llm_model=settings.llm_model_name,
            llm_used=any(r.source == "llm" for r in ranked),
            recommendations=[
                RecommendedProduct(
                    rank=i,
                    productid=r.candidate.productid,
                    title=r.candidate.title,
                    category=r.candidate.category,
                    price=r.candidate.price,
                    imageurl=r.candidate.imageurl,
                    relevance_score=r.relevance_score,
                    model_score=round(r.candidate.model_score, 4),
                    reason=r.reason,
                    source=r.source,
                )
                for i, r in enumerate(ranked, start=1)
            ],
        )
        # Only cache LLM-backed answers so a transient LLM outage is retried
        if response.llm_used or not pool:
            self._cache[cache_key] = (time.monotonic(), response)
        return response

    def _score_candidates(
        self, data: ShopData, model: XGBClassifier, userid: str, excluded: frozenset[str]
    ) -> tuple[list[Candidate], UserProfile]:
        """Score every product for the user and return eligible candidates, best first."""
        now = pd.Timestamp.now()
        as_of = pd.Series({userid: now})
        features = build_features(data.products, data.users, data.lines, [userid], as_of)
        features["model_score"] = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]

        profile = build_user_profile(data.products, data.lines, userid)
        drop = set(excluded)
        if self.settings.exclude_purchased_products:
            drop |= profile.purchased_product_ids

        catalogue = data.products.set_index("productid")
        eligible = features[~features["productid"].isin(drop)].sort_values(
            "model_score", ascending=False
        )
        candidates = [
            Candidate(
                productid=row.productid,
                title=str(catalogue.at[row.productid, "title"]),
                category=str(catalogue.at[row.productid, "category_name"]),
                price=float(catalogue.at[row.productid, "price"]),
                imageurl=catalogue.at[row.productid, "imageurl"],
                model_score=float(row.model_score),
                regional_popularity=float(row.regional_popularity),
                similar_customers_score=float(row.similar_customers_score),
            )
            for row in eligible.itertuples(index=False)
        ]
        logger.info(
            "Scored %d candidates for user %s (excluded %d)", len(candidates), userid, len(drop)
        )
        return candidates, profile

    def metrics_dict(self) -> dict | None:
        """Metrics as a plain dict for the API."""
        return asdict(self._metrics) if self._metrics else None
