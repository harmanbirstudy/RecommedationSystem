"""
app/schemas.py

Pydantic request/response models for the ML service API.
"""

from pydantic import BaseModel, EmailStr, Field


class RecommendationRequest(BaseModel):
    """Body for POST /recommendations."""

    email: EmailStr
    limit: int | None = Field(
        default=None, ge=1, description="Number of products; defaults to RECOMMENDATION_COUNT"
    )
    exclude_product_ids: list[str] = Field(
        default_factory=list, description="Products to leave out, e.g. those already in the cart"
    )


class RecommendedProduct(BaseModel):
    """One recommended product, in relevance order."""

    rank: int
    productid: str
    title: str
    category: str
    price: float
    imageurl: str | None
    relevance_score: int | None = Field(
        description="LLM relevance 0-100; null when back-filled from the model ranking"
    )
    model_score: float = Field(description="XGBoost purchase probability")
    reason: str
    source: str = Field(description="'llm' or 'model' (fallback/back-fill)")


class RecommendationResponse(BaseModel):
    """Response for POST /recommendations."""

    email: str
    llm_provider: str
    llm_model: str
    llm_used: bool
    recommendations: list[RecommendedProduct]


class ModelInfoResponse(BaseModel):
    """Response for GET /model/info and POST /model/train."""

    trained_at: str
    n_training_rows: int
    n_customers: int
    positive_rate: float
    holdout_auc: float
    holdout_hit_rate_at_k: float
    baseline_popularity_hit_rate_at_k: float
    k: int
    feature_importance: dict[str, float]
