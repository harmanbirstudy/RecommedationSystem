"""
app/llm.py

LLM reranking of XGBoost candidates.

The provider is fixed at startup by LLM_PROVIDER (anthropic | ollama) and is the
only one used. The LLM receives an anonymised customer profile (no email or name)
and the candidate products with their model scores, and returns the products it
judges most relevant, each with a 0-100 relevance score and a short reason.

LLM output is never trusted blindly (see merge_rankings): unknown product ids are
dropped, duplicates removed, results sorted by relevance in code, and any
shortfall is back-filled from the XGBoost ranking.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import LlmProvider, Settings, get_settings
from app.features import UserProfile

logger = logging.getLogger(__name__)


class LlmPick(BaseModel):
    """One product chosen by the LLM."""

    product_id: str = Field(description="The id of a product from the candidate list")
    relevance_score: int = Field(
        ge=0, le=100, description="How relevant the product is to this customer, 0-100"
    )
    reason: str = Field(
        description="One short, customer-facing sentence explaining the recommendation"
    )


class LlmRanking(BaseModel):
    """Structured LLM response."""

    recommendations: list[LlmPick] = Field(
        description="Chosen products, most relevant first"
    )


@dataclass(frozen=True)
class Candidate:
    """A product proposed by the XGBoost model."""

    productid: str
    title: str
    category: str
    price: float
    imageurl: str | None
    model_score: float
    regional_popularity: float
    similar_customers_score: float


@dataclass(frozen=True)
class RankedProduct:
    """Final recommendation after LLM reranking (or model fallback)."""

    candidate: Candidate
    relevance_score: int | None
    reason: str
    source: str  # "llm" | "model"


SYSTEM_PROMPT = """You are the recommendation engine for an online store's checkout page.
A machine-learning model has already shortlisted candidate products for a customer,
based on their past orders, what customers in their region buy, and what similar
customers buy. Your job is to choose the {count} most relevant candidates for this
customer and score their relevance.

Rules:
- Only use product_id values that appear in the candidate list. Never invent products.
- Choose exactly {count} products (or every candidate if there are fewer).
- relevance_score is 0-100; higher means more likely to interest this customer now.
  Weigh the model score heavily, then complementarity with recent purchases and the
  customer's favourite categories. Avoid near-duplicates of things they just bought.
- reason is one short, friendly sentence addressed to the customer (max 20 words),
  explaining why the product suits them (e.g. how it complements a past purchase or
  a favourite category). It is shown on the checkout page, so never mention scores,
  rankings, models or algorithms (no "top-scored pick", "highly ranked", etc.),
  and never mention other customers' personal data.
- Return the list ordered from most to least relevant."""


def _format_prompt(profile: UserProfile, candidates: list[Candidate], count: int) -> str:
    """Render the customer profile and candidates as the user message."""
    customer = {
        "region": profile.region or "unknown",
        "number_of_past_orders": profile.n_orders,
        "favourite_categories": profile.category_affinity,
        "recent_purchases": profile.recent_titles,
    }
    shortlist = [
        {
            "product_id": c.productid,
            "title": c.title,
            "category": c.category,
            "price": round(c.price, 2),
            "model_score": round(c.model_score, 3),
            "popularity_in_region": round(c.regional_popularity, 3),
            "bought_by_similar_customers": round(c.similar_customers_score, 3),
        }
        for c in candidates
    ]
    return (
        f"Customer profile:\n{json.dumps(customer, indent=2)}\n\n"
        f"Candidate products (ordered by model score):\n{json.dumps(shortlist, indent=2)}\n\n"
        f"Choose the {count} most relevant products."
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Create the chat model for the configured provider (one per process)."""
    settings: Settings = get_settings()
    if settings.llm_provider is LlmProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        assert settings.anthropic_api_key is not None  # enforced by Settings validator
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0,
            max_tokens=2048,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )


def rerank_with_llm(
    profile: UserProfile, candidates: list[Candidate], count: int
) -> LlmRanking:
    """
    Ask the configured LLM to pick and score the most relevant candidates.

    Raises:
        Exception: Whatever the provider client raises (network, timeout, parsing);
                   the caller falls back to the model ranking.
    """
    settings = get_settings()
    llm = get_chat_model()
    if settings.llm_provider is LlmProvider.OLLAMA:
        # Ollama's native JSON-schema structured outputs
        structured = llm.with_structured_output(LlmRanking, method="json_schema")
    else:
        structured = llm.with_structured_output(LlmRanking)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(count=count)),
        HumanMessage(content=_format_prompt(profile, candidates, count)),
    ]
    result = structured.invoke(messages)
    if not isinstance(result, LlmRanking):
        # some providers return a dict when the schema is passed through
        result = LlmRanking.model_validate(result)
    return result


def merge_rankings(
    candidates: list[Candidate], ranking: LlmRanking | None, count: int
) -> list[RankedProduct]:
    """
    Turn the LLM response into the final, relevance-sorted list.

    - Picks with ids outside the candidate list are dropped (hallucinations).
    - Duplicate ids keep their first occurrence.
    - LLM picks are sorted by relevance_score desc, ties broken by model score.
    - If the LLM returned fewer than `count` valid picks (or failed: ranking=None),
      the remainder is back-filled from the XGBoost order, after the LLM picks.
    """
    by_id = {c.productid: c for c in candidates}
    chosen: list[RankedProduct] = []
    seen: set[str] = set()

    for pick in ranking.recommendations if ranking else []:
        candidate = by_id.get(pick.product_id)
        if candidate is None:
            logger.warning("LLM returned unknown product id %r; ignoring", pick.product_id)
            continue
        if pick.product_id in seen:
            continue
        seen.add(pick.product_id)
        chosen.append(
            RankedProduct(
                candidate=candidate,
                relevance_score=pick.relevance_score,
                reason=pick.reason.strip(),
                source="llm",
            )
        )

    chosen.sort(key=lambda r: (r.relevance_score or 0, r.candidate.model_score), reverse=True)
    chosen = chosen[:count]

    for candidate in sorted(candidates, key=lambda c: c.model_score, reverse=True):
        if len(chosen) >= count:
            break
        if candidate.productid in seen:
            continue
        seen.add(candidate.productid)
        chosen.append(
            RankedProduct(
                candidate=candidate,
                relevance_score=None,
                reason="Popular with customers who shop like you.",
                source="model",
            )
        )
    return chosen
