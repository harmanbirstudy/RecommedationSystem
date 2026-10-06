"""
tests/test_llm_merge.py

merge_rankings: validation, relevance sorting and back-fill of LLM output.
"""

from app.llm import Candidate, LlmPick, LlmRanking, merge_rankings


def _candidate(pid: str, score: float) -> Candidate:
    """Candidate with only the fields merge_rankings cares about."""
    return Candidate(
        productid=pid,
        title=f"Product {pid}",
        category="Books",
        price=10.0,
        imageurl=None,
        model_score=score,
        regional_popularity=0.1,
        similar_customers_score=0.1,
    )


CANDIDATES = [_candidate("a", 0.9), _candidate("b", 0.8), _candidate("c", 0.7), _candidate("d", 0.6)]


def _ranking(*picks: tuple[str, int]) -> LlmRanking:
    """Build an LLM response from (product_id, relevance) pairs."""
    return LlmRanking(
        recommendations=[LlmPick(product_id=p, relevance_score=s, reason="r") for p, s in picks]
    )


def test_sorted_by_llm_relevance_not_llm_order():
    """Output is sorted by relevance_score even if the LLM lists them out of order."""
    result = merge_rankings(CANDIDATES, _ranking(("c", 70), ("d", 95), ("a", 80)), count=3)
    assert [r.candidate.productid for r in result] == ["d", "a", "c"]
    assert all(r.source == "llm" for r in result)


def test_unknown_and_duplicate_ids_are_dropped_and_backfilled():
    """Hallucinated ids and duplicates are ignored; the gap is filled by model order."""
    result = merge_rankings(
        CANDIDATES, _ranking(("zzz", 99), ("c", 90), ("c", 10)), count=3
    )
    assert [r.candidate.productid for r in result] == ["c", "a", "b"]
    assert [r.source for r in result] == ["llm", "model", "model"]
    assert result[1].relevance_score is None


def test_llm_failure_falls_back_to_model_ranking():
    """With no LLM response the model ranking is returned."""
    result = merge_rankings(CANDIDATES, None, count=2)
    assert [r.candidate.productid for r in result] == ["a", "b"]


def test_too_many_picks_are_truncated():
    """Never return more than the requested count."""
    result = merge_rankings(CANDIDATES, _ranking(("a", 50), ("b", 60), ("c", 70)), count=2)
    assert [r.candidate.productid for r in result] == ["c", "b"]


def test_ties_broken_by_model_score():
    """Equal relevance falls back to the XGBoost score."""
    result = merge_rankings(CANDIDATES, _ranking(("c", 80), ("a", 80)), count=2)
    assert [r.candidate.productid for r in result] == ["a", "c"]
