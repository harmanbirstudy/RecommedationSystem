"""
tests/test_features.py

Feature engineering behaviour on the synthetic shop.
"""

import pandas as pd
import pytest

from app.features import FEATURE_COLUMNS, build_features, build_user_profile, user_regions


def _features_for(shop_data, userid):
    """Build features for one user as of a fixed date, indexed by productid."""
    as_of = pd.Series({userid: pd.Timestamp("2025-06-01")})
    frame = build_features(shop_data.products, shop_data.users, shop_data.lines, [userid], as_of)
    return frame.set_index("productid")


def test_prepare_maps_titles_and_drops_unknown_products(shop_data):
    """Order lines are mapped to productids; discontinued titles are dropped."""
    assert set(shop_data.lines["productid"]) == {"p1", "p2", "p3", "p4", "p5"}
    assert shop_data.users["email"].iloc[0] == "user1@example.com"


def test_one_row_per_product_with_all_feature_columns(shop_data):
    """Every catalogue product gets a row and every feature column exists."""
    frame = _features_for(shop_data, "u1")
    assert len(frame) == len(shop_data.products)
    assert list(frame.columns[1:]) == FEATURE_COLUMNS  # first column is userid


def test_user_regions_uses_most_frequent_state(shop_data):
    """Region is the customer's shipping state."""
    regions = user_regions(shop_data.lines)
    assert regions["u1"] == "Texas"
    assert regions["u4"] == "Ohio"


def test_similar_customers_and_region_favour_peer_purchases(shop_data):
    """u1 (Texas, books) should lean to the book bought by similar Texans, not kitchen items."""
    frame = _features_for(shop_data, "u1")
    assert frame.at["p3", "similar_customers_score"] > frame.at["p4", "similar_customers_score"]
    assert frame.at["p3", "regional_popularity"] > frame.at["p4", "regional_popularity"]
    assert frame.at["p3", "copurchase_max"] > 0


def test_past_order_features(shop_data):
    """times_bought, recency and category affinity reflect the user's own orders."""
    frame = _features_for(shop_data, "u5")
    assert frame.at["p4", "times_bought"] == 2
    assert frame.at["p4", "days_since_bought"] == pytest.approx(
        (pd.Timestamp("2025-06-01") - pd.Timestamp("2025-01-20")).days
    )
    assert pd.isna(frame.at["p1", "days_since_bought"])
    assert frame.at["p4", "category_affinity"] == pytest.approx(1.0)
    assert frame.at["p1", "category_affinity"] == 0


def test_cold_start_user_gets_global_popularity(shop_data):
    """A user with no orders still gets rows; region features fall back to global."""
    frame = _features_for(shop_data, "u6")
    assert (frame["times_bought"] == 0).all()
    assert (frame["regional_popularity"] == frame["product_global_popularity"]).all()
    assert frame["user_n_orders"].eq(0).all()


def test_user_profile_has_no_pii_and_recent_first(shop_data):
    """The LLM profile carries region, categories and recent titles only."""
    profile = build_user_profile(shop_data.products, shop_data.lines, "u2")
    assert profile.region == "Texas"
    assert profile.n_orders == 2
    assert profile.recent_titles[0] == "SQL Performance Explained"
    assert profile.category_affinity == {"Books": 1.0}
    assert profile.purchased_product_ids == {"p1", "p2", "p3"}
