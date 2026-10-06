"""
tests/test_model.py

Leak-free training-set construction and a train/save/load round trip.
"""

from app.model import load_model, make_training_set, train_model


def test_training_set_targets_last_order_only(shop_data):
    """Labels come from each multi-order customer's last order; its lines are not in history."""
    features, labels = make_training_set(shop_data)
    positives = set(
        zip(features.loc[labels == 1, "userid"], features.loc[labels == 1, "productid"], strict=True)
    )
    # last orders: u1 o2(p2), u2 o4(p3), u3 o6(p3), u4 o8(p5), u5 o10(p5)
    assert positives == {("u1", "p2"), ("u2", "p3"), ("u3", "p3"), ("u4", "p5"), ("u5", "p5")}
    # u1's history excludes the target order, so p2 was never bought before
    row = features[(features["userid"] == "u1") & (features["productid"] == "p2")].iloc[0]
    assert row["times_bought"] == 0
    # u6 never ordered, so it is not a training customer
    assert "u6" not in set(features["userid"])


def test_train_save_and_load_round_trip(shop_data, tmp_path):
    """A trained model is persisted and reloaded with its metrics."""
    model, metrics = train_model(shop_data, tmp_path, k=2, exclude_purchased=True)
    assert metrics.n_customers == 5
    assert 0.0 <= metrics.holdout_hit_rate_at_k <= 1.0

    loaded = load_model(tmp_path)
    assert loaded is not None
    reloaded, reloaded_metrics = loaded
    assert reloaded_metrics.trained_at == metrics.trained_at
