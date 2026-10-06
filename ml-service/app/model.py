"""
app/model.py

XGBoost "will this customer buy this product in their next order?" classifier.

Training set (leak-free, temporal):
  For every customer with >= 2 orders, their most recent order is the target.
  Features are built from all order lines EXCEPT every customer's most recent
  order, with recency measured as of that target order's date.
  Label = 1 if the product is in the target order, else 0.

The model is evaluated on a held-out 20% of customers (AUC, hit-rate@k vs a
popularity baseline), then refit on all customers and saved to MODEL_DIR.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from app.db import ShopData
from app.features import FEATURE_COLUMNS, build_features

logger = logging.getLogger(__name__)

MODEL_FILE = "xgb_recommender.json"
METRICS_FILE = "xgb_recommender_metrics.json"

_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
}


@dataclass
class TrainingMetrics:
    """Offline evaluation results saved next to the model."""

    trained_at: str
    n_training_rows: int
    n_customers: int
    positive_rate: float
    holdout_auc: float
    holdout_hit_rate_at_k: float
    baseline_popularity_hit_rate_at_k: float
    k: int
    feature_importance: dict[str, float]


def make_training_set(data: ShopData) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the leak-free training frame described in the module docstring.

    Returns:
        (features, labels) where features includes userid/productid columns.
    """
    orders = data.lines[["orderid", "userid", "orderdate"]].drop_duplicates("orderid")
    order_counts = orders.groupby("userid")["orderid"].nunique()
    eligible = order_counts[order_counts >= 2].index

    last_orders = (
        orders[orders["userid"].isin(eligible)]
        .sort_values(["orderdate", "orderid"])
        .groupby("userid")
        .tail(1)
    )
    history = data.lines[~data.lines["orderid"].isin(last_orders["orderid"])]
    as_of = last_orders.set_index("userid")["orderdate"]
    user_ids = sorted(as_of.index)

    features = build_features(data.products, data.users, history, user_ids, as_of)

    targets = data.lines[data.lines["orderid"].isin(last_orders["orderid"])]
    target_pairs = set(zip(targets["userid"], targets["productid"], strict=True))
    labels = pd.Series(
        [
            (u, p) in target_pairs
            for u, p in zip(features["userid"], features["productid"], strict=True)
        ],
        dtype=int,
        name="label",
    )
    return features, labels


def _new_classifier(labels: pd.Series) -> XGBClassifier:
    """Create a classifier with class weighting for the rare positive label."""
    positives = max(int(labels.sum()), 1)
    negatives = max(len(labels) - positives, 1)
    return XGBClassifier(**_XGB_PARAMS, scale_pos_weight=negatives / positives)


def _hit_rate(
    frame: pd.DataFrame, scores: np.ndarray, labels: pd.Series, k: int, exclude_purchased: bool
) -> float:
    """Share of customers with at least one target-order product in their top-k."""
    ranked = frame[["userid", "times_bought"]].assign(score=scores, label=labels.to_numpy())
    if exclude_purchased:
        ranked = ranked[ranked["times_bought"] == 0]
    top = ranked.sort_values("score", ascending=False).groupby("userid").head(k)
    hits = top.groupby("userid")["label"].max()
    n_users = frame["userid"].nunique()
    return float(hits.sum() / n_users) if n_users else 0.0


def train_model(
    data: ShopData, model_dir: Path, k: int, exclude_purchased: bool
) -> tuple[XGBClassifier, TrainingMetrics]:
    """
    Train, evaluate, refit and persist the recommender model.

    Args:
        data:              Snapshot of the shop tables.
        model_dir:         Directory for the model and metrics files.
        k:                 Cut-off for hit-rate@k (the recommendation count).
        exclude_purchased: Mirror the live policy of hiding already-bought products.

    Returns:
        The model fitted on all customers, and its holdout metrics.

    Raises:
        ValueError: If there is not enough order history to train.
    """
    features, labels = make_training_set(data)
    if features.empty or labels.nunique() < 2:
        raise ValueError("Not enough order history to train (need customers with 2+ orders)")

    groups = features["userid"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(features, labels, groups))

    x_all = features[FEATURE_COLUMNS]
    model = _new_classifier(labels.iloc[train_idx])
    model.fit(x_all.iloc[train_idx], labels.iloc[train_idx])

    test_frame = features.iloc[test_idx]
    test_labels = labels.iloc[test_idx]
    test_scores = model.predict_proba(x_all.iloc[test_idx])[:, 1]
    auc = float(roc_auc_score(test_labels, test_scores))
    hit_rate = _hit_rate(test_frame, test_scores, test_labels, k, exclude_purchased)
    baseline = _hit_rate(
        test_frame,
        test_frame["product_global_popularity"].to_numpy(),
        test_labels,
        k,
        exclude_purchased,
    )
    logger.info(
        "Holdout AUC=%.3f hit-rate@%d=%.3f (popularity baseline %.3f)", auc, k, hit_rate, baseline
    )

    # Refit on every customer for serving
    final = _new_classifier(labels)
    final.fit(x_all, labels)

    importance = dict(
        sorted(
            zip(FEATURE_COLUMNS, final.feature_importances_.round(4).tolist(), strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    metrics = TrainingMetrics(
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_training_rows=len(features),
        n_customers=int(groups.nunique()),
        positive_rate=round(float(labels.mean()), 4),
        holdout_auc=round(auc, 4),
        holdout_hit_rate_at_k=round(hit_rate, 4),
        baseline_popularity_hit_rate_at_k=round(baseline, 4),
        k=k,
        feature_importance=importance,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    final.save_model(model_dir / MODEL_FILE)
    (model_dir / METRICS_FILE).write_text(json.dumps(asdict(metrics), indent=2))
    logger.info("Saved model to %s", model_dir / MODEL_FILE)
    return final, metrics


def load_model(model_dir: Path) -> tuple[XGBClassifier, TrainingMetrics] | None:
    """Load a previously saved model and its metrics, or None if absent/corrupt."""
    model_path = model_dir / MODEL_FILE
    metrics_path = model_dir / METRICS_FILE
    if not (model_path.exists() and metrics_path.exists()):
        return None
    try:
        model = XGBClassifier()
        model.load_model(model_path)
        metrics = TrainingMetrics(**json.loads(metrics_path.read_text()))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Ignoring unreadable saved model in %s: %s", model_dir, exc)
        return None
    if list(model.get_booster().feature_names or []) != FEATURE_COLUMNS:
        logger.warning("Saved model was trained on different features; retraining")
        return None
    return model, metrics
