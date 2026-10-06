"""
app/features.py

Feature engineering for (customer, product) pairs.

Every feature is computed only from the `history` order lines passed in, so the
same function builds leak-free training rows (history = everything before each
customer's last order) and live inference rows (history = all orders).

Feature groups:
  - Customer:     order count, items, spend, average order value, recency, tenure
  - Product:      price, category, global popularity
  - Past orders:  times this customer bought the product, days since, category affinity
  - Region:       smoothed share of customers in the customer's state who bought it
  - Similar customers: cosine-weighted purchases of the k most similar customers
  - Co-purchase:  item-item cosine similarity to products the customer already bought
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    # customer
    "user_n_orders",
    "user_n_items",
    "user_total_spend",
    "user_avg_order_value",
    "user_days_since_last_order",
    "user_tenure_days",
    # product
    "product_price",
    "product_category_code",
    "product_global_popularity",
    # customer x product
    "times_bought",
    "days_since_bought",
    "category_affinity",
    "price_ratio_to_user_avg",
    "regional_popularity",
    "region_lift",
    "similar_customers_score",
    "copurchase_max",
    "copurchase_mean",
]

# Number of nearest-neighbour customers used for the similar-customers score
NEIGHBOURS_K = 20
# Pseudo-count that shrinks small regions towards global popularity
REGION_SMOOTHING = 5.0


@dataclass(frozen=True)
class UserProfile:
    """Human-readable summary of a customer, passed to the LLM (no PII)."""

    region: str | None
    n_orders: int
    total_spend: float
    category_affinity: dict[str, float]  # category name -> share of items bought
    recent_titles: list[str]  # most recent purchases first
    purchased_product_ids: set[str]


def _safe_div(num: np.ndarray, den: np.ndarray | float) -> np.ndarray:
    """Element-wise division that returns 0 where the denominator is 0."""
    num = np.asarray(num, dtype=float)
    den = np.broadcast_to(np.asarray(den, dtype=float), num.shape)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _top_k_rows(matrix: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest entries in each row, zeroing the rest."""
    if matrix.shape[1] <= k:
        return matrix
    kth = np.partition(matrix, -k, axis=1)[:, -k][:, None]
    return np.where(matrix >= kth, matrix, 0.0)


def user_regions(history: pd.DataFrame) -> pd.Series:
    """Return each customer's region: the state they ship to most often (latest wins ties)."""
    if history.empty:
        return pd.Series(dtype=object)
    orders = history.drop_duplicates("orderid")
    orders = orders[orders["state"] != ""].sort_values("orderdate", ascending=False)
    counts = orders.groupby(["userid", "state"], sort=False).size().rename("n").reset_index()
    # stable sort keeps the most recent state first among equal counts
    counts = counts.sort_values("n", ascending=False, kind="stable")
    return counts.drop_duplicates("userid").set_index("userid")["state"]


def build_features(
    products: pd.DataFrame,
    users: pd.DataFrame,
    history: pd.DataFrame,
    user_ids: list[str],
    as_of: pd.Series,
) -> pd.DataFrame:
    """
    Build one feature row per (customer in user_ids, product in catalogue).

    Args:
        products: Catalogue (productid, price, category).
        users:    Users table (userid, createddate).
        history:  Order lines visible to the features (userid, orderid, orderdate,
                  state, productid, price, quantity).
        user_ids: Customers to build rows for (may have no history: cold start).
        as_of:    Reference timestamp per userid for recency features.

    Returns:
        DataFrame with userid, productid and FEATURE_COLUMNS, ordered by user then product.
    """
    product_ids = products["productid"].tolist()
    p_index = {pid: i for i, pid in enumerate(product_ids)}
    n_products = len(product_ids)

    # All customers that appear in history (for neighbours/regions) plus the targets
    all_users = sorted(set(history["userid"]).union(user_ids))
    u_index = {uid: i for i, uid in enumerate(all_users)}

    # --- customer x product quantity matrix ---------------------------------
    qty = np.zeros((len(all_users), n_products))
    if not history.empty:
        grouped = history.groupby(["userid", "productid"])["quantity"].sum()
        rows = [u_index[u] for u in grouped.index.get_level_values(0)]
        cols = [p_index[p] for p in grouped.index.get_level_values(1)]
        qty[rows, cols] = grouped.to_numpy()
    bought = (qty > 0).astype(float)

    # --- product popularity -------------------------------------------------
    active = bought.sum(axis=1) > 0
    n_active = max(int(active.sum()), 1)
    buyers_per_product = bought.sum(axis=0)
    global_pop = buyers_per_product / n_active

    # --- region popularity (smoothed towards global) ------------------------
    regions = user_regions(history)
    user_region = np.array([regions.get(u) for u in all_users], dtype=object)
    regional_pop = np.tile(global_pop, (len(all_users), 1))
    for region in pd.unique(regions):
        members = user_region == region
        region_buyers = bought[members].sum(axis=0)
        smoothed = (region_buyers + REGION_SMOOTHING * global_pop) / (
            members.sum() + REGION_SMOOTHING
        )
        regional_pop[members] = smoothed
    region_lift = _safe_div(regional_pop, global_pop)

    # --- similar customers (user-user cosine, top-k neighbours) -------------
    norms = np.sqrt(bought.sum(axis=1))
    user_sim = _safe_div(bought @ bought.T, np.outer(norms, norms))
    np.fill_diagonal(user_sim, 0.0)
    user_sim = _top_k_rows(user_sim, NEIGHBOURS_K)
    similar_score = _safe_div(user_sim @ bought, user_sim.sum(axis=1, keepdims=True))

    # --- co-purchase (item-item cosine) -------------------------------------
    item_norms = np.sqrt(buyers_per_product)
    item_sim = _safe_div(bought.T @ bought, np.outer(item_norms, item_norms))
    np.fill_diagonal(item_sim, 0.0)
    n_owned = bought.sum(axis=1, keepdims=True)
    copurchase_mean = _safe_div(bought @ item_sim, n_owned)
    copurchase_max = (bought[:, :, None] * item_sim[None, :, :]).max(axis=1)

    # --- category affinity --------------------------------------------------
    categories = sorted(products["category"].fillna("").unique())
    c_index = {c: i for i, c in enumerate(categories)}
    product_cat = np.array([c_index[c] for c in products["category"].fillna("")])
    cat_onehot = np.zeros((n_products, len(categories)))
    cat_onehot[np.arange(n_products), product_cat] = 1.0
    user_cat_qty = qty @ cat_onehot
    user_cat_share = _safe_div(user_cat_qty, user_cat_qty.sum(axis=1, keepdims=True))
    category_affinity = user_cat_share[:, product_cat]

    # --- customer aggregates ------------------------------------------------
    hist = history.assign(line_total=history["price"] * history["quantity"])
    per_user = hist.groupby("userid").agg(
        n_orders=("orderid", "nunique"),
        n_items=("quantity", "sum"),
        total_spend=("line_total", "sum"),
        last_order=("orderdate", "max"),
    )
    last_bought = hist.groupby(["userid", "productid"])["orderdate"].max()
    created = users.set_index("userid")["createddate"]

    # --- assemble rows for the requested customers --------------------------
    target_rows = np.array([u_index[u] for u in user_ids], dtype=int)
    n_targets = len(user_ids)
    prices = products["price"].to_numpy(dtype=float)

    stats = per_user.reindex(user_ids)
    n_orders = stats["n_orders"].fillna(0).to_numpy(dtype=float)
    n_items = stats["n_items"].fillna(0).to_numpy(dtype=float)
    spend = stats["total_spend"].fillna(0).to_numpy(dtype=float)
    ref = pd.to_datetime(as_of.reindex(user_ids))
    days_since_last = (ref - stats["last_order"]).dt.days.to_numpy(dtype=float)
    tenure = (ref - created.reindex(user_ids)).dt.days.to_numpy(dtype=float)
    avg_item_price = _safe_div(spend, n_items)

    days_since_bought = np.full((n_targets, n_products), np.nan)
    lb = last_bought.reset_index()
    lb = lb[lb["userid"].isin(user_ids)]
    if not lb.empty:
        pos_map = {u: i for i, u in enumerate(user_ids)}
        r = lb["userid"].map(pos_map).to_numpy(dtype=int)
        c = lb["productid"].map(p_index).to_numpy(dtype=int)
        elapsed = ref.to_numpy()[r] - lb["orderdate"].to_numpy()
        days_since_bought[r, c] = elapsed / np.timedelta64(1, "D")

    def per_user_col(values: np.ndarray) -> np.ndarray:
        """Repeat a per-customer value across all products."""
        return np.repeat(values, n_products)

    def per_product_col(values: np.ndarray) -> np.ndarray:
        """Tile a per-product value across all customers."""
        return np.tile(values, n_targets)

    frame = pd.DataFrame(
        {
            "userid": np.repeat(np.array(user_ids, dtype=object), n_products),
            "productid": per_product_col(np.array(product_ids, dtype=object)),
            "user_n_orders": per_user_col(n_orders),
            "user_n_items": per_user_col(n_items),
            "user_total_spend": per_user_col(spend),
            "user_avg_order_value": per_user_col(_safe_div(spend, n_orders)),
            "user_days_since_last_order": per_user_col(days_since_last),
            "user_tenure_days": per_user_col(tenure),
            "product_price": per_product_col(prices),
            "product_category_code": per_product_col(product_cat.astype(float)),
            "product_global_popularity": per_product_col(global_pop),
            "times_bought": qty[target_rows].ravel(),
            "days_since_bought": days_since_bought.ravel(),
            "category_affinity": category_affinity[target_rows].ravel(),
            "price_ratio_to_user_avg": np.where(
                per_user_col(avg_item_price) > 0,
                per_product_col(prices) / np.maximum(per_user_col(avg_item_price), 1e-9),
                np.nan,
            ),
            "regional_popularity": regional_pop[target_rows].ravel(),
            "region_lift": region_lift[target_rows].ravel(),
            "similar_customers_score": similar_score[target_rows].ravel(),
            "copurchase_max": copurchase_max[target_rows].ravel(),
            "copurchase_mean": copurchase_mean[target_rows].ravel(),
        }
    )
    return frame


def build_user_profile(
    products: pd.DataFrame, history: pd.DataFrame, userid: str, recent_limit: int = 10
) -> UserProfile:
    """Summarise a customer's purchase history for the LLM prompt."""
    mine = history[history["userid"] == userid]
    regions = user_regions(mine)
    catalogue = products.set_index("productid")

    affinity: dict[str, float] = {}
    if not mine.empty:
        cat_qty = (
            mine.assign(category=mine["productid"].map(catalogue["category_name"]))
            .groupby("category")["quantity"]
            .sum()
        )
        affinity = (cat_qty / cat_qty.sum()).round(2).sort_values(ascending=False).to_dict()

    recent = (
        mine.sort_values("orderdate", ascending=False)
        .drop_duplicates("productid")["productid"]
        .head(recent_limit)
        .map(catalogue["title"])
        .tolist()
    )
    return UserProfile(
        region=regions.get(userid),
        n_orders=int(mine["orderid"].nunique()),
        total_spend=round(float((mine["price"] * mine["quantity"]).sum()), 2),
        category_affinity=affinity,
        recent_titles=recent,
        purchased_product_ids=set(mine["productid"]),
    )
