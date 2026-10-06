"""
app/db.py

Read-only loaders for the shoppingwebsite tables (see DataSqlAnalysis/scripts/seed.sql).

order_products is a snapshot of the product at order time and has no productid,
so order lines are joined to the current catalogue by title (products.title is
UNIQUE). Lines for products no longer in the catalogue are dropped.
"""

import logging
from dataclasses import dataclass

import pandas as pd
import psycopg2
import psycopg2.extras

from app.config import get_settings

logger = logging.getLogger(__name__)

_PRODUCTS_SQL = """
    SELECT p.productid, p.title, p.price, p.imageurl, p.category,
           COALESCE(pc.name, p.category) AS category_name
    FROM products p
    LEFT JOIN productcategory pc ON pc.type = p.category
"""

_USERS_SQL = """
    SELECT userid, email, createddate
    FROM users
"""

_ORDER_LINES_SQL = """
    SELECT o.orderid, o.userid, o.orderdate, o.state,
           op.title, op.price, op.quantity
    FROM orders o
    JOIN order_products op ON op.orderid = o.orderid
    WHERE o.userid IS NOT NULL
"""


@dataclass(frozen=True)
class ShopData:
    """In-memory snapshot of the tables the recommender needs."""

    products: pd.DataFrame  # productid, title, price, imageurl, category, category_name
    users: pd.DataFrame  # userid, email, createddate
    lines: pd.DataFrame  # orderid, userid, orderdate, state, productid, price, quantity


class DatabaseError(Exception):
    """Raised when the shoppingwebsite database cannot be read."""


def _fetch(cur: psycopg2.extensions.cursor, sql: str) -> pd.DataFrame:
    """Run a query and return the rows as a DataFrame (columns kept when empty)."""
    cur.execute(sql)
    columns = [desc.name for desc in cur.description]
    return pd.DataFrame([dict(row) for row in cur.fetchall()], columns=columns)


def load_shop_data() -> ShopData:
    """
    Load products, users and order lines from PostgreSQL in one read-only transaction.

    Returns:
        ShopData with typed, cleaned DataFrames.

    Raises:
        DatabaseError: If PostgreSQL is unreachable or a query fails.
    """
    dsn = get_settings().database_url.get_secret_value()
    try:
        conn = psycopg2.connect(dsn)
    except psycopg2.OperationalError as exc:
        raise DatabaseError(f"Could not connect to PostgreSQL: {exc}") from exc

    try:
        conn.set_session(readonly=True)
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            products = _fetch(cur, _PRODUCTS_SQL)
            users = _fetch(cur, _USERS_SQL)
            lines = _fetch(cur, _ORDER_LINES_SQL)
    except psycopg2.Error as exc:
        raise DatabaseError(str(exc)) from exc
    finally:
        conn.close()

    return prepare_shop_data(products, users, lines)


def prepare_shop_data(
    products: pd.DataFrame, users: pd.DataFrame, lines: pd.DataFrame
) -> ShopData:
    """
    Normalise raw table rows: numeric prices, timestamps, lower-cased emails,
    and order lines mapped to productid via title.
    """
    products = products.copy()
    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0.0)
    products = products.sort_values("productid").reset_index(drop=True)

    users = users.copy()
    users["email"] = users["email"].str.strip().str.lower()
    users["createddate"] = pd.to_datetime(users["createddate"])

    lines = lines.copy()
    title_to_id = dict(zip(products["title"], products["productid"], strict=True))
    lines["productid"] = lines["title"].map(title_to_id)
    dropped = int(lines["productid"].isna().sum())
    if dropped:
        logger.info("Dropped %d order line(s) for products no longer in the catalogue", dropped)
    lines = lines.dropna(subset=["productid"])
    lines["orderdate"] = pd.to_datetime(lines["orderdate"])
    lines["price"] = pd.to_numeric(lines["price"], errors="coerce").fillna(0.0)
    lines["quantity"] = lines["quantity"].fillna(1).astype(int)
    lines["state"] = lines["state"].fillna("").str.strip()
    lines = lines.drop(columns=["title"]).reset_index(drop=True)

    logger.info(
        "Loaded %d products, %d users, %d order lines",
        len(products),
        len(users),
        len(lines),
    )
    return ShopData(products=products, users=users, lines=lines)
