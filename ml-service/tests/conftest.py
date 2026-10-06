"""
tests/conftest.py

Synthetic shop data shared by the unit tests (no database or LLM needed).
"""

import os

import pandas as pd
import pytest

# Settings validation requires a key for the default provider; tests never call the LLM
os.environ.setdefault("LLM_PROVIDER", "ollama")

from app.db import ShopData, prepare_shop_data  # noqa: E402


def _line(orderid, userid, date, state, title, qty=1, price="10.00"):
    """One raw order_products row joined with its order."""
    return {
        "orderid": orderid,
        "userid": userid,
        "orderdate": date,
        "state": state,
        "title": title,
        "price": price,
        "quantity": qty,
    }


@pytest.fixture
def shop_data() -> ShopData:
    """
    Tiny shop: books lovers in Texas, kitchen lovers in Ohio.

    u1, u2, u3 (Texas) buy books; u4, u5 (Ohio) buy kitchen items;
    u6 has never ordered (cold start).
    """
    products = pd.DataFrame(
        [
            ("p1", "Clean Code", "37.99", None, "books", "Books"),
            ("p2", "Effective Java", "44.99", None, "books", "Books"),
            ("p3", "SQL Performance Explained", "29.99", None, "books", "Books"),
            ("p4", "Chef's Knife", "49.99", None, "home", "Home & Kitchen"),
            ("p5", "Cutting Board", "19.99", None, "home", "Home & Kitchen"),
            ("p6", "Yoga Mat", "24.99", None, "sports", "Sports & Outdoors"),
        ],
        columns=["productid", "title", "price", "imageurl", "category", "category_name"],
    )
    users = pd.DataFrame(
        [(f"u{i}", f"User{i}@Example.com", "2024-01-01") for i in range(1, 7)],
        columns=["userid", "email", "createddate"],
    )
    lines = pd.DataFrame(
        [
            _line("o1", "u1", "2025-01-01", "Texas", "Clean Code"),
            _line("o2", "u1", "2025-03-01", "Texas", "Effective Java"),
            _line("o3", "u2", "2025-01-05", "Texas", "Clean Code"),
            _line("o3", "u2", "2025-01-05", "Texas", "Effective Java"),
            _line("o4", "u2", "2025-04-01", "Texas", "SQL Performance Explained"),
            _line("o5", "u3", "2025-02-01", "Texas", "Effective Java"),
            _line("o6", "u3", "2025-05-01", "Texas", "SQL Performance Explained"),
            _line("o7", "u4", "2025-01-10", "Ohio", "Chef's Knife"),
            _line("o8", "u4", "2025-02-10", "Ohio", "Cutting Board"),
            _line("o9", "u5", "2025-01-20", "Ohio", "Chef's Knife", qty=2),
            _line("o10", "u5", "2025-03-20", "Ohio", "Cutting Board"),
            # product removed from the catalogue -> dropped
            _line("o10", "u5", "2025-03-20", "Ohio", "Discontinued Gadget"),
        ]
    )
    return prepare_shop_data(products, users, lines)
