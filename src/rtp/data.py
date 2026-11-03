"""UCI Online Retail II -> clean event table.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0), a UK online retailer's
transactions 2009-12-01 .. 2011-12-09. NOT Jocata data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SPLIT_TS = pd.Timestamp("2011-09-01")  # train: before; test (streamed): on/after


def load_raw(xlsx: Path) -> pd.DataFrame:
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    return pd.concat(sheets.values(), ignore_index=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"raw_rows": len(df)}
    df = df.rename(columns={"Customer ID": "customer", "StockCode": "item", "InvoiceDate": "ts", "Invoice": "invoice",
                            "Quantity": "qty", "Price": "price", "Country": "country", "Description": "desc"})
    df = df.dropna(subset=["customer"])
    stats["after_drop_missing_customer"] = len(df)
    df = df[~df["invoice"].str.startswith("C")]  # cancellations
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    stats["after_drop_cancel_nonpositive"] = len(df)
    df = df[df["item"].str.contains(r"\d", regex=True)]  # drop POST, M, BANK CHARGES, etc.
    stats["after_drop_non_product_codes"] = len(df)
    df = df.drop_duplicates()
    stats["after_drop_exact_duplicates"] = len(df)
    df["customer"] = df["customer"].astype(int).astype(str)
    df["item"] = df["item"].str.upper().str.strip()
    df["revenue"] = df["qty"] * df["price"]
    df = df.sort_values(["ts", "invoice"]).reset_index(drop=True)
    stats.update(customers=int(df["customer"].nunique()), items=int(df["item"].nunique()),
                 invoices=int(df["invoice"].nunique()), ts_min=str(df["ts"].min()), ts_max=str(df["ts"].max()))
    return df[["ts", "invoice", "customer", "item", "qty", "price", "revenue", "country", "desc"]], stats


def split(df: pd.DataFrame, ts: pd.Timestamp = SPLIT_TS):
    return df[df["ts"] < ts].reset_index(drop=True), df[df["ts"] >= ts].reset_index(drop=True)


def synthetic_events(n_customers: int = 200, n_items: int = 300, n_invoices: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Small synthetic stand-in with the same schema (for tests / offline quickstart). Customers have
    item affinities over 10 latent 'categories' and re-buy their favourites."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 10, n_items)
    pref = rng.dirichlet(np.ones(10) * 0.3, n_customers)
    price = rng.uniform(0.5, 15, n_items).round(2)
    ts0 = pd.Timestamp("2011-01-01")
    rows = []
    for inv in range(n_invoices):
        c = int(rng.integers(n_customers))
        t = ts0 + pd.Timedelta(minutes=int(inv * 120 + rng.integers(60)))
        for _ in range(int(rng.integers(1, 8))):
            k = rng.choice(10, p=pref[c])
            items = np.where(cat == k)[0]
            it = int(items[(c * 7 + int(rng.integers(0, 3))) % len(items)] if rng.random() < 0.5 else rng.choice(items))
            q = int(rng.integers(1, 12))
            rows.append((t, f"S{inv}", str(c), f"I{it:04d}", q, price[it], q * price[it], "Synthetic", f"item {it}"))
    return pd.DataFrame(rows, columns=["ts", "invoice", "customer", "item", "qty", "price", "revenue", "country", "desc"])
