"""Batch models (trained offline) used by the streaming scorer.

  Popularity        global item counts in the last N days of training data (baseline)
  RepeatBuy         the customer's own past items, frequency x recency (strong retail baseline)
  ALS (Spark MLlib) implicit-feedback matrix factorisation; user vectors are *folded in* at serve time
                    from the customer's live history, so new events change recommendations immediately
  GRU (PyTorch)     next-item sequence model over the customer's item history
  LTV (sklearn)     HistGradientBoosting on RFM features -> next-90-day revenue
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd


class Popularity:
    def fit(self, train: pd.DataFrame, days: int = 30):
        recent = train[train["ts"] >= train["ts"].max() - pd.Timedelta(days=days)]
        self.ranked = [i for i, _ in Counter(recent["item"]).most_common()]
        return self

    def recommend(self, hist: list, k: int = 10) -> list[str]:
        return self.ranked[:k]


def repeat_scores(hist: list[tuple[pd.Timestamp, str]], now: pd.Timestamp, half_life_days: float = 60.0) -> dict[str, float]:
    s: dict[str, float] = {}
    for t, it in hist:
        age = max(0.0, (now - t).total_seconds() / 86400)
        s[it] = s.get(it, 0.0) + 0.5 ** (age / half_life_days)
    return s


class RepeatBuy:
    def __init__(self, pop: Popularity, half_life_days: float = 60.0):
        self.pop, self.hl = pop, half_life_days

    def recommend(self, hist, k=10, now=None):
        s = repeat_scores(hist, now or (hist[-1][0] if hist else pd.Timestamp.now()), self.hl)
        out = [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])][:k]
        return out + [i for i in self.pop.ranked if i not in out][: k - len(out)]


class ALSModel:
    """Spark MLlib ALS (implicit). Item factors are exported to NumPy for fast serving."""

    def fit(self, train: pd.DataFrame, rank: int = 32, reg: float = 0.1, alpha: float = 20.0, iters: int = 12, seed: int = 0):
        from pyspark.ml.recommendation import ALS
        from pyspark.sql import SparkSession

        spark = (SparkSession.builder.master("local[2]").appName("rtp-als")
                 .config("spark.driver.memory", "3g").config("spark.ui.enabled", "false")
                 .config("spark.sql.shuffle.partitions", "8").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        agg = train.groupby(["customer", "item"]).size().rename("n").reset_index()
        self.users = {u: i for i, u in enumerate(sorted(agg["customer"].unique()))}
        self.items = {it: i for i, it in enumerate(sorted(agg["item"].unique()))}
        self.item_ids = np.array(sorted(self.items, key=self.items.get))
        pdf = pd.DataFrame({"u": agg["customer"].map(self.users), "i": agg["item"].map(self.items),
                            "r": np.log1p(agg["n"]).astype(float)})
        df = spark.createDataFrame(pdf)
        als = ALS(rank=rank, regParam=reg, alpha=alpha, maxIter=iters, implicitPrefs=True, seed=seed,
                  userCol="u", itemCol="i", ratingCol="r", coldStartStrategy="drop", nonnegative=False)
        m = als.fit(df)
        self.alpha, self.reg = alpha, reg
        Y = np.zeros((len(self.items), rank), np.float32)
        for row in m.itemFactors.collect():
            Y[row["id"]] = row["features"]
        X = np.zeros((len(self.users), rank), np.float32)
        for row in m.userFactors.collect():
            X[row["id"]] = row["features"]
        self.Y, self.X = Y, X
        self.YtY = Y.T @ Y
        spark.stop()
        return self

    def fold_in(self, hist: list[tuple[pd.Timestamp, str]]) -> np.ndarray | None:
        """Implicit-ALS user vector from current history (Hu, Koren & Volinsky 2008, eq. 4)."""
        cnt = Counter(it for _, it in hist if it in self.items)
        if not cnt:
            return None
        idx = np.array([self.items[i] for i in cnt])
        c = 1 + self.alpha * np.log1p(np.array(list(cnt.values()), np.float32))
        Yi = self.Y[idx]
        A = self.YtY + (Yi.T * (c - 1)) @ Yi + self.reg * np.eye(self.Y.shape[1], dtype=np.float32)
        b = (Yi.T * c).sum(1)
        return np.linalg.solve(A, b)

    def scores(self, hist) -> np.ndarray | None:
        u = self.fold_in(hist)
        return None if u is None else self.Y @ u


class GRUNext:
    """Next-item GRU over the customer's item sequence (PyTorch)."""

    def fit(self, train: pd.DataFrame, max_len: int = 50, dim: int = 64, epochs: int = 4, seed: int = 0, max_items: int = 3000):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        top = [i for i, _ in Counter(train["item"]).most_common(max_items)]
        self.vocab = {it: j + 1 for j, it in enumerate(top)}  # 0 = pad / OOV
        self.inv = np.array(["<pad>"] + top)
        self.max_len = max_len
        seqs = train.sort_values("ts").groupby("customer")["item"].apply(lambda s: [self.vocab.get(i, 0) for i in s if i in self.vocab])
        X, y = [], []
        for s in seqs:
            for t in range(1, len(s)):
                X.append(s[max(0, t - max_len): t])
                y.append(s[t])
        rng = np.random.default_rng(seed)
        if len(X) > 300_000:
            sel = rng.choice(len(X), 300_000, replace=False)
            X, y = [X[i] for i in sel], [y[i] for i in sel]
        V = len(self.inv)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(V, dim, padding_idx=0)
                s.gru = nn.GRU(dim, dim, batch_first=True)
                s.out = nn.Linear(dim, V)

            def forward(s, x):
                h, _ = s.gru(s.emb(x))
                return s.out(h[:, -1])

        self.net = Net()
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        lossf = nn.CrossEntropyLoss()
        self.losses = []
        for ep in range(epochs):
            order = rng.permutation(len(X))
            tot = 0.0
            for b in range(0, len(order), 512):
                j = order[b: b + 512]
                xb = torch.tensor(_pad([X[k] for k in j], max_len))
                yb = torch.tensor([y[k] for k in j])
                opt.zero_grad()
                loss = lossf(self.net(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(j)
            self.losses.append(tot / len(order))
        self.net.eval()
        return self

    def scores(self, hist) -> np.ndarray | None:
        import torch

        s = [self.vocab.get(it, 0) for _, it in hist if it in self.vocab][-self.max_len:]
        if not s:
            return None
        with torch.no_grad():
            return self.net(torch.tensor(_pad([s], self.max_len)))[0].numpy()


def _pad(seqs, L):
    out = np.zeros((len(seqs), L), np.int64)
    for i, s in enumerate(seqs):
        s = s[-L:]
        if s:
            out[i, -len(s):] = s
    return out


def rfm_features(df: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    h = df[df["ts"] < at]
    g = h.groupby("customer")
    inv = h.groupby(["customer", "invoice"])["revenue"].sum().groupby("customer")
    f = pd.DataFrame({
        "recency_days": (at - g["ts"].max()).dt.total_seconds() / 86400,
        "tenure_days": (at - g["ts"].min()).dt.total_seconds() / 86400,
        "n_invoices": g["invoice"].nunique(),
        "revenue_total": g["revenue"].sum(),
        "revenue_90d": h[h["ts"] >= at - pd.Timedelta(days=90)].groupby("customer")["revenue"].sum(),
        "revenue_30d": h[h["ts"] >= at - pd.Timedelta(days=30)].groupby("customer")["revenue"].sum(),
        "avg_basket": inv.mean(),
        "n_items": g["item"].nunique(),
    }).fillna(0.0)
    return f


def ltv_target(df: pd.DataFrame, at: pd.Timestamp, days: int = 90) -> pd.Series:
    w = df[(df["ts"] >= at) & (df["ts"] < at + pd.Timedelta(days=days))]
    return w.groupby("customer")["revenue"].sum()


def ndcg(rec: list, rel: set, k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rec[:k]) if r in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg if idcg else 0.0
