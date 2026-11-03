"""Streaming scorer: consumes invoice events, keeps online per-customer state, serves top-k recommendations.

Two state modes are compared in evaluation:
  frozen  - history as of the end of batch training (what a nightly batch job would serve)
  live    - history updated from the stream after every invoice (real-time path)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .models import ALSModel, GRUNext, Popularity, RepeatBuy, repeat_scores


class OnlineState:
    def __init__(self, max_hist: int = 500):
        self.hist: dict[str, list[tuple[pd.Timestamp, str]]] = defaultdict(list)
        self.max_hist = max_hist

    def bootstrap(self, df: pd.DataFrame) -> "OnlineState":
        for c, g in df.sort_values("ts").groupby("customer"):
            self.hist[c] = list(zip(g["ts"], g["item"]))[-self.max_hist:]
        return self

    def update(self, customer: str, ts: pd.Timestamp, items: list[str]) -> None:
        h = self.hist[customer]
        h.extend((ts, i) for i in items)
        if len(h) > self.max_hist:
            del h[: len(h) - self.max_hist]


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


class Recommender:
    def __init__(self, pop: Popularity, als: ALSModel | None = None, gru: GRUNext | None = None,
                 beta_repeat: float = 1.0, half_life_days: float = 60.0):
        self.pop, self.als, self.gru = pop, als, gru
        self.repeat = RepeatBuy(pop, half_life_days)
        self.beta, self.hl = beta_repeat, half_life_days

    def _blend(self, base_scores: np.ndarray | None, item_ids: np.ndarray, hist, now, k, use_repeat: bool) -> list[str]:
        if base_scores is None:
            return self.repeat.recommend(hist, k, now) if use_repeat else self.pop.recommend(hist, k)
        s = _z(base_scores.astype(np.float64))
        if use_repeat and hist:
            rs = repeat_scores(hist, now, self.hl)
            pos = {it: j for j, it in enumerate(item_ids)}
            for it, v in rs.items():
                if it in pos:
                    s[pos[it]] += self.beta * np.log1p(5 * v) * 3
        top = np.argpartition(-s, k)[:k]
        return [str(item_ids[j]) for j in top[np.argsort(-s[top])] if item_ids[j] != "<pad>"][:k]

    def recommend(self, method: str, hist, now, k: int = 10) -> list[str]:
        if method == "popularity":
            return self.pop.recommend(hist, k)
        if method == "repeat":
            return self.repeat.recommend(hist, k, now)
        if method.startswith("als"):
            return self._blend(self.als.scores(hist), self.als.item_ids, hist, now, k, method.endswith("+repeat"))
        if method.startswith("gru"):
            return self._blend(self.gru.scores(hist), self.gru.inv, hist, now, k, method.endswith("+repeat"))
        raise ValueError(method)
