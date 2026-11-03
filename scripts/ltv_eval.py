"""LTV (next-90-day revenue) model vs naive baseline, comparing loss functions. Separate from replay_eval for speed.

  python scripts/ltv_eval.py --out results/ltv.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtp.data import SPLIT_TS  # noqa: E402
from rtp.models import ltv_target, rfm_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--events", type=Path, default=Path("data/processed/events.parquet"))
ap.add_argument("--out", type=Path, default=Path("results/ltv.json"))
a = ap.parse_args()
df = pd.read_parquet(a.events)
tr_at = SPLIT_TS - pd.Timedelta(days=90)
hist = df[df["ts"] < SPLIT_TS]
Xtr = rfm_features(hist, tr_at)
ytr = ltv_target(hist, tr_at).reindex(Xtr.index).fillna(0.0)
Xte = rfm_features(df, SPLIT_TS)
yte = ltv_target(df, SPLIT_TS).reindex(Xte.index).fillna(0.0).values


def cap(score):
    idx = np.argsort(-score)[: max(1, len(score) // 10)]
    return round(float(yte[idx].sum() / yte.sum()), 4)


rep = {"n_customers": int(len(yte)), "share_zero_target": round(float((yte == 0).mean()), 4),
       "naive_last_90d": {"mae": round(float(np.mean(np.abs(Xte["revenue_90d"].values - yte))), 2),
                          "spearman": round(float(spearmanr(Xte["revenue_90d"].values, yte).statistic), 4), "top_decile_capture": cap(Xte["revenue_90d"].values)}}
for loss in ("absolute_error", "squared_error"):
    m = HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, random_state=0).fit(np.log1p(Xtr), np.log1p(ytr))
    p = np.expm1(m.predict(np.log1p(Xte)))
    rep[f"hgb_{loss}"] = {"mae": round(float(np.mean(np.abs(p - yte))), 2), "spearman": round(float(spearmanr(p, yte).statistic), 4),
                          "top_decile_capture": cap(p), "n_distinct_predictions": int(len(np.unique(np.round(p, 2))))}
print(json.dumps(rep, indent=1))
a.out.write_text(json.dumps(rep, indent=2))
