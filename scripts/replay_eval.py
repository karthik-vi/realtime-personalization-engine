"""Offline replay evaluation of the batch + streaming personalization paths.

1. Train batch models on events before the split date (popularity, Spark ALS, PyTorch GRU, sklearn LTV).
2. Replay test-period invoices through the Kafka-like stream (keyed by customer).
3. For each invoice, BEFORE ingesting it, ask each method for top-10 items and compare with the invoice's items.
   'frozen' = history as of training end; 'live' = history updated from the stream.

All numbers are OFFLINE/SIMULATED replay metrics on public data, not production uplift.

  python scripts/replay_eval.py --events data/processed/events.parquet --out results/replay.json
  python scripts/replay_eval.py --synthetic --out results/replay_synth.json    # 1-minute smoke run
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtp.data import SPLIT_TS, split, synthetic_events  # noqa: E402
from rtp.models import ALSModel, GRUNext, Popularity, ltv_target, ndcg, rfm_features  # noqa: E402
from rtp.scorer import OnlineState, Recommender  # noqa: E402
from rtp.stream import InMemoryBroker  # noqa: E402

METHODS = ["popularity", "repeat", "als", "als+repeat", "gru", "gru+repeat"]


def invoices(df: pd.DataFrame):
    for inv, g in df.groupby("invoice", sort=False):
        yield {"invoice": inv, "customer": g["customer"].iat[0], "ts": g["ts"].iat[0],
               "items": list(dict.fromkeys(g["item"])), "revenue": dict(g.groupby("item")["revenue"].sum())}


def replay(rec: Recommender, train: pd.DataFrame, test: pd.DataFrame, methods, k=10, modes=("frozen", "live"),
           max_invoices: int | None = None, latency_method: str = "als+repeat"):
    broker = InMemoryBroker(partitions=4)
    order = test.sort_values("ts")
    n_sent = 0
    for m in invoices(order):
        broker.send("invoices", m["customer"], m)
        n_sent += 1
        if max_invoices and n_sent >= max_invoices:
            break
    frozen = OnlineState().bootstrap(train)
    live = OnlineState().bootstrap(train)
    seen_train = set(train["customer"])
    acc = {(mo, me): {"hr": [], "recall": [], "ndcg": [], "rev_cov": []} for mo in modes for me in methods}
    acc_ret = {key: {"hr": []} for key in acc}
    lat = []
    batch = broker.poll("scorer", "invoices", max_records=10**9)
    batch.sort(key=lambda r: (pd.Timestamp(r[3]["ts"]), r[3]["invoice"]))
    for p, off, key, m in batch:
        ts = pd.Timestamp(m["ts"])
        rel = set(m["items"])
        rev_total = sum(m["revenue"].values())
        for mo in modes:
            st = frozen if mo == "frozen" else live
            for me in methods:
                t0 = time.perf_counter()
                recs = rec.recommend(me, st.hist.get(key, []), ts, k)
                if mo == "live" and me == latency_method:
                    lat.append(1000 * (time.perf_counter() - t0))
                hit = rel & set(recs)
                a = acc[(mo, me)]
                a["hr"].append(float(bool(hit)))
                a["recall"].append(len(hit) / len(rel))
                a["ndcg"].append(ndcg(recs, rel, k))
                a["rev_cov"].append(sum(m["revenue"][i] for i in hit) / rev_total if rev_total else 0.0)
                if key in seen_train:
                    acc_ret[(mo, me)]["hr"].append(float(bool(hit)))
        t0 = time.perf_counter()
        live.update(key, ts, m["items"])  # ingest AFTER scoring (no leakage)
        broker.commit("scorer", "invoices", p, off)
    out = {}
    for (mo, me), a in acc.items():
        out[f"{mo}/{me}"] = {f"{k_}@{k}": round(float(np.mean(v)), 4) for k_, v in a.items()}
        out[f"{mo}/{me}"]["hr@10_returning"] = round(float(np.mean(acc_ret[(mo, me)]["hr"])), 4) if acc_ret[(mo, me)]["hr"] else None
    meta = {"n_invoices": len(batch), "n_returning": len(acc_ret[(modes[0], methods[0])]["hr"]),
            "consumer_lag_after": broker.lag("scorer", "invoices"),
            "latency_ms_p50": round(statistics.median(lat), 3) if lat else None,
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 3) if lat else None}
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=Path("data/processed/events.parquet"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-invoices", type=int, default=0)
    ap.add_argument("--tune", action="store_true", help="grid-search blend weight on the last train month (validation)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.synthetic:
        df = synthetic_events(seed=a.seed)
        split_ts = df["ts"].quantile(0.75)
    else:
        df = pd.read_parquet(a.events)
        split_ts = SPLIT_TS
    train, test = split(df, split_ts)
    report = {"data": "synthetic" if a.synthetic else str(a.events), "split_ts": str(split_ts),
              "n_train_events": len(train), "n_test_events": len(test), "seed": a.seed}

    t0 = time.time()
    pop = Popularity().fit(train)
    als = ALSModel().fit(train, seed=a.seed)
    report["als_train_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    gru = GRUNext().fit(train, epochs=2 if a.synthetic else 4, seed=a.seed)
    report["gru_train_s"] = round(time.time() - t0, 1)
    report["gru_loss_by_epoch"] = [round(x, 4) for x in gru.losses]

    beta, hl = 1.0, 60.0
    if a.tune:  # validation = last 30 days of train, models refit on the rest
        vs = split_ts - pd.Timedelta(days=30)
        tr2, va = split(train, vs)
        pop2, als2 = Popularity().fit(tr2), ALSModel().fit(tr2, seed=a.seed)
        grid = []
        for b in (0.0, 0.5, 1.0, 2.0, 4.0):
            for h in (30.0, 60.0, 180.0):
                r = Recommender(pop2, als2, None, beta_repeat=b, half_life_days=h)
                res, _ = replay(r, tr2, va, ["als+repeat", "repeat"], modes=("live",),
                                max_invoices=a.max_invoices or 2000)
                grid.append({"beta": b, "half_life": h, **res["live/als+repeat"], "repeat_hr": res["live/repeat"]["hr@10"]})
        best = max(grid, key=lambda g: g["ndcg@10"])
        beta, hl = best["beta"], best["half_life"]
        report["tuning_grid_validation"] = grid
    report["beta_repeat"], report["half_life_days"] = beta, hl

    rec = Recommender(pop, als, gru, beta_repeat=beta, half_life_days=hl)
    res, meta = replay(rec, train, test, METHODS, max_invoices=a.max_invoices or None)
    report["replay"], report["replay_meta"] = res, meta
    for kk, v in res.items():
        print(f"{kk:24s} {v}")
    print(meta)

    # LTV (sklearn): predict next-90-day revenue from RFM at the split date
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor

    tr_at = split_ts - pd.Timedelta(days=90)
    Xtr = rfm_features(df[df["ts"] < split_ts], tr_at)
    ytr = ltv_target(df[df["ts"] < split_ts], tr_at).reindex(Xtr.index).fillna(0.0)
    Xte = rfm_features(df, split_ts)
    yte = ltv_target(df, split_ts).reindex(Xte.index).fillna(0.0)
    reg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, random_state=a.seed)
    reg.fit(np.log1p(Xtr), np.log1p(ytr))
    pred = np.expm1(reg.predict(np.log1p(Xte)))
    naive = Xte["revenue_90d"].values

    def top_decile_capture(score):
        n = max(1, len(score) // 10)
        idx = np.argsort(-score)[:n]
        return float(yte.values[idx].sum() / yte.values.sum()) if yte.values.sum() else 0.0

    report["ltv"] = {
        "n_customers": int(len(yte)), "target": "revenue in the 90 days after split date",
        "model_mae": round(float(np.mean(np.abs(pred - yte.values))), 2), "naive_mae": round(float(np.mean(np.abs(naive - yte.values))), 2),
        "model_spearman": round(float(spearmanr(pred, yte.values).statistic), 4), "naive_spearman": round(float(spearmanr(naive, yte.values).statistic), 4),
        "model_top_decile_revenue_capture": round(top_decile_capture(pred), 4), "naive_top_decile_revenue_capture": round(top_decile_capture(naive), 4),
    }
    print("ltv", report["ltv"])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
