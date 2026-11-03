import pandas as pd

from rtp.data import split, synthetic_events
from rtp.models import Popularity, ndcg, rfm_features
from rtp.scorer import OnlineState, Recommender
from rtp.stream import InMemoryBroker


def test_broker_keeps_per_key_order_and_offsets():
    b = InMemoryBroker(partitions=3)
    for i in range(5):
        b.send("t", "cust-1", {"i": i})
    b.send("t", "cust-2", {"i": 99})
    recs = b.poll("g", "t")
    vals = [v["i"] for p, o, k, v in recs if k == "cust-1"]
    assert vals == [0, 1, 2, 3, 4]
    p, o, _, _ = [r for r in recs if r[2] == "cust-1"][-1]
    b.commit("g", "t", p, o)
    assert all(k != "cust-1" for _, _, k, _ in b.poll("g", "t"))
    assert b.lag("g", "t") == 1


def test_live_state_changes_recommendations():
    df = synthetic_events(n_customers=30, n_items=60, n_invoices=300)
    train, test = split(df, df["ts"].quantile(0.8))
    pop = Popularity().fit(train)
    rec = Recommender(pop)
    st = OnlineState().bootstrap(train)
    c = test["customer"].iat[0]
    before = rec.recommend("repeat", st.hist.get(c, []), test["ts"].iat[0])
    st.update(c, test["ts"].iat[0], ["NEW-ITEM"] * 20)  # a strong new repeat signal must surface immediately
    after = rec.recommend("repeat", st.hist[c], test["ts"].iat[0])
    assert "NEW-ITEM" in after and before != after


def test_metrics_and_features():
    assert ndcg(["a", "b"], {"a"}) == 1.0
    df = synthetic_events(n_customers=20, n_items=40, n_invoices=200)
    f = rfm_features(df, df["ts"].max() + pd.Timedelta(days=1))
    assert {"recency_days", "n_invoices", "revenue_90d"} <= set(f.columns) and (f["n_invoices"] > 0).all()
