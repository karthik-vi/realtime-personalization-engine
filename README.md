# realtime-personalization-engine

Batch models (Spark MLlib ALS, PyTorch GRU, scikit-learn LTV) plus a streaming scoring path that consumes Kafka-style events, keeps per-customer state
online and serves top-10 recommendations, evaluated by replaying a public retail transaction log.

## What this is / is not

| | |
|---|---|
| **Reconstructs** | Resume (Jocata, Machine Learning Engineer, Jan 2024 – Jun 2025), verbatim: *"Increased user lifetime value by 45% by engineering a real-time personalization engine with PyTorch, Scikit-learn, and Spark MLlib predictive models on Kafka streaming data."* |
| **Is** | A reconstruction on **UCI Online Retail II** (CC BY 4.0). The Kafka part is an in-process broker with Kafka semantics (partitions, keys, consumer-group offsets); a kafka-python adapter exists but is untested (no broker). |
| **Is not** | Jocata's engine, data or results. **Every uplift number here is offline/simulated replay on retail data, not Jocata's production +45% LTV.** |

### Period freeze
Jocata, **cutoff Jun 2025**. Python 3.11; PyTorch 2.7.0 (2025-04-23); PySpark 4.0.0 (2025-05-23, MLlib ALS); scikit-learn 1.6.1; pandas 2.2.3; pyarrow 19.0.1.
Lock compiled with `--exclude-newer 2025-06-30`. Details in `docs/DECISIONS.md`.

## Architecture
```
events.parquet ─┬─ batch (train < 2011-09-01): popularity │ Spark MLlib ALS (implicit) → item factors │ PyTorch GRU next-item │ sklearn LTV (RFM)
                └─ stream (test period): producer → topic "invoices" (4 partitions, key = customer) → consumer group "scorer"
                     for each invoice: score top-10 BEFORE ingesting (no leakage) → compare with the invoice → update online state → commit offset
online state = per-customer history; ALS user vector folded in at request time from live history (Hu et al. 2008), blended with repeat-buy
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate     # needs Java 17/21 for Spark
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pytest -q                                              # includes a synthetic end-to-end Spark + stream run
python scripts/prepare_data.py                         # downloads UCI data (45 MB)
python scripts/replay_eval.py --tune --out results/replay.json      # ~40-60 min CPU
python scripts/ltv_eval.py
```

## Results (offline replay, 7,061 test invoices)

| Method | HR@10 frozen (batch) | HR@10 live (streaming) | Revenue coverage@10 live |
|---|---|---|---|
| Popularity | 0.339 | 0.339 | 0.040 |
| Repeat-buy | 0.604 | 0.664 | 0.181 |
| ALS + repeat | 0.615 | 0.675 | 0.185 |
| **GRU + repeat** | 0.620 | **0.679** | **0.186** |

Streaming vs batch state: +6 HR points, +23% relative revenue coverage (ALS+repeat). Scoring p50 3.1 ms. LTV model (squared error): Spearman 0.525 vs
naive 0.484. **The repeat-buy heuristic carries most of the gain; the ML models add ~1–1.5 points on top.**

> Resume claim: +45% user lifetime value | Reproduced here: not measurable offline on public retail data; measured +23% relative revenue coverage (streaming vs batch state) and LTV ranking Spearman 0.525 vs 0.484 | Gap / why: LTV uplift requires a live A/B test with real users.

## Limitations
Retail repurchase data (not fintech); single seed; small GRU (EXP-002); the kafka-python adapter and Docker/Spark-on-cluster paths are untested; "revenue
coverage" is a proxy, not incremental revenue.

## How to extend
Real Kafka via docker-compose; feature store (online/offline parity checks); A/B harness with simulated user response; exclude-already-bought variants for discovery.
