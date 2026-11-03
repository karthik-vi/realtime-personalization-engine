# Experiment log (newest first)

UCI Online Retail II, cleaned (776,840 events). Train < 2011-09-01, replayed test 2011-09-01 → 2011-12-09 (7,061 invoices; 5,922 from customers
seen in training). Seed 0, 2026-11-03, CPU. Raw: `results/replay.json`, `results/ltv.json`, logs in `results/`.
**All numbers are offline replay metrics on public data. None of them is, or estimates, Jocata's production +45% LTV.**

## EXP-004 — LTV model loss bug (FAILURE → fix)
- First run (`replay.json` → `ltv`): HistGradientBoosting with `loss="absolute_error"` on log revenue: Spearman 0.039, MAE 515 — worse than the naive
  "last 90 days" baseline (Spearman 0.484, MAE 390). It produced only **2 distinct predictions**: 57% of customers have zero next-90-day revenue, so the
  L1-optimal prediction collapses to the median.
- `scripts/ltv_eval.py` (`results/ltv.json`): `squared_error` → Spearman **0.525** vs naive 0.484, top-decile revenue capture **0.583** vs 0.568, MAE 391 vs 390.
  A small, real ranking gain; no MAE gain.

## EXP-003 — Batch vs streaming replay (test, 7,061 invoices)

| Method | Frozen state (nightly batch) HR@10 | **Live state (streaming)** HR@10 | Live NDCG@10 | Live revenue coverage@10 |
|---|---|---|---|---|
| popularity (last 30 days of train) | 0.339 | 0.339 | 0.068 | 0.040 |
| repeat-buy (frequency × recency) | 0.604 | 0.664 | 0.243 | 0.181 |
| Spark ALS (implicit, user fold-in) | 0.448 | 0.477 | 0.112 | 0.081 |
| ALS + repeat blend | 0.615 | 0.675 | 0.249 | 0.185 |
| PyTorch GRU next-item | 0.374 | 0.383 | 0.082 | 0.052 |
| **GRU + repeat blend** | 0.620 | **0.679** | **0.249** | **0.186** |

- Streaming updates help every personalised method: ALS+repeat revenue coverage 0.151 → 0.185 (+23% relative, offline).
- **Surprise / honest finding:** the plain repeat-buy heuristic carries most of the signal (0.664 live). ALS and GRU alone are far weaker; blending
  adds only +1.1–1.5 HR points on top of repeat-buy. Retail gift buyers re-order the same items.
- Scoring latency (live ALS+repeat, per invoice): p50 3.1 ms, p95 6.5 ms. Consumer lag 0 after replay (every message committed).
- Blend weight / half-life tuned on the last 30 train days (validation): see `tuning_grid_validation` in `replay.json`.

## EXP-002 — Run-time cut (process)
The first full run (GRU on 300k pairs, 4 epochs, 15-point grid) had not finished training after ~100 min on a shared CPU; restarted with GRU on 100k
pairs × 2 epochs and a 6-point grid. GRU numbers are therefore from a deliberately small model.

## EXP-001 — Smoke run on the synthetic generator (`--synthetic`): pipeline + Spark + stream wiring verified; not a result.

## Resume-claim comparison
> Resume claim: +45% user lifetime value | Reproduced here: not measurable — this is offline replay on a public retail dataset. Measured instead: +23% relative revenue coverage of next-invoice items for streaming vs nightly-batch state (ALS+repeat), and an LTV ranking model with Spearman 0.525 vs 0.484 naive | Gap / why: LTV uplift needs a live experiment with real users; no such data exists here.
