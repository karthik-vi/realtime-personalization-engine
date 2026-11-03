# Decisions
**Period freeze:** Jocata, cutoff Jun 2025. torch 2.7.0, pyspark 4.0.0 (2025-05-23; Spark 3.5 does not run on this machine's Java 21), scikit-learn 1.6.1,
pandas 2.2.3, numpy 2.2.5, pyarrow 19.0.1, openpyxl 3.1.5, FastAPI 0.115.12 (listed, not used by scripts). Lock `--exclude-newer 2025-06-30`. No exceptions.

- **ADR-001 Data: UCI Online Retail II (2026-11-03).** Public, CC BY 4.0, timestamped customer events; fintech event data is not public.
- **ADR-002 In-process Kafka-semantics broker (2026-11-03).** Demo must run without a cluster; keyed partitions + offsets keep per-customer ordering semantics.
- **ADR-003 Models (2026-11-03).** Spark MLlib ALS (batch factors) with serve-time fold-in for streaming; PyTorch GRU; sklearn HGB for LTV; repeat-buy and
  popularity baselines because retail data strongly favours repeat purchases.
- **ADR-004 Evaluation: replay with score-before-ingest (2026-11-03).** HR/recall/NDCG/revenue coverage @10; frozen vs live state isolates the value of streaming.
- **ADR-005 Deployment (2026-11-03).** Scripts + Docker; no live API in this repo.
