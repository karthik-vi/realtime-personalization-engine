# Data provenance

| Item | Value |
|---|---|
| Dataset | Online Retail II, UCI Machine Learning Repository id 502 (Chen, 2019) |
| URL | https://archive.ics.uci.edu/dataset/502/online+retail+ii (zip 45,622,418 bytes, downloaded 2026-11-03) |
| License | CC BY 4.0 (stated on the UCI page, checked 2026-11-03) |
| Content | All transactions of a UK-based online gift retailer, 2009-12-01 → 2011-12-09; invoice, item code, quantity, price, customer id, country |
| How obtained | `python scripts/prepare_data.py` |

**Not Jocata data.** Jocata's personalization ran on fintech event streams; this is a public retail dataset used to rebuild the
same *pattern* (batch models + streaming scoring on Kafka-style events). Nothing here measures Jocata's +45% LTV.

## Cleaning (`rtp.data.clean`, counts from `data/processed/clean_stats.json`)

| Step | Rows |
|---|---|
| raw (both sheets) | 1,067,371 |
| drop rows without customer id | 824,364 |
| drop cancellations (invoice starts with C) and qty/price ≤ 0 | 805,549 |
| drop non-product codes (POST, M, BANK CHARGES … — codes without digits) | 802,900 |
| drop exact duplicate rows | **776,840** |

Result: 5,853 customers, 4,624 items, 36,613 invoices.

## Splits
- Train: events before **2011-09-01**. Test (replayed through the stream): 2011-09-01 → 2011-12-09.
- Tuning (blend weight, half-life): the last 30 days of train as validation, with models refit on the earlier part.
- The test period is the pre-Christmas peak: seasonal shift is part of the test, deliberately.

## Synthetic generator
`rtp.data.synthetic_events` produces a small dataset with the same schema (customers with category affinities who re-buy
favourites) for tests and the Docker smoke run. No metric in the README comes from it.

## Limitations
- Retail gift purchases, not financial products. Re-purchase behaviour is much stronger here than in most fintech settings.
- Customer id missing for ~23% of raw rows; those (guest) purchases are dropped, which biases toward registered customers.
