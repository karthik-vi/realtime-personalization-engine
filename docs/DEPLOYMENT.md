# Deployment
Local: README quickstart (Java 17/21 required). Docker: `docker build -t rtp . && docker run --rm rtp` (synthetic smoke run).
Real Kafka: set `KAFKA_BOOTSTRAP` (adapter untested). Latency: ~3 ms p50 per scoring call on CPU; Spark ALS training on 600k events takes minutes.
Not production-ready: no feature store, no model registry, no online A/B testing, single-process consumer.
