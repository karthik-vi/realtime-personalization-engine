"""Kafka-like event stream that runs in-process (no broker needed for the demo).

Semantics mirrored from Kafka: topics with N partitions, key-hash partitioning (so all events of one
customer stay ordered), append-only logs, consumer groups with committed offsets, at-least-once poll/commit.
`make_stream()` returns this in-memory broker unless KAFKA_BOOTSTRAP is set, in which case a thin
kafka-python adapter is used (adapter is untested in this repo: no broker was available).
"""
from __future__ import annotations

import json
import os
import zlib
from collections import defaultdict


class InMemoryBroker:
    def __init__(self, partitions: int = 4):
        self.partitions = partitions
        self.logs: dict[str, list[list]] = defaultdict(lambda: [[] for _ in range(self.partitions)])
        self.offsets: dict[tuple[str, str, int], int] = defaultdict(int)  # (group, topic, partition) -> next offset

    def partition_for(self, key: str) -> int:
        return zlib.crc32(key.encode()) % self.partitions

    def send(self, topic: str, key: str, value: dict) -> tuple[int, int]:
        p = self.partition_for(key)
        log = self.logs[topic][p]
        log.append((key, json.dumps(value, default=str)))
        return p, len(log) - 1

    def poll(self, group: str, topic: str, max_records: int = 500) -> list[tuple[int, int, str, dict]]:
        """Round-robin over partitions; returns (partition, offset, key, value). Does not commit."""
        out = []
        for p in range(self.partitions):
            start = self.offsets[(group, topic, p)]
            for off, (k, v) in enumerate(self.logs[topic][p][start: start + max_records], start=start):
                out.append((p, off, k, json.loads(v)))
        return out

    def commit(self, group: str, topic: str, partition: int, offset: int) -> None:
        self.offsets[(group, topic, partition)] = offset + 1

    def lag(self, group: str, topic: str) -> int:
        return sum(len(self.logs[topic][p]) - self.offsets[(group, topic, p)] for p in range(self.partitions))


class KafkaPythonAdapter:  # pragma: no cover - needs a running broker
    def __init__(self, bootstrap: str):
        from kafka import KafkaConsumer, KafkaProducer

        self._KafkaConsumer = KafkaConsumer
        self.bootstrap = bootstrap
        self.producer = KafkaProducer(bootstrap_servers=bootstrap, key_serializer=str.encode,
                                      value_serializer=lambda v: json.dumps(v, default=str).encode())
        self.consumers = {}

    def send(self, topic, key, value):
        md = self.producer.send(topic, key=key, value=value).get(timeout=10)
        return md.partition, md.offset

    def poll(self, group, topic, max_records=500):
        c = self.consumers.setdefault((group, topic), self._KafkaConsumer(
            topic, bootstrap_servers=self.bootstrap, group_id=group, enable_auto_commit=False,
            value_deserializer=lambda b: json.loads(b.decode()), key_deserializer=lambda b: b.decode()))
        recs = c.poll(timeout_ms=500, max_records=max_records)
        return [(tp.partition, r.offset, r.key, r.value) for tp, rs in recs.items() for r in rs]

    def commit(self, group, topic, partition, offset):
        from kafka import TopicPartition
        from kafka.structs import OffsetAndMetadata

        self.consumers[(group, topic)].commit({TopicPartition(topic, partition): OffsetAndMetadata(offset + 1, None, -1)})


def make_stream(partitions: int = 4):
    bs = os.getenv("KAFKA_BOOTSTRAP")
    return KafkaPythonAdapter(bs) if bs else InMemoryBroker(partitions)
