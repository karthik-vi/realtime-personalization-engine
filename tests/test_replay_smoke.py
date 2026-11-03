"""End-to-end smoke: Spark ALS + GRU + streaming replay + LTV on the synthetic generator (needs Java 17/21)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_replay_synthetic(tmp_path):
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "replay_eval.py"), "--synthetic", "--out", str(out)], check=True, timeout=900)
    r = json.loads(out.read_text())
    assert r["replay_meta"]["consumer_lag_after"] == 0
    assert r["replay"]["live/repeat"]["hr@10"] >= r["replay"]["live/popularity"]["hr@10"]
    assert "model_mae" in r["ltv"]
