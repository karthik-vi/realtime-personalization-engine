"""Download UCI Online Retail II (CC BY 4.0) and write a clean event table.

  python scripts/prepare_data.py --out data/processed/events.parquet
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtp.data import RAW_URL, clean, load_raw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
ap.add_argument("--out", type=Path, default=Path("data/processed/events.parquet"))
a = ap.parse_args()
xlsx = a.raw_dir / "online_retail_II.xlsx"
if not xlsx.exists():
    a.raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RAW_URL, timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(a.raw_dir)
df, stats = clean(load_raw(xlsx))
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(a.out, index=False)
(a.out.parent / "clean_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
