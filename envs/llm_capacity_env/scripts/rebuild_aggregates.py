"""Rebuild the shipped aggregate from upstream BurstGPT CSV or CSV.gz files.

Download the three v2.0 release assets yourself; this program has no network access.
Writes only the requested output path, never edits raw data.
"""

import argparse
import csv
import gzip
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    totals = {}
    days = set()
    for part in (1, 2, 3):
        raw = args.raw_dir / f"BurstGPT_{part}.csv"
        if not raw.exists():
            raw = raw.with_suffix(".csv.gz")
        opener = gzip.open if raw.suffix == ".gz" else open
        with opener(raw, "rt", newline="") as f:
            for row in csv.DictReader(f):
                if row["Log Type"] != "Conversation log" or row["Model"] not in (
                    "ChatGPT",
                    "GPT-4",
                ):
                    continue
                second = int(float(row["Timestamp"]))
                day, hour = second // 86400, second % 86400 // 3600
                days.add(day)
                key = (day, hour, row["Model"])
                totals[key] = totals.get(key, 0) + int(row["Total tokens"])
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log_day", "hour", "model", "total_tokens"])
        for day in sorted(days):
            for hour in range(24):
                for model in ("ChatGPT", "GPT-4"):
                    w.writerow([day, hour, model, totals.get((day, hour, model), 0)])
    print("days=", len(days), "sha256=", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
