import glob, gzip, csv, sys
from datetime import datetime, timezone

def load_ticks(pattern="ticks/*.csv.gz"):
    ticks = []
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticks.append((float(row["ts_utc"]), float(row["price"])))
    return ticks

if __name__ == "__main__":
    t = load_ticks(sys.argv[1] if len(sys.argv) > 1 else "ticks/*.csv.gz")
    if not t:
        print("No archived ticks yet.")
    else:
        print(f"Loaded {len(t):,} ticks | {datetime.fromtimestamp(t[0][0], timezone.utc).isoformat()} -> {datetime.fromtimestamp(t[-1][0], timezone.utc).isoformat()}")
