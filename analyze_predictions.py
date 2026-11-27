"""Slice the bot's prediction audit log and test it against the breakeven win rate.

Usage (standard library only):
    python3 analyze_predictions.py                     # reads predictions/*.csv.gz
    python3 analyze_predictions.py path/to/file.csv.gz --min-n 5
"""
import argparse
import csv
import glob
import gzip
import math
import sys
from collections import defaultdict
from datetime import datetime

COLS = [
    "predicted_at", "resolved_at", "utc_hour", "direction", "confidence",
    "entry_price", "exit_price", "outcome", "correct", "lgb_proba",
    "nn_proba", "ensemble_proba", "edge", "volatility", "vol_regime",
    "gate_open", "features",
]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(paths):
    rows, bad = [], 0
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or header[0] not in ("predicted_at",):
                continue
            for rec in reader:
                if len(rec) < 17:
                    bad += 1
                    continue
                if len(rec) > 17:  # legacy format: unquoted features list split across columns
                    rec = rec[:16] + [",".join(rec[16:])]
                r = dict(zip(COLS, rec))
                try:
                    r["t"] = datetime.fromisoformat(r["predicted_at"].replace("Z", "+00:00"))
                    r["utc_hour"] = int(r["utc_hour"])
                except Exception:
                    bad += 1
                    continue
                r["entry"], r["exit"] = fnum(r["entry_price"]), fnum(r["exit_price"])
                r["proba"] = fnum(r["ensemble_proba"])
                r["gate_bool"] = r["gate_open"].strip().lower() == "true"
                if (r["entry"] is not None and not (1.00 <= r["entry"] <= 1.18)) or (
                    r["exit"] is not None and not (1.00 <= r["exit"] <= 1.18)
                ):
                    bad += 1
                    continue
                rows.append(r)
    rows.sort(key=lambda r: r["t"])
    return rows, bad


def despace(rows, spacing):
    if spacing <= 0:
        return list(rows)
    out, last = [], None
    for r in rows:
        if last is None or (r["t"] - last).total_seconds() >= spacing:
            out.append(r)
            last = r["t"]
    return out


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = w / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    m = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (c - m) / d), min(1.0, (c + m) / d)


def verdict(lo, hi, be):
    if lo > be:
        return "ABOVE breakeven"
    if hi < be:
        return "below breakeven"
    return "inconclusive"


def scored(rows):
    return [r for r in rows if r["outcome"] in ("WIN", "LOSS")]


def stats(rows):
    s = scored(rows)
    w = sum(1 for r in s if r["outcome"] == "WIN")
    ties = sum(1 for r in rows if r["outcome"] == "TIE")
    return w, len(s), ties


def table(title, groups, be, min_n, order=None):
    print(f"\n== {title} ==")
    print(f"{'slice':<16}{'n':>6}{'ties':>6}{'win%':>8}{'95% interval':>17}{'1st/2nd half':>15}  verdict")
    keys = order or sorted(groups, key=lambda k: str(k))
    shown = 0
    for k in keys:
        rs = groups.get(k, [])
        w, n, ties = stats(rs)
        if n < min_n:
            continue
        shown += 1
        lo, hi = wilson(w, n)
        mid = len(rs) // 2
        w1, n1, _ = stats(rs[:mid])
        w2, n2, _ = stats(rs[mid:])
        halves = f"{100*w1/n1:4.0f}/{100*w2/n2:<4.0f}" if n1 >= 5 and n2 >= 5 else "   n/a"
        print(f"{str(k):<16}{n:>6}{ties:>6}{100*w/n:>7.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{halves:>15}  {verdict(lo, hi, be)}")
    if shown == 0:
        print(f"(no slice has {min_n}+ scored samples yet)")
    return shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--spacing", type=int, default=300)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()

    paths = a.files or sorted(glob.glob("predictions/*.csv.gz"))
    if not paths:
        sys.exit("No files found. Put predictions/*.csv.gz in this folder or pass file paths.")
    be = 1.0 / (1.0 + a.payout)

    raw, bad = load(paths)
    if not raw:
        sys.exit("No readable prediction rows.")
    rows = despace(raw, a.spacing)
    print(f"Files: {len(paths)} | raw rows: {len(raw):,} | after keeping one per {a.spacing}s: {len(rows):,}"
          + (f" | unreadable rows skipped: {bad}" if bad else ""))
    print(f"Range: {raw[0]['t']:%Y-%m-%d %H:%M} to {raw[-1]['t']:%Y-%m-%d %H:%M} UTC")
    print(f"Breakeven at payout {a.payout:.2f}: {100*be:.2f}%")

    w, n, ties = stats(rows)
    lo, hi = wilson(w, n)
    print(f"\nOVERALL (every {a.spacing}s, all confidence levels): {w}/{n} = {100*w/max(n,1):.1f}% (ties={ties})  "
          f"[{100*lo:.1f}, {100*hi:.1f}]  -> {verdict(lo, hi, be)}")
    if n:
        p = w / n
        print(f"Expected profit per 1.00 staked at observed rate: {p*a.payout - (1 - p):+.3f}u")

    # Market drift baseline: always BUY vs always SELL
    mv = [
        (r["exit"] > r["entry"])
        for r in rows
        if r["entry"] is not None and r["exit"] is not None and r["exit"] not in (r["entry"],)
    ]
    if mv:
        up = sum(mv)
        print(f"Baseline: price rose in {100*up/len(mv):.1f}% of {len(mv)} windows "
              f"(always BUY {100*up/len(mv):.1f}%, always SELL {100*(len(mv)-up)/len(mv):.1f}%)")

    def group(keyfn):
        g = defaultdict(list)
        for r in rows:
            g[keyfn(r)].append(r)
        return g

    allowed = {8, 9, 21, 22}
    hour_labels = [f"{h:02d}:00 {'*' if h in allowed else ' '}" for h in range(24)]

    shown = 0
    shown += table("By confidence", group(lambda r: r["confidence"]), be, a.min_n,
                   order=["High", "Medium", "Low"])
    shown += table("By UTC hour (* = allowed window)",
                   group(lambda r: f"{r['utc_hour']:02d}:00 {'*' if r['utc_hour'] in allowed else ' '}"),
                   be, a.min_n, order=hour_labels)
    shown += table("By volatility regime", group(lambda r: r["vol_regime"]), be, a.min_n,
                   order=["low", "normal", "high", "spike"])
    shown += table("By direction", group(lambda r: r["direction"]), be, a.min_n,
                   order=["BUY", "SELL"])
    shown += table("By gate open", group(lambda r: "OPEN" if r["gate_bool"] else "BLOCKED"), be, a.min_n,
                   order=["OPEN", "BLOCKED"])

    # Calibration table
    print("\n== Calibration (predicted probability of chosen side vs actual win rate) ==")
    print(f"{'predicted':<13}{'n':>6}{'avg pred':>10}{'actual%':>10}{'calib err':>11}{'95% interval':>17}")
    buckets = defaultdict(list)
    for r in scored(rows):
        if r["proba"] is None:
            continue
        pdir = r["proba"] if r["direction"] == "BUY" else (1.0 - r["proba"])
        b_idx = min(int(pdir * 20) / 20.0, 0.95)
        buckets[b_idx].append((pdir, r["outcome"] == "WIN"))
    for b in sorted(buckets):
        items = buckets[b]
        if len(items) >= a.min_n:
            bn = len(items)
            bw = sum(1 for _, win in items if win)
            avg_p = sum(p for p, _ in items) / bn
            act_p = bw / bn
            blo, bhi = wilson(bw, bn)
            print(f"{100*b:3.0f}-{100*(b+0.05):3.0f}%{'':<5}{bn:>6}{100*avg_p:>9.1f}%{100*act_p:>9.1f}%{100*(act_p - avg_p):>+10.1f}%  [{100*blo:4.1f}, {100*bhi:4.1f}]")
    print("A calibrated model shows actual% rising with avg pred%. Flat/negative err means overconfidence.")

    print(f"\nHOW TO READ THIS")
    print(f"- {shown} slices tested. At a 95% interval, ~1 in 20 slices looks good by pure chance,")
    print("  so a lone 'ABOVE breakeven' slice is a lead to retest on later data, not proof.")
    print("- Trust a slice only if its interval's low end clears breakeven AND both halves agree.")
    print(f"- Plan for 300+ scored samples ({a.spacing}s apart) before acting on anything.")


if __name__ == "__main__":
    main()
