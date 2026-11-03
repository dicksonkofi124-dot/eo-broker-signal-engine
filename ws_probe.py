"""
Works out WHERE and HOW the live price appears in the data the page's own
websocket RECEIVES. Read-only: it sends nothing and clicks nothing.

    python ws_probe.py

1. Log in, open EUR/USD, wait until the chart is live, press Enter.
2. It listens 15 s and prints a structural summary (shapes, field names,
   which fields look like a scaled EUR/USD price). Digits and long strings
   are masked in the shapes, and outgoing messages are never printed.
Optional: pip install msgpack   (lets it try that binary format too)
"""
import asyncio
import json
import os
import re
import statistics
import time
import zlib
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from dotenv import load_dotenv
from playwright.async_api import async_playwright

try:
    import msgpack  # optional
except Exception:  # pragma: no cover
    msgpack = None

load_dotenv()
PROFILE = os.getenv("USER_DATA_DIR", "./eo_profile")
LO = float(os.getenv("PRICE_MIN", "0.9") or 0.9)
HI = float(os.getenv("PRICE_MAX", "1.4") or 1.4)
CHANNEL = os.getenv("BROWSER_CHANNEL", "").strip()
BARGS = os.getenv("BROWSER_ARGS", "").split()
SCALES = [10 ** k for k in range(0, 9)]


def skeleton(s: str) -> str:
    s = re.sub(r'"[^"]{12,}"', '"S"', s)
    s = re.sub(r"\d+(?:\.\d+)?", "9", s)
    s = re.sub(r"(?:9,){3,}9", "9,...", s)
    return s[:110]


def try_decompress(b: bytes):
    for wbits, name in ((15, "zlib"), (-15, "deflate"), (31, "gzip")):
        try:
            return zlib.decompress(b, wbits), name
        except Exception:
            continue
    return None, None


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk(v, f"{path}[]")
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, str):
        try:
            yield path + "(str)", float(obj)
        except ValueError:
            return


def decode(payload):
    """-> (kind, obj_or_text). kind: json | text | msgpack | binary"""
    note = ""
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        dec, how = try_decompress(raw)
        if dec is not None:
            raw, note = dec, how + "+"
        if msgpack is not None:
            try:
                return note + "msgpack", msgpack.unpackb(raw, raw=False, strict_map_key=False)
            except Exception:
                pass
        try:
            payload = raw.decode("utf-8")
        except UnicodeDecodeError:
            return note + "binary", raw
    else:
        note = ""
    s = payload
    m = re.match(r"^\d{1,3}(?=[\[{])", s)  # socket.io style prefix like 42[...
    body = s[m.end():] if m else s
    try:
        return note + "json", json.loads(body)
    except Exception:
        return note + "text", s


def inspect(frames, lo=LO, hi=HI):
    lines = []
    kinds = Counter()
    sizes = []
    skels = Counter()
    hits = defaultdict(lambda: {"n": 0, "vals": []})
    bin_heads = Counter()
    for payload in frames:
        sizes.append(len(payload))
        kind, obj = decode(payload)
        kinds[kind] += 1
        if kind.endswith("binary"):
            bin_heads[obj[:2].hex()] += 1
            continue
        text = obj if kind.endswith("text") else json.dumps(obj, default=str)
        skels[skeleton(text)] += 1
        if kind.endswith("text"):
            for m in re.finditer(r"\d+(?:\.\d+)?", obj):
                v = float(m.group(0))
                for sc in SCALES:
                    if lo <= v / sc <= hi:
                        h = hits[("(raw text)", sc)]
                        h["n"] += 1
                        if v not in h["vals"] and len(h["vals"]) < 5:
                            h["vals"].append(v)
        else:
            for path, v in walk(obj):
                for sc in SCALES:
                    if lo <= v / sc <= hi:
                        h = hits[(path or "(root)", sc)]
                        h["n"] += 1
                        if v not in h["vals"] and len(h["vals"]) < 5:
                            h["vals"].append(v)
    lines.append(f"Frames: {len(frames)}   formats: {dict(kinds)}")
    if sizes:
        lines.append(f"Size bytes: min={min(sizes)} median={int(statistics.median(sizes))} max={max(sizes)}")
    lines.append("\nMost common message shapes (digits masked):")
    for sk, n in skels.most_common(6):
        lines.append(f"  x{n}: {sk}")
    if bin_heads:
        lines.append("\nUndecoded binary: first 2 bytes (hex) frequency: " + str(bin_heads.most_common(5)))
    lines.append("\nFields that look like an EUR/USD price (value / scale in range):")
    ranked = sorted(hits.items(), key=lambda kv: -kv[1]["n"])[:8]
    if not ranked:
        lines.append("  (none)")
    for (path, sc), h in ranked:
        lines.append(f"  {path}  scale=1/{sc}  hits={h['n']}  sample values={h['vals']}")
    return lines


async def main() -> None:
    frames = []  # (ts, payload) received only
    counts = defaultdict(lambda: [0, 0])

    def attach(page):
        def on_ws(ws):
            u = urlsplit(ws.url)
            label = f"{u.scheme}://{u.netloc}{u.path}"[:90]
            ws.on("framereceived", lambda p: (frames.append((time.time(), p)), counts[label].__setitem__(0, counts[label][0] + 1)))
            ws.on("framesent", lambda p: counts[label].__setitem__(1, counts[label][1] + 1))
        page.on("websocket", on_ws)

    async with async_playwright() as p:
        kw = {"headless": False, "viewport": {"width": 1400, "height": 900}, "timeout": 60_000}
        if BARGS:
            kw["args"] = BARGS
        if CHANNEL:
            kw["channel"] = CHANNEL
        ctx = await p.chromium.launch_persistent_context(PROFILE, **kw)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        attach(page)
        ctx.on("page", attach)
        await page.goto("https://expertoption.com", timeout=90_000)
        await asyncio.to_thread(input, "\nLog in, open EUR/USD, wait until the chart is live, then press Enter... ")
        t0 = time.time()
        print("Listening for 15 seconds, please wait...")
        await asyncio.sleep(15)
        recent = [p_ for ts, p_ in frames if ts >= t0]
        print(f"\nWebsocket connections: {dict((k, tuple(v)) for k, v in counts.items())}  (received, sent)")
        print(f"msgpack available: {msgpack is not None}\n")
        print("\n".join(inspect(recent)))
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
