"""
Step 2 probe: finds the EUR/USD asset id and the layout of candle updates.
Read-only. Prints ONLY market-data fields (asset names/ids, candle fields).
Other traders' messages (trades/options) are ignored and never printed.

    python ws_probe_candles.py

Log in, open EUR/USD, wait until the chart is live, press Enter, wait ~20 s.
"""
import asyncio
import os
import re
import time
from collections import defaultdict

from dotenv import load_dotenv
from playwright.async_api import async_playwright

from ws_probe import decode, PROFILE, LO, HI, CHANNEL, BARGS

load_dotenv()
EUR_RE = re.compile(r"EUR.{0,3}USD", re.I)


def scalar_fields(d: dict) -> dict:
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 30)}


def find_eurusd_assets(assets: list) -> list:
    out = []
    for a in assets:
        if not isinstance(a, dict):
            continue
        sc = scalar_fields(a)
        if any(isinstance(v, str) and EUR_RE.search(v) for v in sc.values()):
            out.append(sc)
    return out


def summarize_candles(msgs, lo=LO, hi=HI):
    """msgs: list of 'candles' message dicts (already the inner 'message')."""
    by_asset = defaultdict(list)
    for m in msgs:
        aid = m.get("assetId")
        for c in m.get("candles") or []:
            v = c.get("v")
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) \
                    and all(lo <= x <= hi for x in v):
                by_asset[aid].append(c)
    lines = []
    for aid, cs in by_asset.items():
        lines.append(f"\nassetId={aid}: {len(cs)} in-range candle updates")
        lines.append(f"  candle keys: {sorted(cs[0].keys())}")
        lines.append(f"  v length: {sorted({len(c['v']) for c in cs})}   tf values: {sorted({c.get('tf') for c in cs}, key=str)}")
        for c in cs[-6:]:
            lines.append("  " + str({k: c[k] for k in c if k in ('tf', 't', 'tt', 'v')}))
    if not lines:
        lines.append("(no in-range candle updates seen)")
    return lines


async def main() -> None:
    assets_msgs: list = []
    candles: list = []
    state = {"rec": False}

    def attach(page):
        def on_ws(ws):
            def rx(payload):
                kind, obj = decode(payload)
                if not (kind.endswith("json") and isinstance(obj, dict)):
                    return
                act = obj.get("action")
                msg = obj.get("message")
                if act == "assets" and isinstance(msg, dict):
                    assets_msgs.append(msg.get("assets") or [])
                elif act == "candles" and isinstance(msg, dict) and state["rec"]:
                    candles.append(msg)
            ws.on("framereceived", rx)
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
        state["rec"] = True
        print("Listening for 20 seconds, please wait...")
        await asyncio.sleep(20)

        print(f"\n'assets' messages seen: {len(assets_msgs)}")
        if assets_msgs:
            allassets = assets_msgs[-1]
            print(f"assets in last message: {len(allassets)}")
            if allassets and isinstance(allassets[0], dict):
                print("asset keys:", sorted(k for k in allassets[0].keys()))
            matches = find_eurusd_assets(allassets)
            print(f"\nEUR/USD-like assets ({len(matches)}):")
            for m in matches[:6]:
                print("  ", m)
        else:
            print("(none: the asset list arrives at page load. Re-run and press Enter only AFTER the page fully loads, "
                  "or reload the page once while this is running)")
        print("\n=== Candle updates with EUR/USD-range values ===")
        print("\n".join(summarize_candles(candles)))
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
