"""
Deep probe for Expert Option live price.
Looks in:
  - normal DOM + shadow DOM + iframes
  - window / global JS objects
  - common chart library structures
  - performance / memory hints

Usage (bot must be stopped):
    python probe_price.py

1. Log in and open EUR/USD so you can see the live price.
2. Press Enter in the terminal.
"""

import asyncio
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()
PROFILE = os.getenv("USER_DATA_DIR", "./eo_profile")
LO = float(os.getenv("PRICE_MIN", "0.9") or 0.9)
HI = float(os.getenv("PRICE_MAX", "1.4") or 1.4)

DOM_PROBE = """
([lo, hi]) => {
  const out = {dom: [], canvases: 0, shadows: 0};
  const walk = (root, path) => {
    for (const el of root.querySelectorAll('*')) {
      if (el.tagName === 'CANVAS') out.canvases++;
      if (el.shadowRoot) {
        out.shadows++;
        walk(el.shadowRoot, path + ' >> shadow');
      }
      const t = (el.textContent || '').replace(/\\s+/g, ' ').trim();
      if (!t || t.length > 30) continue;
      const m = t.match(/(\\d+[.,]\\d{2,6})/);
      if (!m) continue;
      const v = parseFloat(m[1].replace(',', '.'));
      if (v < lo || v > hi) continue;
      if ([...el.children].some(c => (c.textContent||'').includes(m[1]))) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) continue;
      let sel = el.tagName.toLowerCase();
      if (el.id) sel += '#' + CSS.escape(el.id);
      else if (el.classList.length)
        sel += [...el.classList].slice(0,3).map(c => '.'+CSS.escape(c)).join('');
      out.dom.push({sel, text: t.slice(0,40), x: Math.round(r.x), y: Math.round(r.y), v});
    }
  };
  walk(document, 'main');
  return out;
}
"""

WINDOW_PROBE = """
([lo, hi]) => {
  const hits = [];
  const seen = new Set();
  const add = (path, val) => {
    if (typeof val === 'number' && val >= lo && val <= hi) {
      const k = path + '=' + val;
      if (!seen.has(k)) { seen.add(k); hits.push({path, val}); }
    }
  };
  const walk = (obj, path, depth) => {
    if (depth > 4 || !obj || typeof obj !== 'object') return;
    try {
      for (const key of Object.keys(obj)) {
        if (key.startsWith('_') && depth > 1) continue;
        let val;
        try { val = obj[key]; } catch { continue; }
        const p = path + '.' + key;
        if (typeof val === 'number') add(p, val);
        else if (typeof val === 'string') {
          const m = val.match(/^\\d+[.,]\\d{2,6}$/);
          if (m) add(p, parseFloat(m[0].replace(',','.')));
        } else if (val && typeof val === 'object' && depth < 4) {
          if (/price|rate|quote|last|close|bid|ask|value|current/i.test(key))
            walk(val, p, depth + 1);
        }
      }
    } catch {}
  };

  const candidates = [
    'window', 'window.app', 'window.store', 'window.__STORE__',
    'window.__PRELOADED_STATE__', 'window.chart', 'window.tvWidget',
    'window.TradingView', 'window.EO', 'window.eo', 'window.ExpertOption',
    'window.quote', 'window.quotes', 'window.ticker', 'window.symbol',
    'window.currentAsset', 'window.activeAsset', 'window.price', 'window.lastPrice',
  ];
  for (const c of candidates) {
    try {
      const obj = eval(c);
      if (obj) walk(obj, c, 0);
    } catch {}
  }
  for (const key of Object.keys(window)) {
    if (/price|quote|chart|store|asset|ticker|trade|rate/i.test(key)) {
      try { walk(window[key], 'window.' + key, 0); } catch {}
    }
  }
  return hits.slice(0, 40);
}
"""

NETWORK_HINT = """
() => {
  return performance.getEntriesByType('resource')
    .filter(e => /price|quote|tick|ws|socket|stream|asset|candle/i.test(e.name))
    .map(e => e.name)
    .slice(0, 20);
}
"""


async def probe_frame(frame, lo, hi):
    results = {"url": frame.url[:80], "dom": [], "js": [], "canvases": 0, "shadows": 0}
    try:
        dom = await frame.evaluate(DOM_PROBE, [lo, hi])
        results["dom"] = dom.get("dom", [])
        results["canvases"] = dom.get("canvases", 0)
        results["shadows"] = dom.get("shadows", 0)
    except Exception as e:
        results["dom_error"] = str(e)[:120]
    try:
        js = await frame.evaluate(WINDOW_PROBE, [lo, hi])
        results["js"] = js
    except Exception as e:
        results["js_error"] = str(e)[:120]
    try:
        results["network_hints"] = await frame.evaluate(NETWORK_HINT)
    except Exception:
        results["network_hints"] = []
    return results


async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE,
            headless=False,
            viewport={"width": 1400, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        if "expertoption" not in (page.url or ""):
            await page.goto("https://app.expertoption.com", timeout=90_000)

        print("\\n>>> Log in (if needed) and open the EUR/USD chart so the live price is visible.")
        print(">>> Then press Enter here...\\n")
        await asyncio.to_thread(input)

        print("Probing all frames...\\n")
        all_dom = []
        all_js = []
        total_canvas = 0

        for frame in page.frames:
            r = await probe_frame(frame, LO, HI)
            total_canvas += r.get("canvases", 0)
            if r.get("dom"):
                print(f"[DOM] frame: {r['url']}")
                for d in r["dom"][:15]:
                    print(f"  {d['sel']} @({d['x']},{d['y']})  →  {d['text']}")
                    all_dom.append(d)
            if r.get("js"):
                print(f"[JS ] frame: {r['url']}")
                for h in r["js"][:15]:
                    print(f"  {h['path']}  =  {h['val']}")
                    all_js.append(h)
            if r.get("network_hints"):
                print(f"[NET] {r['network_hints'][:5]}")

        print("\\n========== SUMMARY ==========")
        print(f"Total <canvas> elements found: {total_canvas}")
        print(f"DOM candidates in price range: {len(all_dom)}")
        print(f"JS object candidates in price range: {len(all_js)}")

        if all_js:
            print("\\nMost promising JS paths:")
            for h in all_js[:10]:
                print(f"  {h['path']} = {h['val']}")
            print("\\nIf one of these looks like the live price, we can read it with page.evaluate().")

        if all_dom:
            print("\\nMost promising DOM selectors:")
            for d in all_dom[:8]:
                print(f"  {d['sel']}")
            print("\\nPut the best one in .env as PRICE_SELECTOR=...")

        if total_canvas > 0 and not all_dom and not all_js:
            print("\\nPrice is almost certainly drawn on a canvas.")
            print("Next options: intercept WebSocket/network ticks, or use an external price API.")

        if not all_dom and not all_js:
            print("\\nNo price-like numbers found in DOM or common JS objects.")
            print("Dumping a few global keys for manual inspection...")
            try:
                keys = await page.evaluate("""() => Object.keys(window).filter(k =>
                  /price|quote|chart|store|asset|ticker|trade|rate|eo|expert/i.test(k)
                ).slice(0, 40)""")
                print("Interesting window keys:", keys)
            except Exception as e:
                print("Could not list keys:", e)

        print("\\nDone. Chromium will stay open while this script is running.")
        await asyncio.to_thread(input, ">>> Press Enter here when you're ready to close Chromium...")
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
