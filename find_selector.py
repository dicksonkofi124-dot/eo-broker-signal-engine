"""
Finds which element on the Expert Option page holds the live price.
Run it with the bot STOPPED (it uses the same saved browser profile):

    python find_selector.py

1. Log in and open the EUR/USD chart. Wait until you can SEE the live price.
2. Press Enter in the terminal.
It looks inside the page, iframes and shadow DOM, takes two readings 2 s apart,
and lists elements whose number changed (= the live price).
If nothing is found it prints what numbers it DID see, so we can work out why.
"""
import asyncio
import os

from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()
PROFILE = os.getenv("USER_DATA_DIR", "./eo_profile")
LO = float(os.getenv("PRICE_MIN", "0.9") or 0.9)
HI = float(os.getenv("PRICE_MAX", "1.4") or 1.4)

JS = """
([lo, hi]) => {
  const found = {}, seen = {};
  let canvases = 0;
  const sel = (el) => {
    if (el.id) return el.tagName.toLowerCase() + '#' + CSS.escape(el.id);
    const cls = [...el.classList].filter(c => !/^\\d/.test(c)).slice(0, 3)
      .map(c => '.' + CSS.escape(c)).join('');
    return el.tagName.toLowerCase() + cls;
  };
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) {
      if (el.tagName === 'CANVAS') canvases++;
      if (el.shadowRoot) walk(el.shadowRoot);
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      const t = (el.textContent || '').replace(/\\s+/g, '');
      if (t.length === 0 || t.length > 24) continue;
      const m = t.match(/^\\D{0,3}(\\d+[.,]\\d{2,6})\\D{0,3}$/);
      if (!m) continue;
      // keep only the smallest element holding this text
      if ([...el.children].some(c => (c.textContent || '').replace(/\\s+/g, '') === t)) continue;
      const key = sel(el) + ' @' + Math.round(r.x) + ',' + Math.round(r.y);
      seen[key] = t;
      const v = parseFloat(m[1].replace(',', '.'));
      if (v >= lo && v <= hi) found[key] = t;
    }
  };
  walk(document);
  return {found, seen, canvases};
}
"""


async def read_all(page):
    found, seen, canv = {}, {}, 0
    for fr in page.frames:
        try:
            r = await fr.evaluate(JS, [LO, HI])
        except Exception:
            continue
        tag = "" if fr == page.main_frame else f" [iframe {fr.url[:60]}]"
        found.update({k + tag: v for k, v in r["found"].items()})
        seen.update({k + tag: v for k, v in r["seen"].items()})
        canv += r["canvases"]
    return found, seen, canv


async def main() -> None:
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            PROFILE, headless=False, viewport={"width": 1400, "height": 900}
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://expertoption.com", timeout=90_000)
        await asyncio.to_thread(
            input, "\nLog in, open EUR/USD, wait until the live price is visible, then press Enter here... "
        )
        a, seen_a, canv = await read_all(page)
        await asyncio.sleep(2)
        b, seen_b, _ = await read_all(page)

        live = {k: (a[k], b[k]) for k in b if k in a and a[k] != b[k]}
        print(f"\nFrames on page: {len(page.frames)}   Canvas elements: {canv}   URL: {page.url[:80]}")
        print("\n=== Elements whose value CHANGED in range (live price) ===")
        for k, (x, y) in live.items():
            print(f"{k}\n    {x}  ->  {y}")
        if not live:
            print("(none)")
            print("\nIn-range numbers seen (static, maybe market closed):")
            for k, v in b.items():
                print(f"  {k}   {v}")
            if not b:
                print("  (none in range)")
                print("\nAny decimal numbers on the page, to see the format:")
                for k, v in list(seen_b.items())[:25]:
                    print(f"  {k}   {v}")
        print("\nSet PRICE_SELECTOR to the part before ' @' (e.g. span.current-price).")
        await ctx.close()


asyncio.run(main())