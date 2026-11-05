"""
EO Broker Live Price Scraper + Adaptive Signal Engine  (v2 – reviewed)
======================================================================
Scrapes live prices from Expert Option using a headful Playwright browser
(persistent profile, so your login survives restarts) and feeds them into
adaptive online logistic-regression models that generate PAPER-trading signals.

IMPORTANT: Automated access to Expert Option likely violates their Terms of
Service. Personal paper-trading research only, at your own risk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from ensemble_models import DualModelEnsemble, EnsemblePrediction
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from playwright.async_api import async_playwright, BrowserContext, Page, Playwright
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
USER_DATA_DIR = os.getenv("USER_DATA_DIR", "./eo_profile")
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "").strip()  # "chrome" = use installed Google Chrome
BROWSER_PATH = os.getenv("BROWSER_PATH", "").strip()        # or full path to a browser executable
BROWSER_ARGS = os.getenv("BROWSER_ARGS", "").split()          # e.g. --disable-gpu
def _viewport() -> dict:
    m = re.match(r"^(\d+)x(\d+)$", os.getenv("BROWSER_VIEWPORT", "1180x560").strip())
    return {"width": int(m.group(1)), "height": int(m.group(2))} if m else {"width": 1180, "height": 560}


PRICE_SOURCE = os.getenv("PRICE_SOURCE", "ws").strip().lower()  # ws = page's websocket ticks, dom = page text
SIGNAL_TEST_MODE = os.getenv("SIGNAL_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")
TEST_INTERVAL_SECONDS = max(30, int(os.getenv("TEST_INTERVAL_SECONDS", "120") or 120))
_ASSET_RAW = os.getenv("ASSET_ID", "").strip()
ASSET_AUTO = _ASSET_RAW.lower() == "auto"       # pick the single asset ticking in the price range
ASSET_ID = int(_ASSET_RAW) if _ASSET_RAW.isdigit() else 0                  # EUR/USD asset id in the websocket feed
PRICE_SELECTOR = os.getenv("PRICE_SELECTOR", "").strip()  # exact CSS selector (from find_selector.py)
PRICE_MIN = float(os.getenv("PRICE_MIN", "0") or 0)       # sanity range, e.g. 0.9 for EUR/USD
PRICE_MAX = float(os.getenv("PRICE_MAX", "0") or 0)             # e.g. 1.4 ; 0 = no range check

SIGNAL_GATE = os.getenv("SIGNAL_GATE", "relaxed").lower()  # off|relaxed|strict
EDGE_MODE = os.getenv("EDGE_MODE", "adaptive").lower()


def _float_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


SIGNAL_TOP_PERCENT = _float_env("SIGNAL_TOP_PERCENT", 3.0, 0.5, 50.0)
PAYOUT = _float_env("PAYOUT", 0.85, 0.5, 1.0)  # broker payout on a win
BREAKEVEN = 1.0 / (1.0 + PAYOUT)  # win rate needed to break even

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 20
OBSERVATION_SECONDS = 12
TRADE_EXPIRATION_SECONDS = 30
RESOLUTION_POLL_INTERVAL = 3.0
VOID_AFTER_SECONDS = 120  # trade older than this with no exit price -> VOID
LEARNING_HISTORY_START_AT = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
LEARNING_HISTORY_LIMIT = 200

LEARNING_SAMPLE_SECONDS = 5
MODEL_STATE_ID = "eobroker_live_v2"  # new id: feature set changed (5 -> 9)
MODEL_SAVE_SECONDS = 300

LR_START = 0.02
LR_FLOOR = 0.002
LR_DECAY_SAMPLES = 300
L2_STRENGTH = 0.01
GRAD_CLIP = 3.0
STATS_MIN_ALPHA = 0.002  # stats forget old regimes (~500-sample memory)

# (name, horizon seconds, feature window seconds, min relative move to learn)
HORIZONS = [
    ("30s", 30, None, 0.00002),
    ("1m", 60, 60, 0.00003),
    ("5m", 300, 300, 0.00005),
    ("15m", 900, 900, 0.00008),
    ("1h", 3600, 1800, 0.00015),
]
HISTORY_SECONDS = 1800
STORE_EVERY_N = 3
SAMPLE_FLUSH_SECONDS = 60
SAMPLE_BUFFER_MAX = 3000
MIN_TRAIN_SAMPLES = 300
MIN_EDGE = 0.08
MEDIUM_EDGE = 0.10
HIGH_EDGE = 0.15
SIGNAL_COOLDOWN_SECONDS = 55

# Browser recovery
BROWSER_HEALTH_CHECK_INTERVAL = 30
PRICE_STALE_SECONDS = 20
MAX_RELOADS_BEFORE_RELAUNCH = 2
MAX_BROWSER_RESTARTS = 10
MAX_PRICE_JUMP = 0.05  # reject >5% tick-to-tick jumps as mis-scrapes
JUMP_ACCEPT_AFTER = 8  # ...unless it persists (asset switched)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
for noisy in ("httpx", "httpcore", "playwright"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("EO_Scraping_Engine")


async def db(fn):
    """Run a blocking supabase call off the event loop."""
    return await asyncio.to_thread(fn)


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------
def _safe_err(exc: Exception) -> str:
    """Error text that never contains the bot token (httpx puts URLs in errors)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    msg, code = getattr(exc, "message", None), getattr(exc, "code", None)  # Supabase APIError
    if msg or code:
        return f"{type(exc).__name__} code={code} message={msg}"
    return type(exc).__name__


def format_signal(direction: str, price: float, score: float, expiry: int,
                  confidence: Optional[str] = None, gate: Optional[str] = None,
                  test: bool = False) -> str:
    """Telegram message: green block for BUY, red block for SELL (Telegram cannot colour text)."""
    buy = direction == "BUY"
    dot = "🟢" if buy else "🔴"
    arrow = "▲" if buy else "▼"
    bar = dot * 8
    lines = []
    if test:
        lines += ["🧪 <b>TEST ONLY, DO NOT TRADE</b>", ""]
    lines += [bar, f"<b>{direction} {arrow}  EUR/USD</b>", bar, ""]
    lines += [f"💰 Price: <code>{price:.5f}</code>", f"⏱ Expiry: {expiry}s"]
    if confidence:
        lines.append(f"🎯 Confidence: {confidence}")
    lines.append(f"📊 Score: <code>{score:+.4f}</code>")
    lines.append(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}")
    if gate:
        lines.append(f"🚦 Gate: {gate}")
    lines.append("")
    lines.append("<i>Pipeline check only: the model has shown no edge.</i>" if test
                 else "<i>Paper signal, not financial advice.</i>")
    return "\n".join(lines)


async def broadcast_telegram(
    message: str, db_client: Client, subscribers: set[str] | None = None
) -> None:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("YOUR_"):
        logger.warning("Telegram token not configured – skipping broadcast.")
        return

    targets: set[str] = set()
    if TELEGRAM_CHAT_ID:
        targets.add(TELEGRAM_CHAT_ID)
    if subscribers:
        targets.update(subscribers)

    try:
        response = await db(lambda: db_client.table("bot_users").select("chat_id").execute())
        for user in response.data or []:
            targets.add(str(user["chat_id"]))
    except Exception as exc:
        logger.error("Error fetching subscriber list: %s", _safe_err(exc))

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=12) as client:
        for chat_id in targets:
            try:
                resp = await client.post(
                    url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
                )
                resp.raise_for_status()
            except Exception as exc:
                logger.error("Telegram send to %s failed: %s", chat_id, _safe_err(exc))


# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?![\d,])(?:\.\d+)?|\d+(?:[.,]\d+)?")


def parse_price(text: str) -> Optional[float]:
    """First number in text. Handles '1,234.56', '1.08543', '1,08543'.
    Ignores trailing things like '+0.2%' (old code glued them together)."""
    m = _NUM_RE.search(text or "")
    if not m:
        return None
    s = m.group(0)
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        # '1,234' / '12,345,678' = thousands separators; '1,08543' = decimal comma
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s) else s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) and v > 0 else None


# ---------------------------------------------------------------------------
# Playwright price feed with recovery
# ---------------------------------------------------------------------------
class PlaywrightScraperFeed:
    def __init__(self) -> None:
        self.playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        self.price_buffer: deque[float] = deque(maxlen=SEQUENCE_LENGTH)
        self.history: deque[tuple[float, float]] = deque(maxlen=HISTORY_SECONDS + 60)
        self.latest_price: Optional[float] = None
        self.last_update_at: Optional[datetime] = None
        self._last_sample_ts: Optional[float] = None
        self._pending_jump: list[float] = []
        self._candidates: dict = {}
        self._last_frame_at: Optional[float] = None
        self._frames_seen = 0
        self._auto_id = None
        self._seen_ticks: dict = {}
        self._cand_logged_at = 0.0

        self.is_running = True
        self._restart_count = 0
        self._reload_count = 0
        self._ever_had_price = False
        self._waiting_logged_at = datetime.now(timezone.utc)
        self._tasks: list[asyncio.Task] = []

        # Best-effort selectors – inspect the live page (F12) and put the
        # most specific one FIRST. Broad ones like div[class*='price'] can
        # match balance/payout widgets; the jump filter below is a safety net.
        self.price_selectors = ([PRICE_SELECTOR] if PRICE_SELECTOR else []) + [
            ".current-price",
            ".price-value",
            "div[class*='currentPrice']",
            "span[class*='deal-price']",
            "[data-testid*='price']",
            ".chart-price",
            ".asset-price",
            "div[class*='price']",
        ]

    async def start(self) -> None:
        logger.info("Starting Playwright scraper (headful, persistent profile)...")
        await self._launch_browser()
        self._tasks = [
            *([asyncio.create_task(self._extraction_loop(), name="extract")] if PRICE_SOURCE == "dom" else []),
            asyncio.create_task(self._sampler_loop(), name="sampler"),
            asyncio.create_task(self._health_monitor(), name="health"),
        ]

    async def _launch_browser(self) -> None:
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
        if self.playwright is None:
            self.playwright = await async_playwright().start()

        # Persistent profile => login cookies survive relaunches.
        kwargs: dict = {"headless": False, "viewport": _viewport(), "timeout": 60_000}
        kwargs["args"] = ["--window-position=0,0"] + BROWSER_ARGS  # keep the window on screen
        if BROWSER_PATH:
            kwargs["executable_path"] = BROWSER_PATH
        elif BROWSER_CHANNEL:
            kwargs["channel"] = BROWSER_CHANNEL
        self.context = await self.playwright.chromium.launch_persistent_context(USER_DATA_DIR, **kwargs)
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        if PRICE_SOURCE == "ws":
            self.page.on("websocket", self._on_ws)
            self.context.on("page", lambda pg: pg.on("websocket", self._on_ws))
        logger.info("Browser ready. Log in manually if needed, then open the EUR/USD chart.")
        await self.page.goto("https://expertoption.com", timeout=90_000)

    # ---- websocket price source (read-only: only incoming market ticks are used) ----
    def _on_ws(self, ws) -> None:
        ws.on("framereceived", self._on_frame)

    def _on_frame(self, payload) -> None:
        self._last_frame_at = datetime.now(timezone.utc).timestamp()
        self._frames_seen += 1
        try:
            if isinstance(payload, (bytes, bytearray)):
                payload = bytes(payload).decode("utf-8")
            if '"candles"' not in payload:  # ignore everything except market candles
                return
            obj = json.loads(payload)
            if obj.get("action") != "candles":
                return
            msg = obj.get("message")
            if not isinstance(msg, dict):
                return
            aid = msg.get("assetId")
            ticks: list[float] = []
            for c in msg.get("candles") or []:
                v = c.get("v")
                if c.get("tf") == 0 and isinstance(v, list) and v and isinstance(v[0], (int, float)):
                    ticks.append(float(v[0]))  # tf=0 is a live tick: v = [price]
            if not ticks:
                return
            if (PRICE_MAX <= 0 or PRICE_MIN <= ticks[-1] <= PRICE_MAX) and len(self._seen_ticks) < 20:
                self._seen_ticks[aid] = ticks[-1]
            target = ASSET_ID or self._auto_id
            if target is None:
                if PRICE_MAX > 0 and PRICE_MIN <= ticks[-1] <= PRICE_MAX:
                    self._candidates[aid] = ticks[-1]
                if ASSET_AUTO and len(self._candidates) == 1:
                    self._auto_id = next(iter(self._candidates))
                    logger.info("ASSET_ID=auto: using asset id %s (the only asset ticking in your price range). "
                                "Check that its price matches your EUR/USD chart.", self._auto_id)
                    target = self._auto_id
                else:
                    now = datetime.now(timezone.utc).timestamp()
                    if self._candidates and now - self._cand_logged_at > 20:
                        self._cand_logged_at = now
                        logger.warning("ASSET_ID is not set. Assets with prices in range: %s. "
                                       "Put ASSET_ID=<id> (or auto) in .env.", self._candidates)
                    if target is None:
                        return
            if aid != target:
                return
            for tick in ticks:
                self._accept_price(tick)
        except Exception as exc:
            logger.debug("ws frame ignored: %s", exc)

    async def _extraction_loop(self) -> None:
        while self.is_running:
            try:
                if self.page is None or self.page.is_closed():
                    await asyncio.sleep(1)
                    continue

                price_text: Optional[str] = None
                for selector in self.price_selectors:
                    try:
                        locator = self.page.locator(selector).first
                        if await locator.count() > 0 and await locator.is_visible(timeout=300):
                            text = await locator.inner_text(timeout=500)
                            if text and any(c.isdigit() for c in text):
                                price_text = text
                                break
                    except Exception:
                        continue

                val = parse_price(price_text) if price_text else None
                if val is not None:
                    self._accept_price(val)
            except Exception as exc:
                logger.debug("Extraction tick error (non-fatal): %s", exc)
            await asyncio.sleep(0.25)

    def _accept_price(self, val: float) -> None:
        if PRICE_MAX > 0 and not (PRICE_MIN <= val <= PRICE_MAX):
            return  # wrong element (balance, payout, %, ...) – ignore
        prev = self.latest_price
        if prev is not None and abs(val / prev - 1) > MAX_PRICE_JUMP:
            self._pending_jump.append(val)
            if len(self._pending_jump) < JUMP_ACCEPT_AFTER:
                return  # probably a mis-scrape
            logger.warning("Price level changed %.5f -> %.5f persistently; resetting buffers.", prev, val)
            self.price_buffer.clear()
            self.history.clear()
        self._pending_jump.clear()
        if not self._ever_had_price:
            logger.info("First price received: %.5f (compare with your chart)", val)
        self.latest_price = val
        self.last_update_at = datetime.now(timezone.utc)
        self._ever_had_price = True
        self._reload_count = 0
        self._restart_count = 0

    async def _sampler_loop(self) -> None:
        while self.is_running:
            await asyncio.sleep(1)
            if self.latest_price is None or not self.has_fresh_price():
                continue
            ts = datetime.now(timezone.utc).timestamp()
            # A gap means old samples are not contiguous with new ones.
            if self._last_sample_ts is not None and ts - self._last_sample_ts > 3.5:
                self.price_buffer.clear()
            self._last_sample_ts = ts
            self.price_buffer.append(self.latest_price)
            self.history.append((ts, self.latest_price))

    async def _health_monitor(self) -> None:
        while self.is_running:
            await asyncio.sleep(BROWSER_HEALTH_CHECK_INTERVAL)
            now = datetime.now(timezone.utc)

            if not self._ever_had_price:
                # Still waiting for the user to log in – never restart here.
                if (now - self._waiting_logged_at).total_seconds() >= 60:
                    fa = (now.timestamp() - self._last_frame_at) if self._last_frame_at else None
                    logger.info(
                        "Waiting for EUR/USD ticks. Site data frames received: %s (last %s). "
                        "Assets ticking in range: %s. ASSET_ID=%s",
                        self._frames_seen, "never" if fa is None else f"{fa:.0f}s ago",
                        self._seen_ticks or "none", ASSET_ID or "unset")
                    if self._frames_seen == 0:
                        logger.info("-> No data from the site at all: is the page loaded and logged in?")
                    elif ASSET_ID and self._seen_ticks and ASSET_ID not in self._seen_ticks:
                        logger.warning("-> Ticks exist for other asset ids (see list above). Set ASSET_ID=auto, or the id matching your chart.")
                    elif not self._seen_ticks:
                        logger.info("-> Site is connected but no EUR/USD ticks yet: open the EUR/USD chart.")
                    self._waiting_logged_at = now
                continue

            age = (now - self.last_update_at).total_seconds() if self.last_update_at else 1e9
            if age <= PRICE_STALE_SECONDS:
                continue

            frame_age = (now.timestamp() - self._last_frame_at) if self._last_frame_at else 1e9
            if PRICE_SOURCE == "ws" and frame_age < 15:
                logger.warning(
                    "No EUR/USD ticks for %.0fs, but the site's data IS flowing. Is the EUR/USD chart open "
                    "and the browser window visible (not minimised/hidden)? Not reloading.", age)
                continue
            logger.warning("Price stale for %.0fs (last site data %.0fs ago).", age, min(frame_age, 9999))
            try:
                if self.page is None or self.page.is_closed() or self._reload_count >= MAX_RELOADS_BEFORE_RELAUNCH:
                    if self._restart_count >= MAX_BROWSER_RESTARTS:
                        logger.error("Max browser restarts reached. Stopping scraper.")
                        self.is_running = False
                        break
                    self._restart_count += 1
                    self._reload_count = 0
                    logger.warning("Relaunching browser (restart #%s)", self._restart_count)
                    await self._launch_browser()
                else:
                    self._reload_count += 1
                    logger.warning("Reloading page (attempt %s)", self._reload_count)
                    await self.page.reload(timeout=60_000)
            except Exception as exc:
                logger.error("Browser recovery failed: %s", exc)
                await asyncio.sleep(10)

    # Public helpers
    def get_latest_prices(self, n: int) -> list[float]:
        return list(self.price_buffer)[-n:]

    def get_current_price(self) -> Optional[float]:
        return self.price_buffer[-1] if self.price_buffer else None

    def has_full_price_window(self) -> bool:
        return len(self.price_buffer) == SEQUENCE_LENGTH

    def has_fresh_price(self, max_age_seconds: float = 5.0) -> bool:
        if self.last_update_at is None:
            return False
        return 0 <= (datetime.now(timezone.utc) - self.last_update_at).total_seconds() <= max_age_seconds

    def price_at(self, ts: float, tolerance: float = 3.0) -> Optional[float]:
        """Sampled price closest to unix time ts (None if no sample within tolerance)."""
        best: Optional[tuple[float, float]] = None
        for t, p in reversed(self.history):
            d = abs(t - ts)
            if best is None or d < best[0]:
                best = (d, p)
            if t < ts - tolerance:
                break
        return best[1] if best and best[0] <= tolerance else None

    def newest_ts(self) -> Optional[float]:
        return self.history[-1][0] if self.history else None

    def get_window(self, seconds: float, points: int = SEQUENCE_LENGTH) -> Optional[list[float]]:
        if not self.history:
            return None
        newest_ts = self.history[-1][0]
        recent = [(ts, p) for ts, p in self.history if newest_ts - ts <= seconds]
        if len(recent) < 0.8 * seconds or newest_ts - recent[0][0] < 0.9 * seconds:
            return None
        last = len(recent) - 1
        return [recent[round(i * last / (points - 1))][1] for i in range(points)]

    async def stop(self) -> None:
        self.is_running = False
        for t in self._tasks:
            t.cancel()
        try:
            if self.context:
                await self.context.close()
        except Exception:
            pass
        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception:
            pass
        logger.info("Playwright scraper stopped.")


# ---------------------------------------------------------------------------
# Adaptive ML components
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "trend", "mom5", "mom10", "last_move", "zscore",
    "vol_bps", "vol_ratio", "range_pos", "up_down_bal",
]
N_FEATURES = len(FEATURE_NAMES)


class RunningStats:
    """Welford-style mean/variance that gradually forgets (exponential floor on alpha)."""

    def __init__(self, size: int) -> None:
        self.n = 0
        self.mean = [0.0] * size
        self.var = [0.0] * size

    def update(self, x: list[float]) -> None:
        self.n += 1
        a = max(1.0 / self.n, STATS_MIN_ALPHA)
        for i, v in enumerate(x):
            d = v - self.mean[i]
            self.mean[i] += a * d
            self.var[i] = (1 - a) * (self.var[i] + a * d * d)

    def standardize(self, x: list[float]) -> list[float]:
        if self.n < 30:
            return [0.0] * len(x)
        out = []
        for i, v in enumerate(x):
            std = math.sqrt(self.var[i]) if self.var[i] > 0 else 0.0
            out.append(0.0 if std < 1e-9 else max(-5.0, min(5.0, (v - self.mean[i]) / std)))
        return out


class OnlineLogisticModel:
    def __init__(self, n_features: int = N_FEATURES, l2: float = L2_STRENGTH) -> None:
        self.w = [0.0] * n_features
        self.b = 0.0
        self.l2 = l2
        self.stats = RunningStats(n_features)
        self.n_updates = 0

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, features: list[float]) -> float:
        x = self.stats.standardize(features)
        return self._sigmoid(self.b + sum(w * v for w, v in zip(self.w, x)))

    def update(self, features: list[float], label: float) -> None:
        self.stats.update(features)
        x = self.stats.standardize(features)
        p = self._sigmoid(self.b + sum(w * v for w, v in zip(self.w, x)))
        err = max(-GRAD_CLIP, min(GRAD_CLIP, p - label))
        lr = max(LR_FLOOR, LR_START / (1 + self.n_updates / LR_DECAY_SAMPLES))
        for i in range(len(self.w)):
            self.w[i] -= lr * (err * x[i] + self.l2 * self.w[i])
        self.b -= lr * err  # bias is not regularised
        self.n_updates += 1

    def to_dict(self) -> dict:
        return {
            "w": self.w, "b": self.b, "n_updates": self.n_updates,
            "stats_n": self.stats.n, "stats_mean": self.stats.mean, "stats_var": self.stats.var,
        }

    def load_dict(self, d: dict) -> None:
        n = len(self.w)
        w, mean, var = d["w"], d["stats_mean"], d["stats_var"]
        if not (len(w) == len(mean) == len(var) == n):
            raise ValueError(f"saved model has {len(w)} features, expected {n}")
        self.w = [float(v) for v in w]
        self.b = float(d["b"])
        self.n_updates = int(d["n_updates"])
        self.stats.n = int(d["stats_n"])
        self.stats.mean = [float(v) for v in mean]
        self.stats.var = [float(v) for v in var]


def extract_features(prices: list[float]) -> Optional[list[float]]:
    n = len(prices)
    if n < 10 or min(prices) <= 0:
        return None
    returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, n)]
    mean_r = sum(returns) / len(returns)
    vol = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
    if vol == 0:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(prices) / n
    sxx = sum((i - x_mean) ** 2 for i in range(n))
    slope = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n)) / sxx
    price_std = math.sqrt(sum((p - y_mean) ** 2 for p in prices) / n)

    trend = slope / (vol * prices[-1])
    mom5 = sum(returns[-5:]) / (vol * math.sqrt(5))
    mom10 = sum(returns[-10:]) / (vol * math.sqrt(min(10, len(returns))))
    last_move = returns[-1] / vol
    zscore = (prices[-1] - y_mean) / (price_std + 1e-12)
    vol_bps = min(vol * 1e4, 50.0)
    rms_all = math.sqrt(sum(r * r for r in returns) / len(returns))
    rms_recent = math.sqrt(sum(r * r for r in returns[-5:]) / 5)
    vol_ratio = min(rms_recent / rms_all, 5.0) if rms_all > 0 else 1.0
    hi, lo = max(prices), min(prices)
    range_pos = 0.0 if hi == lo else 2 * (prices[-1] - lo) / (hi - lo) - 1
    up = sum(r for r in returns if r > 0)
    dn = -sum(r for r in returns if r < 0)
    balance = (up - dn) / (up + dn) if (up + dn) > 0 else 0.0

    feats = [trend, mom5, mom10, last_move, zscore, vol_bps, vol_ratio, range_pos, balance]
    return [max(-10.0, min(10.0, f)) for f in feats]


class JarvisLearningModule:
    def __init__(self, memory_size: int = 200) -> None:
        self.outcome_memory: deque[tuple[float, bool]] = deque(maxlen=memory_size)
        self.total_wins = 0
        self.total_trades = 0
        self.model = OnlineLogisticModel()
        self.last_edge: Optional[float] = None
        self.recent_abs_edges: deque[float] = deque(maxlen=1500)
        self.shadow_memory: deque[bool] = deque(maxlen=60)
        self.last_shadow_at: Optional[datetime] = None

    extract_features = staticmethod(extract_features)

    def evaluate_sequence(self, prices: list[float]) -> tuple[int, float, int, str]:
        features = self.extract_features(prices)
        if features is None or self.model.n_updates < MIN_TRAIN_SAMPLES:
            return 0, 0.0, TRADE_EXPIRATION_SECONDS, "Low"

        edge = self.model.predict_proba(features) - 0.5
        self.last_edge = edge
        self.recent_abs_edges.append(abs(edge))

        min_edge, medium_edge, high_edge = self.thresholds()
        if abs(edge) < min_edge:
            return 0, edge, TRADE_EXPIRATION_SECONDS, "Low"

        direction = 1 if edge > 0 else -1
        confidence = "High" if abs(edge) >= high_edge else ("Medium" if abs(edge) >= medium_edge else "Low")
        if confidence != "Low" and not self.gate_allows():
            confidence = "Low"
        return direction, edge, TRADE_EXPIRATION_SECONDS, confidence

    def thresholds(self) -> tuple[float, float, float]:
        if EDGE_MODE != "adaptive" or len(self.recent_abs_edges) < 300:
            return MIN_EDGE, MEDIUM_EDGE, HIGH_EDGE
        ranked = sorted(self.recent_abs_edges)
        last = len(ranked) - 1
        medium = ranked[int(last * (1 - SIGNAL_TOP_PERCENT / 100))]
        high = ranked[int(last * (1 - SIGNAL_TOP_PERCENT / 500))]
        return max(medium, 0.002), medium, max(high, medium)

    def record_shadow(self, edge: float, correct: bool, when: datetime) -> None:
        if abs(edge) < self.thresholds()[0]:
            return
        if self.last_shadow_at is not None and (when - self.last_shadow_at).total_seconds() < TRADE_EXPIRATION_SECONDS:
            return
        self.last_shadow_at = when
        self.shadow_memory.append(bool(correct))

    def gate_allows(self) -> bool:
        outcomes = list(self.shadow_memory)
        if SIGNAL_GATE == "off":
            return True
        if SIGNAL_GATE == "strict":
            # Needs positive evidence: enough data AND lower bound above 50%.
            return len(outcomes) >= 30 and self.wilson_lower_bound(outcomes) >= 0.5
        # relaxed: only blocks when provably worse than a coin flip
        return not (len(outcomes) >= 30 and self.wilson_upper_bound(outcomes) < 0.50)

    def gate_status(self) -> str:
        outcomes = list(self.shadow_memory)
        if not outcomes:
            return f"{SIGNAL_GATE}:{'open' if self.gate_allows() else 'BLOCKED'}(no data)"
        return (
            f"{SIGNAL_GATE}:{'open' if self.gate_allows() else 'BLOCKED'}"
            f"(win={100 * sum(outcomes) / len(outcomes):.0f}% n={len(outcomes)}, "
            f"breakeven={100 * BREAKEVEN:.0f}%)"
        )

    @staticmethod
    def _wilson(outcomes: list[bool], sign: int, default: float) -> float:
        n = len(outcomes)
        if n == 0:
            return default
        p = sum(outcomes) / n
        z = 1.96
        denom = 1 + z**2 / n
        return (p + z**2 / (2 * n) + sign * z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom

    @staticmethod
    def wilson_upper_bound(outcomes: list[bool]) -> float:
        return JarvisLearningModule._wilson(outcomes, +1, 1.0)

    @staticmethod
    def wilson_lower_bound(outcomes: list[bool]) -> float:
        return JarvisLearningModule._wilson(outcomes, -1, 0.0)

    def _record_outcome(self, score: float, was_win: bool) -> None:
        self.outcome_memory.append((score, was_win))
        self.total_trades += 1
        if was_win:
            self.total_wins += 1

    def learn_from_outcome(self, score: float, was_win: bool) -> None:
        self._record_outcome(score, was_win)
        wins = sum(won for _, won in self.outcome_memory)
        logger.info(
            "[JARVIS] recent_signals=%s win_rate=%.1f%% (breakeven %.1f%%)",
            len(self.outcome_memory),
            100 * wins / len(self.outcome_memory) if self.outcome_memory else 0,
            100 * BREAKEVEN,
        )


# ---------------------------------------------------------------------------
# Horizon trackers
# ---------------------------------------------------------------------------
class HorizonTracker:
    def __init__(self, name: str, seconds: int, window_seconds: Optional[int], min_move: float) -> None:
        self.name = name
        self.seconds = seconds
        self.window_seconds = window_seconds
        self.min_move = min_move  # RELATIVE move (fraction of entry price)
        self.model = OnlineLogisticModel()
        self.pending: deque = deque()
        self.hits: deque[bool] = deque(maxlen=500)
        self.strong_hits: deque[bool] = deque(maxlen=500)
        self.ties = 0
        self.small_moves = 0
        self.seen = 0
        self.on_result = None
        self.recent_abs: deque[float] = deque(maxlen=1000)

    def add_sample(self, now: datetime, features: list[float], price: float) -> None:
        edge = (self.model.predict_proba(features) - 0.5) if self.model.n_updates >= 30 else None
        self.pending.append((now, features, price, edge))

    def resolve(self, now: datetime, price: float) -> list[dict]:
        finished = []
        while self.pending:
            sampled_at, features, entry, edge = self.pending[0]
            age = (now - sampled_at).total_seconds()
            if age < self.seconds:
                break
            self.pending.popleft()

            if age > self.seconds + 10:
                continue  # resolved too late (feed gap) – exit price not comparable
            if price == entry:
                self.ties += 1
                continue

            went_up = price > entry
            if edge is not None:
                correct = (edge > 0) == went_up
                self.hits.append(correct)
                if self.on_result:
                    self.on_result(edge, correct, now)
                self.recent_abs.append(abs(edge))
                strong_cut = (
                    sorted(self.recent_abs)[int(0.9 * (len(self.recent_abs) - 1))]
                    if len(self.recent_abs) >= 100 else MEDIUM_EDGE
                )
                if abs(edge) >= strong_cut:
                    self.strong_hits.append(correct)

            if abs(price / entry - 1) < self.min_move:
                self.small_moves += 1
            else:
                self.model.update(features, 1 if went_up else 0)

            self.seen += 1
            if self.seen % STORE_EVERY_N == 0:
                finished.append({
                    "created_at": sampled_at.isoformat(),
                    "horizon_seconds": self.seconds,
                    "features": [round(f, 6) for f in features],
                    "entry_price": entry,
                    "exit_price": round(price, 5),
                    "label": 1 if went_up else 0,
                    "predicted_edge": None if edge is None else round(edge, 4),
                })
        return finished

    def summary(self) -> str:
        def pct(v: deque) -> str:
            return f"{100 * sum(v) / len(v):.1f}% (n={len(v)})" if v else "n/a"
        return f"{self.name}: learned={self.model.n_updates} acc={pct(self.hits)} strong={pct(self.strong_hits)}"


# ---------------------------------------------------------------------------
# Main bot engine
# ---------------------------------------------------------------------------
class EOLivePredictorBot:
    def __init__(self) -> None:
        for name, value in (
            ("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_KEY", SUPABASE_KEY),
            ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN), ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
        ):
            if not value:
                raise RuntimeError(f"Missing required environment variable: {name}")

        self.feed = PlaywrightScraperFeed()
        self.jarvis = JarvisLearningModule()
        self.trackers = [HorizonTracker(*h) for h in HORIZONS]
        self.ensemble = DualModelEnsemble()
        self._ensemble_samples: list[dict] = []
        self.trackers[0].model = self.jarvis.model
        self.trackers[0].on_result = self.jarvis.record_shadow

        self.sample_buffer: list[dict] = []
        self.subscribers: set[str] = set()
        self.is_running = True
        self.trading_unlocked = False
        self.unlock_time = datetime.now() + timedelta(seconds=OBSERVATION_SECONDS)
        self.last_signal_time = datetime.now() - timedelta(seconds=60)
        self.last_test_time = datetime.now() - timedelta(seconds=TEST_INTERVAL_SECONDS)
        self.last_model_save = datetime.now(timezone.utc)

        self.db_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase.")

    def load_samples_for_ensemble(self, limit: int = 5000) -> list[dict]:
        """Pull recent learning_samples from Supabase for ensemble training."""
        try:
            resp = (
                self.db_client.table("learning_samples")
                .select("features, label, horizon_seconds, predicted_edge")
                .order("id", desc=True)
                .limit(limit)
                .execute()
            )
            rows = resp.data or []
            samples = []
            for r in rows:
                feats = r.get("features")
                label = r.get("label")
                if feats is None or label is None:
                    continue
                if isinstance(feats, str):
                    feats = json.loads(feats)
                if len(feats) != 9:
                    continue
                samples.append({
                    "features": [float(x) for x in feats],
                    "label": int(label),
                    "horizon_seconds": r.get("horizon_seconds"),
                })
            return samples
        except Exception as exc:
            logger.error("Failed to load samples for ensemble: %s", exc)
            return []

    async def maybe_retrain_ensemble(self) -> None:
        """Retrain LightGBM + Neural Net if enough new data exists."""
        samples = self.load_samples_for_ensemble(limit=8000)
        if len(samples) < 400:
            return
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, self.ensemble.maybe_retrain, samples)

    def check_database(self) -> None:
        """Fail loudly and clearly if Supabase is mis-set-up (wrong key / missing tables)."""
        try:
            import base64, json
            part = SUPABASE_KEY.split(".")[1]
            role = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("role")
            if role == "anon":
                logger.warning("SUPABASE_KEY is the ANON key. Use the service_role key "
                               "(Supabase > Project Settings > API), or turn off RLS on the tables.")
        except Exception:
            pass
        for table in ("bot_users", "paper_trades", "model_state", "learning_samples"):
            try:
                self.db_client.table(table).select("*").limit(1).execute()
            except Exception as exc:
                logger.error("Supabase table '%s' NOT usable: %s  -> run the SQL from the README "
                             "and use the service_role key.", table, _safe_err(exc))

    async def start(self) -> None:
        logger.info("EO Live Predictor Bot starting...")
        await db(self.check_database)
        await db(self.restore_model_state)
        await db(self.restore_learning_history)
        await self.feed.start()

        tasks = [
            asyncio.create_task(self.monitor_phase(), name="monitor"),
            asyncio.create_task(self.telegram_listener_loop(), name="telegram"),
            asyncio.create_task(self.execution_loop(), name="execution"),
            asyncio.create_task(self.learning_loop(), name="learning"),
            asyncio.create_task(self.resolve_trades(), name="resolve"),
            asyncio.create_task(self.periodic_maintenance(), name="maintenance"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled.")
        finally:
            for t in tasks:
                t.cancel()

    def restore_model_state(self) -> None:
        try:
            response = self.db_client.table("model_state").select("state").eq("id", MODEL_STATE_ID).execute()
            if not response.data:
                return
            saved = response.data[0]["state"].get("horizons", {})
            for tracker in self.trackers:
                if tracker.name in saved:
                    try:
                        tracker.model.load_dict(saved[tracker.name])
                    except Exception as exc:
                        logger.warning("Skipping saved model %s: %s", tracker.name, exc)
            logger.info("Model state restored.")
        except Exception as exc:
            logger.error("Error restoring model state: %s", _safe_err(exc))

    def model_state_snapshot(self) -> dict:
        return {"version": 3, "features": FEATURE_NAMES, "horizons": {t.name: t.model.to_dict() for t in self.trackers}}

    def save_model_state(self, state: Optional[dict] = None) -> None:
        try:
            self.db_client.table("model_state").upsert(
                {
                    "id": MODEL_STATE_ID,
                    "state": state or self.model_state_snapshot(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="id",
            ).execute()
            logger.info("Model state saved.")
        except Exception as exc:
            logger.error("Error saving model state: %s", _safe_err(exc))

    async def flush_samples(self) -> None:
        batch = self.sample_buffer[:500]
        if not batch:
            return
        try:
            await db(lambda: self.db_client.table("learning_samples").insert(batch).execute())
            del self.sample_buffer[: len(batch)]
        except Exception as exc:
            logger.error("Error flushing samples: %s", _safe_err(exc))

    def restore_learning_history(self) -> None:
        try:
            response = (
                self.db_client.table("paper_trades")
                .select("asset_pair,pytorch_output,status,executed_at")
                .eq("asset_pair", "EO Broker Live Chart")
                .in_("status", ["WIN", "LOSS"])
                .gte("executed_at", LEARNING_HISTORY_START_AT.isoformat())
                .order("executed_at", desc=True)
                .limit(LEARNING_HISTORY_LIMIT)
                .execute()
            )
            for trade in reversed(response.data or []):
                self.jarvis._record_outcome(float(trade["pytorch_output"]), trade["status"] == "WIN")
            logger.info("Restored %s historical paper trades.", len(response.data or []))
        except Exception as exc:
            logger.error("Error restoring learning history: %s", _safe_err(exc))

    async def monitor_phase(self) -> None:
        while self.is_running:
            if not self.trading_unlocked and datetime.now() >= self.unlock_time:
                self.trading_unlocked = True
                await broadcast_telegram(
                    "🚀 <b>Scraping Engine Live</b>\nSignals will be calculated once live prices arrive.",
                    self.db_client, self.subscribers,
                )
                logger.info("Trading unlocked – observation period complete.")
            await asyncio.sleep(0.5)

    async def telegram_listener_loop(self) -> None:
        offset = 0
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        while self.is_running:
            try:
                async with httpx.AsyncClient(timeout=45) as client:
                    response = await client.get(url, params={"offset": offset, "timeout": 20})
                    data = response.json()
                    for update in (data.get("result") or []) if data.get("ok") else []:
                        offset = update["update_id"] + 1
                        message = update.get("message")
                        if not message:
                            continue
                        text = message.get("text", "")
                        chat_id = str(message["chat"]["id"])
                        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower() if text else ""
                        if command == "/start":
                            self.subscribers.add(chat_id)
                            name = message["chat"].get("first_name", "Trader")
                            try:
                                await db(lambda: self.db_client.table("bot_users").upsert(
                                    {"chat_id": chat_id, "username": name}, on_conflict="chat_id").execute())
                                await broadcast_telegram("✅ Subscribed to EO Broker signals. Send /stop to unsubscribe.",
                                                         self.db_client, {chat_id})
                            except Exception as exc:
                                logger.error("Error registering subscriber: %s", _safe_err(exc))
                        elif command == "/stop":
                            self.subscribers.discard(chat_id)
                            try:
                                await db(lambda: self.db_client.table("bot_users").delete().eq("chat_id", chat_id).execute())
                            except Exception as exc:
                                logger.error("Error removing subscriber: %s", _safe_err(exc))
            except Exception as exc:
                logger.warning("Telegram listener error: %s", _safe_err(exc))
                await asyncio.sleep(5)
            await asyncio.sleep(2)

    async def log_trade(self, direction: str, price: float, score: float,
                        asset_pair: str = "EO Broker Live Chart") -> bool:
        try:
            await db(lambda: self.db_client.table("paper_trades").insert({
                "asset_pair": asset_pair,
                "direction": direction,
                "entry_price": round(price, 5),
                "pytorch_output": round(score, 4),
                "status": "PENDING",
                # our own clock, so resolution uses the same timebase as price history
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }).execute())
            return True
        except Exception as exc:
            logger.error("DB insertion fault: %s", _safe_err(exc))
            return False

    async def execution_loop(self) -> None:
        while self.is_running:
            try:
                if self.trading_unlocked and self.feed.has_fresh_price() and self.feed.has_full_price_window():
                    prices = self.feed.get_latest_prices(SEQUENCE_LENGTH)
                    features = self.jarvis.extract_features(prices)
                    if self.ensemble.is_ready and features is not None:
                        pred = self.ensemble.predict(features)
                        decision = pred.direction
                        score = pred.edge
                        confidence = pred.confidence
                        if confidence not in ("Medium", "High") or not self.jarvis.gate_allows():
                            decision = 0
                        expiration = TRADE_EXPIRATION_SECONDS
                    else:
                        decision, score, expiration, confidence = self.jarvis.evaluate_sequence(prices)
                    price = prices[-1]
                    since_last = (datetime.now() - self.last_signal_time).total_seconds()

                    if SIGNAL_TEST_MODE and score != 0 and \
                            (datetime.now() - self.last_test_time).total_seconds() >= TEST_INTERVAL_SECONDS:
                        self.last_test_time = datetime.now()
                        direction = "BUY" if score > 0 else "SELL"
                        if await self.log_trade(direction, price, score, asset_pair="EO Broker TEST"):
                            await broadcast_telegram(
                                format_signal(direction, price, score, TRADE_EXPIRATION_SECONDS, test=True),
                                self.db_client, self.subscribers)

                    if decision != 0 and confidence != "Low" and since_last >= SIGNAL_COOLDOWN_SECONDS:
                        direction = "BUY" if decision == 1 else "SELL"
                        self.last_signal_time = datetime.now()  # set first: no double-fire on DB retry
                        if not await self.log_trade(direction, price, score):
                            await asyncio.sleep(1)
                            continue
                        msg = format_signal(direction, price, score, expiration,
                                            confidence=confidence, gate=self.jarvis.gate_status())
                        await broadcast_telegram(msg, self.db_client, self.subscribers)
            except Exception as exc:
                logger.error("Execution loop error: %s", exc)
            await asyncio.sleep(1)

    async def learning_loop(self) -> None:
        last_sample_at: Optional[datetime] = None
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                if self.feed.has_fresh_price() and self.feed.has_full_price_window():
                    price = self.feed.get_current_price()
                    if price is None:
                        await asyncio.sleep(1)
                        continue

                    if last_sample_at is None or (now - last_sample_at).total_seconds() >= LEARNING_SAMPLE_SECONDS:
                        last_sample_at = now
                        for tracker in self.trackers:
                            window = (
                                self.feed.get_latest_prices(SEQUENCE_LENGTH)
                                if tracker.window_seconds is None
                                else self.feed.get_window(tracker.window_seconds)
                            )
                            if not window:
                                continue
                            features = self.jarvis.extract_features(window)
                            if features is not None:
                                tracker.add_sample(now, features, price)

                    for tracker in self.trackers:
                        self.sample_buffer.extend(tracker.resolve(now, price))

                    if len(self.sample_buffer) > SAMPLE_BUFFER_MAX:
                        self.sample_buffer = self.sample_buffer[-SAMPLE_BUFFER_MAX:]
            except Exception as exc:
                logger.error("Learning loop error: %s", exc)
            await asyncio.sleep(1)

    async def resolve_trades(self) -> None:
        """Resolve each trade at the price nearest executed_at + expiry (not 'whenever we poll')."""
        fails = 0
        while self.is_running:
            try:
                response = await db(lambda: self.db_client.table("paper_trades").select("*").eq("status", "PENDING").execute())
                fails = 0
                now_ts = datetime.now(timezone.utc).timestamp()
                for trade in response.data or []:
                    executed_at_str = trade.get("executed_at")
                    if not executed_at_str:
                        continue
                    executed_ts = datetime.fromisoformat(executed_at_str.replace("Z", "+00:00")).timestamp()
                    due_ts = executed_ts + TRADE_EXPIRATION_SECONDS
                    newest = self.feed.newest_ts()
                    if now_ts < due_ts or newest is None or newest < due_ts - 1:
                        continue

                    exit_price = self.feed.price_at(due_ts)
                    trade_id = trade["id"]
                    if exit_price is None:
                        if now_ts - due_ts > VOID_AFTER_SECONDS:
                            await db(lambda: self.db_client.table("paper_trades").update({"status": "VOID"})
                                     .eq("id", trade_id).eq("status", "PENDING").execute())
                            logger.warning("Trade %s voided (no price at expiry).", trade_id)
                        continue

                    direction = trade["direction"]
                    entry = float(trade["entry_price"])
                    score = float(trade.get("pytorch_output") or 0)
                    was_win = (direction == "BUY" and exit_price > entry) or (direction == "SELL" and exit_price < entry)
                    result = "TIE" if exit_price == entry else ("WIN" if was_win else "LOSS")

                    upd = await db(lambda: self.db_client.table("paper_trades").update(
                        {"status": result, "exit_price": round(exit_price, 5)}
                    ).eq("id", trade_id).eq("status", "PENDING").execute())
                    if not upd.data:
                        continue  # another instance resolved it first

                    if result != "TIE" and trade.get("asset_pair") == "EO Broker Live Chart":
                        self.jarvis.learn_from_outcome(score, was_win)
                    logger.info("Resolved trade %s → %s (entry=%.5f exit=%.5f)", trade_id, result, entry, exit_price)
            except Exception as exc:
                fails += 1
                if fails <= 2 or fails % 20 == 0:
                    logger.error("Resolution loop error (x%s): %s", fails, _safe_err(exc))
            await asyncio.sleep(min(60.0, RESOLUTION_POLL_INTERVAL * (1 + fails)))

    async def periodic_maintenance(self) -> None:
        while self.is_running:
            try:
                await self.flush_samples()
                now = datetime.now(timezone.utc)
                if (now - self.last_model_save).total_seconds() >= MODEL_SAVE_SECONDS:
                    snapshot = self.model_state_snapshot()
                    await db(lambda: self.save_model_state(snapshot))
                    await self.maybe_retrain_ensemble()
                    self.last_model_save = now
                    for t in self.trackers:
                        logger.info(t.summary())
            except Exception as exc:
                logger.error("Maintenance error: %s", exc)
            await asyncio.sleep(SAMPLE_FLUSH_SECONDS)

    async def stop(self) -> None:
        logger.info("Shutting down bot...")
        self.is_running = False
        await db(self.save_model_state)
        await self.flush_samples()
        await self.feed.stop()
        logger.info("Bot stopped cleanly.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    bot = EOLivePredictorBot()
    app.state.bot = bot
    task = asyncio.create_task(bot.start())

    def _log_failure(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception():
            logger.error("Bot crashed: %r", t.exception())

    task.add_done_callback(_log_failure)
    try:
        yield
    finally:
        await bot.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="EO Broker Signal Engine", lifespan=lifespan)


@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    bot: Optional[EOLivePredictorBot] = getattr(app.state, "bot", None)
    info = {"status": "Bot is alive and running", "timestamp": datetime.now(timezone.utc).isoformat()}
    if bot:
        info.update({
            "fresh_price": bot.feed.has_fresh_price(),
            "model_updates_30s": bot.jarvis.model.n_updates,
            "gate": bot.jarvis.gate_status(),
        })
    return info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("broker:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
