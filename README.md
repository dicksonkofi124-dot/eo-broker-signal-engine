# EO Broker Signal Bot (v2)

Reads the live EUR/USD price from the Expert Option web page, learns from it
with an online machine-learning model, and sends you **paper-trading signals on
Telegram**. You place any real trades yourself. The bot never clicks or trades.

> **Disclaimer.** Reading prices from Expert Option's page may go against their
> Terms of Service, and you use this at your own risk. Signals are not advice.
> A 30-second up/down call is close to a coin flip: with an ~85% payout you need
> to win about **54%** of trades just to break even. Judge the bot on its logged
> win rate over hundreds of paper trades before risking any real money.

---

## What you need

| Item | Notes |
|------|-------|
| A Windows / Mac / Linux PC **with a screen** | The browser window must stay open. It cannot run on Render or a phone. |
| Python 3.10 or newer | python.org. On Windows tick **Add Python to PATH** in the installer. |
| A free Supabase project | supabase.com. Stores trades and the model's learning. |
| A Telegram bot | Made with @BotFather. Sends you the signals. |
| An Expert Option account | You log in yourself in the bot's browser. |

---

## Step 1: Get the project onto your PC

Unzip `eo_broker_bot_v2.zip` to a folder, for example `C:\eo_broker_bot`.
You should see `broker.py`, `find_selector.py`, `requirements.txt`, `.env.example`.

## Step 2: Install

Open a terminal in that folder (Windows: open the folder, click the address bar,
type `cmd`, press Enter). Then run these one at a time:

```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

You should see `(venv)` at the start of the terminal line after the second command.
If `python` is not found, try `py` instead.

## Step 3: Create the Telegram bot

1. In Telegram, open **@BotFather** and send `/newbot`. Follow the prompts.
2. Copy the **token** it gives you (looks like `123456789:ABC...`).
3. Open your new bot and press **Start** (or send `/start`).
4. Get your chat id: message **@userinfobot**, it replies with your numeric `Id`.

## Step 4: Set up Supabase

1. Create a project at supabase.com.
2. Open **SQL Editor**, paste the block below, and press **Run**.
3. Open **Project Settings > API**. Copy the **Project URL** and the
   **service_role** key. Keep the service_role key secret and never commit it.

```sql
create table bot_users (
  chat_id text primary key,
  username text,
  created_at timestamptz default now()
);

create table paper_trades (
  id bigint generated always as identity primary key,
  asset_pair text,
  direction text,
  entry_price numeric,
  exit_price numeric,
  pytorch_output numeric,                 -- the model's edge score
  status text default 'PENDING',          -- PENDING | WIN | LOSS | TIE | VOID
  executed_at timestamptz default now()
);

create table model_state (
  id text primary key,
  state jsonb,
  updated_at timestamptz
);

create table learning_samples (
  id bigint generated always as identity primary key,
  created_at timestamptz,
  horizon_seconds int,
  features jsonb,
  entry_price numeric,
  exit_price numeric,
  label int,
  predicted_edge numeric
);
```

If the bot later logs `permission denied for table ...` (code 42501), run this
too in the SQL Editor (newer Supabase projects don't grant API access to new tables by default):

```sql
grant usage on schema public to service_role;
grant all on all tables in schema public to service_role;
grant all on all sequences in schema public to service_role;
alter default privileges in schema public grant all on tables to service_role;
alter default privileges in schema public grant all on sequences to service_role;
```

## Step 5: Fill in `.env`

1. Copy `.env.example` to a new file named `.env` (same folder).
2. Edit it:

```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your-service_role-key
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_CHAT_ID=your-numeric-id

PRICE_MIN=0.9
PRICE_MAX=1.4
PRICE_SELECTOR=
```

Leave `PRICE_SELECTOR` empty for now. The next step finds it.

## Step 6: Find the price selector (once)

The bot has to know which element on the page holds the price.

```bash
python find_selector.py
```

1. A browser opens on Expert Option. **Log in.**
2. Open the **EUR/USD** chart on the trading screen.
3. Go back to the terminal and press **Enter**.
4. It prints elements whose value changed in 2 seconds. Pick the one showing the
   live price (like `1.08543 -> 1.08546`).
5. Copy the part before ` @` (for example `span.current-price`) into `.env`:
   `PRICE_SELECTOR=span.current-price`
6. Close the browser window.

If it prints "None changed", the chart wasn't open or the market was closed
(forex pauses on weekends). Try again when EUR/USD is moving.

## Step 6b: Price source (recommended, replaces the selector)

Expert Option draws its chart on a canvas, so there is no text to scrape. The
bot instead reads the live EUR/USD ticks the page already receives over its own
websocket (read-only; other traders' messages are ignored).

In `.env`:

```
PRICE_SOURCE=ws
ASSET_ID=auto
```

`142` was the EUR/USD id in one session and `245` in another, so ids can change. Use `ASSET_ID=auto` (the bot then uses the one asset ticking inside your `PRICE_MIN`/`PRICE_MAX` range). To confirm it: run the bot,
open the EUR/USD chart, and compare the log line `First price received: ...`
with the price on your chart. If they don't match, leave `ASSET_ID` empty;
the bot then logs candidate ids with prices in your `PRICE_MIN`/`PRICE_MAX`
range. Pick the one that matches your chart. Keep the EUR/USD chart open while
the bot runs. If EO changes its data format, the bot goes quiet ("Price stale")
and `ws_probe.py` / `ws_probe_candles.py` can be rerun to find the new layout.

## Step 7: Run the bot

```bash
python broker.py
```

1. The browser opens. Log in if asked, then open the EUR/USD trading screen.
   Your login is saved in the `eo_profile` folder, so next time it's usually automatic.
2. Leave the window open and don't minimize-to-sleep the PC.
3. Telegram sends "Scraping Engine Live" once it starts.
4. In the terminal you should see lines like `Model state saved` and
   per-horizon summaries every 5 minutes.

To stop: press **Ctrl+C** in the terminal and wait for "Bot stopped cleanly",
so the learning is saved.

---

## What to expect

- **Warm-up:** no signals until about 300 samples are learned, roughly 30 to 60
  minutes of live prices. Per-horizon accuracy shows in the logs.
- **Signals:** Telegram messages like "BUY PATTERN ALERT (paper)" with the price,
  score, 30-second expiry and confidence. At most one per ~55 seconds.
- **Resolution:** each signal is scored WIN / LOSS / TIE (or VOID if the price
  feed dropped) 30 seconds later and stored in `paper_trades`.
- **Weekends:** forex closes, so the price stops and the bot goes quiet. That's normal.

## Check whether it actually works

In Supabase, open the `paper_trades` table. After a few hundred resolved trades:

```sql
select status, count(*) from paper_trades
where status in ('WIN','LOSS') group by status;
```

Win rate = WIN / (WIN + LOSS). It needs to stay above about **54%** to make
money at an 85% payout. Under that, the signals lose money in the long run.

---

## Settings (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `PRICE_SOURCE` | ws | `ws` = read live ticks from the page's websocket; `dom` = old text scraping |
| `ASSET_ID` | none | EUR/USD id in the feed (see Step 6b) |
| `PRICE_SELECTOR` | empty | Only for `PRICE_SOURCE=dom` |
| `PRICE_MIN` / `PRICE_MAX` | 0.9 / 1.4 | Ignore any number outside this range (EUR/USD) |
| `SIGNAL_GATE` | relaxed | `off`, `relaxed` (blocks only if provably below 50%), `strict` (signals only after proven edge) |
| `EDGE_MODE` | adaptive | `adaptive` uses top-percent thresholds, `fixed` uses fixed ones |
| `SIGNAL_TOP_PERCENT` | 3 | Lower = fewer, more selective signals |
| `PAYOUT` | 0.85 | Your broker payout, used to show the breakeven rate |
| `USER_DATA_DIR` | ./eo_profile | Saved browser profile (keeps your login) |
| `BROWSER_HEADLESS` | false | Run Chromium without a visible window; use only after logging in and saving the browser profile |
| `ENSEMBLE_ENABLED` | true | Train/use the LightGBM + neural-network ensemble; disable on low-memory machines |

`strict` is the safest choice: it stays silent until the model has shown a real edge.

The ensemble additionally requires NumPy, scikit-learn, and LightGBM. On a
low-memory machine, set `ENSEMBLE_ENABLED=false` to use the online logistic
model without loading those packages. Headless mode cannot be used to manually
log in; first create an authenticated browser profile, then enable it.

---

## Telegram commands

- `/start` subscribes a chat to signals
- `/stop` unsubscribes

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Missing required environment variable` | A value in `.env` is empty. Fill it in. |
| `python` / `pip` not found | Reinstall Python with "Add to PATH", or use `py`. |
| Browser opens but no price in the logs | Wrong selector. Re-run `find_selector.py` with EUR/USD open. |
| "Waiting for you to log in..." keeps showing | Log in and open the trading screen. It never restarts while waiting. |
| "Price stale" warnings | The chart stopped (market closed or page frozen). It reloads, then relaunches automatically. |
| No Telegram messages | Press Start on your bot, check the token and chat id. |
| `permission denied for table` (42501) | Run the GRANT SQL in Step 4, and make sure `SUPABASE_KEY` is the service_role/secret key. |
| `Price stale` again and again | Keep the browser window visible and the Mac awake: run `caffeinate -dims python broker.py`. Check your connection (Telegram timeouts mean it is flaky). |
| `APIError` / `Supabase table ... NOT usable` | Use the **service_role** key (not anon) and run the SQL from Step 4. The bot checks this at startup and says which table failed. |
| `find_selector.py` finds nothing | Make sure the live price is visible on screen first. Forex is **closed on weekends** (Fri evening to Sun ~22:00 UTC), so EUR/USD won't move; try again when it is open. Paste the output if it still finds nothing. |
| PC went to sleep, bot silent | Turn off sleep in power settings, then restart the bot. |
| Browser hangs or crashes on launch | Install Google Chrome (google.com/chrome), then in `.env` set `BROWSER_CHANNEL=chrome` and `USER_DATA_DIR=./eo_profile_chrome`. Quit all Chrome windows first. If it still hangs: `pkill -f eo_profile`, delete the profile folder, retry. |
| GPU / EGL errors, browser hangs | Add `BROWSER_ARGS=--disable-gpu` to `.env` (software rendering; the chart still works). |
| `playwright install` fails | Re-run it with the `venv` active. |

---

## Files

| File | Purpose |
|------|---------|
| `broker.py` | The bot |
| `find_selector.py` | One-time helper to locate the price element |
| `.env.example` | Template for your settings |
| `requirements.txt` | Python packages |
| `eo_profile/` | Created automatically. Holds your saved login, never commit or share it. |

## Changes in v2

- Browser uses a saved profile, so recovery no longer logs you out, and it never restarts while you log in.
- Database calls no longer freeze the scraper; the Telegram token no longer shows in logs.
- Better price parsing, a range guard and a jump filter against wrong elements.
- Trades are scored at the price exactly 30 seconds after entry.
- Model: 9 features, forgetting statistics, gradient clipping. Saved model id is `eobroker_live_v2`, so it starts fresh.
- `strict` gate now requires a proven edge (lower 95% bound at or above 50%).
