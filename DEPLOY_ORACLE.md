# Running the bot 24/7 on an Oracle Cloud free VM (Ubuntu)

Stop the bot on your Mac first. Two copies share one Telegram bot and one
Supabase project and will fight each other.

## 1. Swap (do this if the VM has 1 GB RAM)

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

On 1 GB, also put `ENSEMBLE_ENABLED=false` in `.env` (Chromium + LightGBM + the
network do not fit comfortably).

## 2. Packages

```bash
sudo apt update && sudo apt install -y python3-venv xvfb x11vnc
cd ~/eo_broker_bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install --with-deps chromium
chmod 600 .env
```

On ARM (Ampere A1) there is no Google Chrome: leave `BROWSER_CHANNEL` unset and
use Playwright's own Chromium (installed above).

## 3. First login (once), through a virtual screen

The bot needs a logged-in Expert Option session. A profile copied from your Mac
usually will not work, so log in on the server:

```bash
Xvfb :99 -screen 0 1280x800x24 &
export DISPLAY=:99
x11vnc -display :99 -localhost -forever -nopw &
python broker.py
```

On your Mac, in another terminal: `ssh -L 5900:localhost:5900 ubuntu@SERVER_IP`
then Finder > Go > Connect to Server > `vnc://localhost:5900`.
Log in, open EUR/USD, check the log shows `First price received`, then press
Ctrl+C in the server terminal and stop Xvfb/x11vnc (`pkill Xvfb; pkill x11vnc`).
The login stays in `eo_profile/`. Do NOT open ports 5900 or 8000 in Oracle's
firewall; the SSH tunnel is enough.

Expert Option may email a verification code the first time it sees a new
location/IP. Complete it in the VNC window.

## 4. Run it as a service

Save as `/etc/systemd/system/eo-broker.service` (change the user and paths):

```ini
[Unit]
Description=EO Broker Signal Bot
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/eo_broker_bot
ExecStart=/usr/bin/xvfb-run -a --server-args="-screen 0 1280x800x24" /home/ubuntu/eo_broker_bot/venv/bin/python broker.py
Restart=on-failure
RestartSec=20
KillSignal=SIGINT
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now eo-broker
journalctl -u eo-broker -f            # live log
sudo systemctl stop eo-broker         # saves the model, like Ctrl+C
```

`KillSignal=SIGINT` makes a stop behave like Ctrl+C, so the model is saved.

## 5. Check it is healthy

```bash
curl -s localhost:8000/
```

`fresh_price` should be `true` and `model_updates_30s` should keep growing. If it
does not grow for several minutes while the market is open, the page is logged
out or the quote is frozen: reconnect with VNC and look.

## Notes

- `BROWSER_HEADLESS=true` runs without a screen once the profile is logged in.
  Try it only if Xvfb is a problem; some sites behave differently headless.
- Keep the clock correct: `timedatectl` should say "System clock synchronized: yes".
- Rotate any Telegram token or Supabase key that was ever pasted into a chat.
