Systemd service units for candle cache and TCN multi-policy live trader.

These units mirror the style of the ingestion services under streamer/systemd,
but are focused on:

- Keeping the candle cache HTTP service running with auto-restart.
- Keeping the tcn_multi_policy_live_trader running with auto-restart and
  depending on the candle cache.

Adjust paths (WorkingDirectory, ExecStart) as needed for your host/venv. The
examples below assume the code is deployed under /root/rl-trader.


1) Create environment files under the repo
------------------------------------------

Instead of using /etc/default, these units load credentials from the local
`.secrets` directory under the project root. This avoids permission issues and
keeps everything self-contained.

From `/home/kingjames/rl-trader`:

1a) Candle cache (`.secrets/candle-cache.env`):

  cat > .secrets/candle-cache.env <<'EOF'
OANDA_ENV=practice
OANDA_DEMO_KEY=YOUR_OANDA_TOKEN
EOF

Required for candle cache:

- OANDA_ENV            practice|live (here: practice)
- One of:
  - OANDA_DEMO_KEY     (for practice)
  - OANDA_ACCESS_TOKEN (alternative var name)


1b) TCN multi-policy live trader (`.secrets/tcn-multi-live.env`):

  cat > .secrets/tcn-multi-live.env <<'EOF'
OANDA_ENV=practice
OANDA_DEMO_ACCOUNT_ID=YOUR_OANDA_ACCOUNT_ID
OANDA_DEMO_KEY=YOUR_OANDA_TOKEN
# Optional: override candle cache base URL (defaults to http://127.0.0.1:9100)
CANDLE_CACHE_BASE=http://127.0.0.1:9100
EOF

Required for tcn_multi_policy_live_trader (practice environment):

- OANDA_DEMO_ACCOUNT_ID   your practice account id
- OANDA_DEMO_KEY          your practice REST API token

Optional:

- OANDA_ENV               practice|live (top-level config also controls this)
- OANDA_LIVE_ACCOUNT_ID   (if you ever switch to env="live")
- OANDA_LIVE_KEY          (if you ever switch to env="live")
- CANDLE_CACHE_BASE       URL of the candle cache (defaults to http://127.0.0.1:9100)


2) Install unit files
---------------------

From the project root (rl-trader), copy the units into /etc/systemd/system:

  sudo install -m 0644 -o root -g root forex-rl/systemd/candle-cache.service /etc/systemd/system/candle-cache.service
  sudo install -m 0644 -o root -g root forex-rl/systemd/tcn-multi-live.service /etc/systemd/system/tcn-multi-live.service


3) Reload systemd and enable services
-------------------------------------

  sudo systemctl daemon-reload
  sudo systemctl enable --now candle-cache.service
  sudo systemctl enable --now tcn-multi-live.service


4) Checking status and logs
---------------------------

- Service status:

  systemctl status candle-cache.service
  systemctl status tcn-multi-live.service

- Follow logs (journal):

  journalctl -u candle-cache.service -f
  journalctl -u tcn-multi-live.service -f

- Retrain audit log (written by tcn_multi_policy_live_trader):

  # From forex-rl/
  tail -F live_retrain.log

  # Example queries:
  grep '"instrument": "EUR_USD"' live_retrain.log
  grep '"kind": "scheduled"' live_retrain.log


5) Notes
--------

- Both units are configured with Restart=always so they automatically restart
  on failure or crashes.
- The tcn-multi-live service declares a dependency on candle-cache.service so
  the live trader will only start after the candle cache is up, and will be
  restarted independently if either side crashes.
- If you prefer file-based logs instead of journal-only output, you can edit
  the unit files to use StandardOutput=append:/path/to/log. The trader also
  writes a compact retrain audit log to forex-rl/live_retrain.log regardless
  of how the service is started.

