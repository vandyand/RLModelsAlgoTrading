Systemd service units for ingestion to QuestDB.

1) Create environment files with credentials and options (edit paths if needed):

  sudo install -m 0640 -o root -g root /dev/stdin /etc/default/oanda-fx-ingest <<'EOF'
OANDA_DEMO_ACCOUNT_ID=YOUR_ACCOUNT_ID
OANDA_DEMO_KEY=YOUR_ACCESS_TOKEN
# QuestDB connection
QUESTDB_HOST=127.0.0.1
QUESTDB_ILP_PORT=9009
QUESTDB_HTTP_PORT=9000
# Optional batching (0 = immediate)
QDB_BATCH=0
# Optional overrides
QDB_FX_TICKS_TABLE=fx_ticks
QDB_FX_DOM_TABLE=fx_dom
EOF

  sudo install -m 0640 -o root -g root /dev/stdin /etc/default/bitunix-crypto-ingest <<'EOF'
# QuestDB connection
QUESTDB_HOST=127.0.0.1
QUESTDB_ILP_PORT=9009
QUESTDB_HTTP_PORT=9000
# Channels: enable or disable via flags in unit ExecStart as needed
QDB_BATCH=200
# Default symbols (comma-separated). Override as needed.
SYMBOLS=BTCUSDT,ETHUSDT,XRPUSDT,BCHUSDT,LTCUSDT,BNBUSDT,ADAUSDT,BATUSDT,ETCUSDT,XLMUSDT,ZRXUSDT,DOGEUSDT,ATOMUSDT,DOTUSDT,LINKUSDT,UNIUSDT,SOLUSDT,AVAXUSDT,MATICUSDT,FILUSDT
EOF

2) Install unit files:

  sudo install -m 0644 -o root -g root streamer/systemd/oanda-fx-ingest.service /etc/systemd/system/oanda-fx-ingest.service
  sudo install -m 0644 -o root -g root streamer/systemd/bitunix-crypto-ingest.service /etc/systemd/system/bitunix-crypto-ingest.service

3) Reload and enable:

  sudo systemctl daemon-reload
  sudo systemctl enable --now oanda-fx-ingest.service
  sudo systemctl enable --now bitunix-crypto-ingest.service

4) Check logs:

  journalctl -u oanda-fx-ingest -f
  journalctl -u bitunix-crypto-ingest -f

Notes:
- Services auto-restart on failure. They depend on network-online and docker.service to ensure QuestDB is reachable.
- Tune QDB_BATCH for throughput vs latency. 0 means send each line immediately.
- For Bitunix, adjust ExecStart to include --enable-trade/--enable-price if needed.
