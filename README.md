# RL Trader

Multi-component research and live-trading sandbox for FX, crypto, and equities using reinforcement learning, genetic algorithms, and walk‑forward optimization.

## Top-level layout

- `forex-rl/` – Python trading research & execution stack (actor‑critic, GA, TCN policies, walk‑forward optimization, tools).
- `local-broker/` – Go-based local broker that emulates an OANDA‑style REST/streaming surface for fast offline and paper trading.
- `streamer/` – Market data ingestion and account state utilities (OANDA FX, Bitunix crypto) with QuestDB integration.
- `bkup-forex-rl-YYYYMMDD/` – Snapshots of earlier `forex-rl` trees kept for reference.
- `ref/` – Reference code and vendor SDKs used during experiments (e.g. Alpaca examples).

## Python stack (`forex-rl/`)

Key pieces:

- Offline and live trading runners (e.g. `online_trader.py`, `multi_instrument_box_trading.py`, `tpsl_trader.py`).
- Actor‑critic and TCN research code under `actor-critic/` plus GA stacks under `ga-multi20/`, `ga-ness/`, and `ga-trailing-20/`.
- Walk‑forward optimization framework under `wfo/` (see `forex-rl/wfo/README.md` for detailed CLI docs).
- Feature engineering and hybrid models under `hybrid-agnostic/` and the autoencoder scripts in `supervised-ae/` and `unsupervised-ae/`.
- Tools for universe construction, account PnL, and OANDA/Alpaca connectivity under `tools/`.

There is intentionally no single pinned `requirements.txt` for the Python stack; dependencies are managed per experiment. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Core libraries (adjust as needed for a given script)
pip install numpy pandas scipy torch matplotlib oandapyV20 requests
```

Running tests (quick smoke over loaders, simulators, and strategies):

```bash
cd forex-rl
pytest -q
```

For walk‑forward experiments, see the examples in `forex-rl/wfo/README.md`. For GA Trailing‑20 work, see `forex-rl/ga-trailing-20/ROADMAP.md`.

## Local broker (`local-broker/`)

A Go service that exposes an OANDA‑compatible REST + streaming surface backed by an in‑memory engine and pluggable pricing sources.

High level:

- Pricing sources for OANDA streams and synthetic ticks (`internal/pricing`).
- In‑memory engine with TP/SL/trailing‑stop enforcement (`internal/state`).
- File‑backed store for equity/time‑series snapshots (`internal/storage`).
- HTTP server wiring + config loader under `cmd/broker` and `internal/server`.

Development loop:

```bash
cd local-broker
go test ./...
go run ./cmd/broker
```

More details live in `local-broker/LOCAL_BROKER_PLAN.md`.

## Market data and streaming (`streamer/`)

Utilities to ingest and explore market data, with scripts to push OANDA FX and Bitunix crypto into QuestDB and to track account state.

Examples:

- `forex-stream.py` – OANDA FX tick/stream recorder.
- `bitunix_to_questdb.py`, `oanda_to_questdb.py` – ingestion to QuestDB.
- `questdb_ingest.py`, `questdb_admin.py` – admin/helpers for QuestDB tables.
- `orders.py`, `account_state.py` – simple account/order abstractions used by the streaming tools.

Systemd unit examples for long‑running ingestion live under `streamer/systemd/`.

## Secrets and environment

- Local secrets (API keys, account IDs, etc.) live under `.secrets/` and are **not** intended to be committed.
- Most Python entrypoints read credentials from environment variables or `.env`‑style files in `.secrets/`; see the relevant script’s docstring for exact names.

## Optional: OANDA tick recorder

- See `streamer/forex-stream.py` (requires `oandapyV20` and env vars like `OANDA_DEMO_ACCOUNT_ID`, `OANDA_DEMO_KEY`).
