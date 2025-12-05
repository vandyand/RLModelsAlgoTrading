# Online Forex RL

A reinforcement learning trading loop with a Dueling Double DQN over historical minute candles or a UDP stream. Includes a simple Flask dashboard and a CSV-to-UDP simulator.

## Quick start (historical)

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -r forex-rl/requirements.txt

# Run training on historical CSV
cd forex-rl
python dqn_trading.py
```

- Dashboard runs at http://localhost:5000 and serves JSON endpoints `/data` and `/time_stats` and a basic HTML page at `/`.
- Adjust `forex-rl/config.json` to change DQN or market parameters and date ranges. Keep `use_historical: true` for this mode.

## UDP streaming mode

```bash
# Terminal A (listener + trainer)
cd forex-rl
python dqn_trading.py

# Terminal B (simulator)
cd forex-rl
python stream_candlesv2.py
```

- Configure UDP IP/port in `forex-rl/config.json` and in `stream_candlesv2.py`'s `CONFIG` (defaults may not match your host).
- Set `use_historical: false` to use UDP live stream.

## Notes
- Run from the `forex-rl` directory so relative files resolve correctly. The code now also resolves `config.json` relative to the module itself.
- Large defaults like `n_env=150` are compute-heavy; reduce for laptops.
- The simulator requires CSV `nq-1m-processed.csv` in `forex-rl/` (or edit path in config/simulator).

## Optional: OANDA tick recorder
- See `streamer/forex-stream.py` (requires `oandapyV20` and env vars `OANDA_DEMO_ACCOUNT_ID`, `OANDA_DEMO_KEY`).
