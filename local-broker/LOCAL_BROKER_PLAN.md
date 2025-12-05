# Local Broker Plan (Living Document)

Keep this file updated as implementation progresses so we can coordinate across algorithms and infra.

## ✅ Completed
- Selected Go (Chi + stdlib) for the high-throughput broker service and initialized the module layout (`cmd/`, `internal/`).
- Added configuration loader with sane defaults plus env overrides for account metadata and upstream OANDA credentials.
- Implemented pricing layer:
  - `OandaStreamSource` reconnects to the OANDA PricingStream and normalizes ticks.
  - `SyntheticSource` generates deterministic ticks for local dev/testing.
  - `Manager` caches top-of-book snapshots and fans them out to subscribers.
- Built the in-memory trading engine with:
  - Market-order fills using latest bid/ask,
  - Account balance/NAV tracking,
  - Automatic take-profit, stop-loss, and trailing-stop enforcement on every tick.
- Exposed an OANDA-compatible REST surface (`/v3/accounts/...`) so existing RL runners can swap to the local broker by flipping the base URL.
- Added entrypoint wiring (config → pricing manager → engine → HTTP server) plus graceful shutdown signals.
- Wired persistence via a file-backed store (state snapshots + equity JSONL) and restored state on boot.
- Added `/v3/accounts/{id}/pricing/stream` SSE endpoint and tick fan-out for low-latency consumers.
- Introduced live multi-account creation (`POST /v3/accounts`) and engine support for concurrent ledgers.
- Delivered historical data endpoints for dashboards: `/transactions` and `/equity`.
- Added LIMIT/STOP order entry with pending-order evaluation plus configurable margin checks.
- Created Docker image (`local-broker/Dockerfile`) and CI workflow (Go fmt + tests).

## ⏭️ Next Up
- Expand analytics endpoints (CSV downloads, filtered queries, PnL rollups).
- Broaden order book: support more order types (stop-loss orders on existing positions, guaranteed stops) and advanced margin/margin-call flows.
- Multi-account management: CRUD (delete/transfer), per-account permissions, and eventual auth/tokening.
- Scenario/backtest harness that replays recorded ticks through the engine for regression testing before going live.
- Broker telemetry: Prometheus metrics + alerting hooks so ops can monitor latency, fills, and margin utilization.

_Last updated: 2025-11-26_
