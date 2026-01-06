#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

echo "Stopping existing candle_cache_service (if any)..."
# Try to gracefully stop by matching the script name; ignore errors if none found.
pkill -f "candle_cache_service.py" >/dev/null 2>&1 || true

sleep 1

echo "Starting candle_cache_service..."
nohup python3 candle_cache_service.py --host 127.0.0.1 --port 9100 > candle_cache.log 2>&1 &
echo "candle_cache_service restarted; logs: candle_cache.log"

