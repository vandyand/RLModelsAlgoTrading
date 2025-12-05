#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/rl-trader"
cd "$REPO_ROOT"

# Source ~/.bashrc to load exported env (work around PS1 guard)
if [[ -f "/root/.bashrc" ]]; then
  export PS1="cron"
  # Temporarily disable nounset to avoid unbound var errors from .bashrc
  set +u
  # shellcheck source=/dev/null
  . "/root/.bashrc"
  set -u
fi

# Also source optional secrets file if present
SECRETS_DIR="$REPO_ROOT/.secrets"
SECRETS_FILE="$SECRETS_DIR/oanda.env"
if [[ -f "$SECRETS_FILE" ]]; then
  # shellcheck source=/dev/null
  set -a
  . "$SECRETS_FILE"
  set +a
fi

# Lightweight arg parsing: --force/--check
ARG_FORCE=0
ARG_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --force) ARG_FORCE=1 ;;
    --check) ARG_CHECK=1 ;;
  esac
done

if [[ $ARG_FORCE -eq 1 ]]; then
  export SKIP_TIME_GUARD=1
fi

# Prefer project venv python if available; otherwise fallback to system python
PYTHON_CANDIDATES=(
  "/root/aid-env/bin/python3"
  "/usr/bin/python3"
)
if command -v python3 >/dev/null 2>&1; then PYTHON_CANDIDATES+=("$(command -v python3)"); fi
if command -v python >/dev/null 2>&1; then PYTHON_CANDIDATES+=("$(command -v python)"); fi

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found" >&2
  exit 1
fi

# ET time gating (DST-safe): default to 18:24 ET Sun–Thu
if [[ "${SKIP_TIME_GUARD:-0}" != "1" ]]; then
  ET_HOUR_EXPECTED="${ET_HOUR:-18}"
  ET_MINUTE_EXPECTED="${ET_MINUTE:-24}"
  ET_NOW_HOUR=$(TZ=America/New_York date +%H)
  ET_NOW_MIN=$(TZ=America/New_York date +%M)
  ET_NOW_DOW=$(TZ=America/New_York date +%u)
  case "$ET_NOW_DOW" in
    7|1|2|3|4) : ;;
    *) echo "Skip: ET DOW $ET_NOW_DOW not in Sun-Thu"; exit 0 ;;
  esac
  if [[ "$ET_NOW_HOUR" != "$ET_HOUR_EXPECTED" || "$ET_NOW_MIN" != "$ET_MINUTE_EXPECTED" ]]; then
    echo "Skip: ET time $ET_NOW_HOUR:$ET_NOW_MIN != ${ET_HOUR_EXPECTED}:${ET_MINUTE_EXPECTED}"
    exit 0
  fi
fi

if [[ $ARG_CHECK -eq 1 ]]; then
  echo "Check mode: verifying environment and scripts..."
  echo "REPO_ROOT=$REPO_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "SECRETS_FILE=$SECRETS_FILE (exists=$([[ -f \"$SECRETS_FILE\" ]] && echo yes || echo no))"
  [[ -f "forex-rl/unsupervised-ae/infer_td3_on_latent.py" ]] || { echo "Missing infer_td3_on_latent.py" >&2; exit 2; }
  [[ -f "forex-rl/actor-critic/align_positions_to_targets.py" ]] || { echo "Missing align_positions_to_targets.py" >&2; exit 3; }
  if [[ -z "${OANDA_DEMO_ACCOUNT_ID:-}" || -z "${OANDA_DEMO_KEY:-}" ]]; then
    echo "Warning: OANDA creds not present in environment (OANDA_DEMO_ACCOUNT_ID / OANDA_DEMO_KEY)" >&2
  else
    echo "OANDA creds detected in environment"
  fi
  echo "All checks passed."
  exit 0
fi

echo "[$(date -Iseconds)] Starting pca-2head -> align pipeline"
"$PYTHON_BIN" "forex-rl/unsupervised-ae/infer_td3_on_latent.py" --last-only --json | \
  "$PYTHON_BIN" "forex-rl/actor-critic/align_positions_to_targets.py" --targets - --account-suffix 2 --invert
EXIT_CODE=$?
echo "[$(date -Iseconds)] Pipeline finished with exit code $EXIT_CODE"
exit $EXIT_CODE
