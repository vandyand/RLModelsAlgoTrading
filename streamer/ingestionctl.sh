#!/usr/bin/env bash
set -euo pipefail

# Simple manager for FX/Crypto ingestion services (systemd)
# - Installs/updates service units
# - Creates env files from current shell env
# - Starts/stops/restarts, tails logs, basic health checks
#
# Usage examples:
#   streamer/ingestionctl.sh install           # install units + env files, daemon-reload
#   streamer/ingestionctl.sh enable            # enable both services at boot
#   streamer/ingestionctl.sh start [fx|crypto] # start services
#   streamer/ingestionctl.sh stop [fx|crypto]
#   streamer/ingestionctl.sh restart [fx|crypto]
#   streamer/ingestionctl.sh status [fx|crypto]
#   streamer/ingestionctl.sh logs [fx|crypto] [-f] [-n 200]
#   streamer/ingestionctl.sh health           # recent row counts
#   streamer/ingestionctl.sh setdepth 15      # set crypto depth level and restart
#   streamer/ingestionctl.sh setsymbols BTCUSDT,ETHUSDT,... # set crypto symbols and restart
#   streamer/ingestionctl.sh configure        # (re)write env files from current env
#   streamer/ingestionctl.sh uninstall        # disable + remove units (keeps data)
#
# Notes:
# - Requires systemd (systemctl, journalctl) and curl for health.
# - Run as root for system wide unit installation.

# Resolve repo root based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FX_UNIT_SRC="${REPO_ROOT}/streamer/systemd/oanda-fx-ingest.service"
CR_UNIT_SRC="${REPO_ROOT}/streamer/systemd/bitunix-crypto-ingest.service"
FX_UNIT_DST="/etc/systemd/system/oanda-fx-ingest.service"
CR_UNIT_DST="/etc/systemd/system/bitunix-crypto-ingest.service"
FX_ENV="/etc/default/oanda-fx-ingest"
CR_ENV="/etc/default/bitunix-crypto-ingest"

need_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "This command must be run as root (sudo)." >&2
    exit 1
  fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

reload_daemon() { systemctl daemon-reload; }

enable_services() {
  systemctl enable oanda-fx-ingest.service || true
  systemctl enable bitunix-crypto-ingest.service || true
}

start_services() {
  local which=${1:-all}
  if [[ "$which" == "fx" || "$which" == "all" ]]; then systemctl start oanda-fx-ingest.service; fi
  if [[ "$which" == "crypto" || "$which" == "all" ]]; then systemctl start bitunix-crypto-ingest.service; fi
}

stop_services() {
  local which=${1:-all}
  if [[ "$which" == "fx" || "$which" == "all" ]]; then systemctl stop oanda-fx-ingest.service || true; fi
  if [[ "$which" == "crypto" || "$which" == "all" ]]; then systemctl stop bitunix-crypto-ingest.service || true; fi
}

restart_services() {
  local which=${1:-all}
  if [[ "$which" == "fx" || "$which" == "all" ]]; then systemctl restart oanda-fx-ingest.service; fi
  if [[ "$which" == "crypto" || "$which" == "all" ]]; then systemctl restart bitunix-crypto-ingest.service; fi
}

status_services() {
  local which=${1:-all}
  if [[ "$which" == "fx" || "$which" == "all" ]]; then systemctl --no-pager --full status oanda-fx-ingest.service || true; fi
  if [[ "$which" == "crypto" || "$which" == "all" ]]; then systemctl --no-pager --full status bitunix-crypto-ingest.service || true; fi
}

logs_service() {
  local which=${1:-all}; shift || true
  local tail_args=("-n" "200");
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--follow) tail_args+=("-f"); shift;;
      -n) tail_args+=("-n" "$2"); shift 2;;
      *) break;;
    esac
  done
  if [[ "$which" == "fx" || "$which" == "all" ]]; then echo "--- logs: oanda-fx-ingest ---"; journalctl -u oanda-fx-ingest.service "${tail_args[@]}"; fi
  if [[ "$which" == "crypto" || "$which" == "all" ]]; then echo "--- logs: bitunix-crypto-ingest ---"; journalctl -u bitunix-crypto-ingest.service "${tail_args[@]}"; fi
}

install_units() {
  need_root
  if [[ ! -f "$FX_UNIT_SRC" || ! -f "$CR_UNIT_SRC" ]]; then
    echo "Cannot find unit files under ${REPO_ROOT}/streamer/systemd/." >&2
    exit 1
  fi
  install -m 0644 -o root -g root "$FX_UNIT_SRC" "$FX_UNIT_DST"
  install -m 0644 -o root -g root "$CR_UNIT_SRC" "$CR_UNIT_DST"
  reload_daemon
  echo "Installed unit files to /etc/systemd/system." >&2
}

write_env_files() {
  need_root
  # Compose defaults from current shell env (leave blank if missing)
  local acc="${OANDA_DEMO_ACCOUNT_ID:-${OANDA_ACCOUNT_ID:-}}"
  local key="${OANDA_DEMO_KEY:-${OANDA_ACCESS_TOKEN:-}}"
  local qhost="${QUESTDB_HOST:-127.0.0.1}"
  local qilp="${QUESTDB_ILP_PORT:-9009}"
  local qhttp="${QUESTDB_HTTP_PORT:-9000}"
  local batch="${QDB_BATCH:-0}"

  umask 0137 # 0640
  cat > "$FX_ENV" <<EOF
# OANDA credentials (practice or live)
OANDA_DEMO_ACCOUNT_ID=${acc}
OANDA_DEMO_KEY=${key}
# QuestDB connection
QUESTDB_HOST=${qhost}
QUESTDB_ILP_PORT=${qilp}
QUESTDB_HTTP_PORT=${qhttp}
# Ingest batching (0=immediate)
QDB_BATCH=${batch}
# Optional table overrides
# QDB_FX_TICKS_TABLE=fx_ticks
# QDB_FX_DOM_TABLE=fx_dom
EOF
  echo "Wrote $FX_ENV" >&2

  local symbols_default="BTCUSDT,ETHUSDT,XRPUSDT,BCHUSDT,LTCUSDT,BNBUSDT,ADAUSDT,BATUSDT,ETCUSDT,XLMUSDT,ZRXUSDT,DOGEUSDT,ATOMUSDT,DOTUSDT,LINKUSDT,UNIUSDT,SOLUSDT,AVAXUSDT,MATICUSDT,FILUSDT"
  local symbols="${SYMBOLS:-$symbols_default}"
  cat > "$CR_ENV" <<EOF
# QuestDB connection
QUESTDB_HOST=${qhost}
QUESTDB_ILP_PORT=${qilp}
QUESTDB_HTTP_PORT=${qhttp}
# Ingest batching for crypto
QDB_BATCH=${batch}
# Default Bitunix symbols (CSV)
SYMBOLS=${symbols}
EOF
  echo "Wrote $CR_ENV" >&2
}

health() {
  local base="http://${QUESTDB_HOST:-127.0.0.1}:${QUESTDB_HTTP_PORT:-9000}"
  has_cmd curl || { echo "curl not found" >&2; exit 1; }
  echo "- fx_ticks (30m):"; curl -s "${base}/exec?query=select%20count()%20from%20fx_ticks%20where%20ts%20%3E%20dateadd('m',-30,now())" || true; echo
  echo "- crypto_ticker (30m):"; curl -s "${base}/exec?query=select%20count()%20from%20crypto_ticker%20where%20ts%20%3E%20dateadd('m',-30,now())" || true; echo
  echo "- crypto_depth (30m):"; curl -s "${base}/exec?query=select%20count()%20from%20crypto_depth%20where%20ts%20%3E%20dateadd('m',-30,now())" || true; echo
}

setdepth() {
  need_root
  local level=${1:-}
  if [[ -z "$level" ]]; then echo "Usage: $0 setdepth <1|5|15>" >&2; exit 1; fi
  local batch_val=${QDB_BATCH:-200}
  mkdir -p /etc/systemd/system/bitunix-crypto-ingest.service.d
  cat > /etc/systemd/system/bitunix-crypto-ingest.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/python3 /root/rl-trader/streamer/bitunix_to_questdb.py --depth-level ${level} --batch ${batch_val}
EOF
  reload_daemon
  systemctl restart bitunix-crypto-ingest.service
  echo "Updated crypto depth level to ${level} and restarted." >&2
}

setsymbols() {
  need_root
  local csv=${1:-}
  if [[ -z "$csv" ]]; then echo "Usage: $0 setsymbols "BTCUSDT,ETHUSDT,..."" >&2; exit 1; fi
  if [[ ! -f "$CR_ENV" ]]; then echo "Missing $CR_ENV, run: $0 configure" >&2; exit 1; fi
  sed -i -E "s/^SYMBOLS=.*/SYMBOLS=${csv//\//\\/}/" "$CR_ENV"
  systemctl restart bitunix-crypto-ingest.service
  echo "Updated SYMBOLS in $CR_ENV and restarted crypto ingester." >&2
}

uninstall() {
  need_root
  stop_services all || true
  systemctl disable oanda-fx-ingest.service || true
  systemctl disable bitunix-crypto-ingest.service || true
  rm -f "$FX_UNIT_DST" "$CR_UNIT_DST"
  rm -rf /etc/systemd/system/bitunix-crypto-ingest.service.d || true
  reload_daemon
  echo "Units removed. Env files preserved: $FX_ENV $CR_ENV" >&2
}

case "${1:-}" in
  install) install_units; write_env_files;;
  enable) enable_services;;
  start) start_services "${2:-all}";;
  stop) stop_services "${2:-all}";;
  restart) restart_services "${2:-all}";;
  status) status_services "${2:-all}";;
  logs) shift; logs_service "${1:-all}" "${@:2}";;
  health) health;;
  configure) write_env_files;;
  setdepth) shift; setdepth "${1:-}";;
  setsymbols) shift; setsymbols "${1:-}";;
  uninstall) uninstall;;
  *)
    cat <<USAGE
Usage: $0 <command> [args]
  install            Install/overwrite units and write env files
  enable             Enable both services at boot
  start [fx|crypto]  Start services (default: both)
  stop [fx|crypto]   Stop services
  restart [fx|crypto] Restart services
  status [fx|crypto] Show systemd status
  logs [fx|crypto] [-f] [-n N] Tail logs
  health             Show recent row counts via QuestDB /exec
  configure          Re-write env files from current environment
  setdepth <1|5|15>  Set crypto depth level via override and restart
  setsymbols <CSV>   Update crypto SYMBOLS in env file and restart
  uninstall          Disable and remove unit files (keeps data/env)
USAGE
    exit 1
  ;;
esac
