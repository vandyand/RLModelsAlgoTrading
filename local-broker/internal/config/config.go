package config

import (
	"os"
	"strconv"
	"strings"
)

// Config captures runtime tunables for the local broker service.
type Config struct {
	ListenAddr      string
	AccountID       string
	BaseCurrency    string
	StartingBalance float64
	DataDir         string
	MaxLeverage     float64
	Instruments     []string

	// OANDA upstream settings (optional; falls back to synthetic prices when unset).
	OandaAccountID string
	OandaAPIKey    string
	OandaEnv       string
}

const defaultAccountID = "LB-0001"

// Load builds Config from environment variables with sane defaults.
func Load() Config {
	cfg := Config{
		ListenAddr:      getEnv("BROKER_LISTEN_ADDR", ":8080"),
		AccountID:       getEnv("BROKER_ACCOUNT_ID", defaultAccountID),
		BaseCurrency:    getEnv("BROKER_BASE_CURRENCY", "USD"),
		StartingBalance: parseFloat(getEnv("BROKER_START_BALANCE", "100000")),
		DataDir:         getEnv("BROKER_DATA_DIR", "./broker-data"),
		MaxLeverage:     parseFloat(getEnv("BROKER_MAX_LEVERAGE", "30")),
		Instruments:     parseInstruments(getEnv("BROKER_INSTRUMENTS", "")),
		OandaAccountID:  os.Getenv("OANDA_DEMO_ACCOUNT_ID"),
		OandaAPIKey:     os.Getenv("OANDA_DEMO_KEY"),
		OandaEnv:        getEnv("BROKER_OANDA_ENV", "practice"),
	}
	if len(cfg.Instruments) == 0 {
		cfg.Instruments = defaultInstrumentSet()
	}
	if cfg.MaxLeverage <= 0 {
		cfg.MaxLeverage = 30
	}
	return cfg
}

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

func parseFloat(val string) float64 {
	f, err := strconv.ParseFloat(val, 64)
	if err != nil {
		return 0
	}
	return f
}

func parseInstruments(csv string) []string {
	csv = strings.TrimSpace(csv)
	if csv == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.ToUpper(strings.TrimSpace(p)); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func defaultInstrumentSet() []string {
	return []string{
		"EUR_USD",
		"USD_JPY",
		"GBP_USD",
		"USD_CHF",
		"AUD_USD",
		"USD_CAD",
		"NZD_USD",
		"EUR_JPY",
		"GBP_JPY",
		"EUR_GBP",
		"EUR_CHF",
		"EUR_AUD",
		"EUR_CAD",
		"AUD_JPY",
		"CHF_JPY",
		"CAD_JPY",
		"NZD_JPY",
		"GBP_CHF",
		"GBP_CAD",
		"AUD_CAD",
	}
}
