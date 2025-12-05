package main

import (
	"context"
	"log"
	"os/signal"
	"syscall"
	"time"

	"localbroker/internal/config"
	"localbroker/internal/pricing"
	"localbroker/internal/server"
	"localbroker/internal/state"
	"localbroker/internal/storage"
)

func main() {
	cfg := config.Load()
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	priceSource := selectPriceSource(cfg)
	priceMgr := pricing.NewManager(priceSource, cfg.Instruments)
	store, err := storage.NewFileStore(cfg.DataDir)
	if err != nil {
		log.Fatalf("failed to init storage: %v", err)
	}
	engine := state.NewEngine(cfg, priceMgr, store)

	priceMgr.Start(ctx)
	go attachEngineToPrices(ctx, priceMgr, engine)

	srv := server.New(cfg, engine, priceMgr)
	if err := srv.Run(ctx); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func selectPriceSource(cfg config.Config) pricing.Source {
	if cfg.OandaAccountID != "" && cfg.OandaAPIKey != "" {
		log.Printf("using OANDA pricing stream for %s (%s)", cfg.AccountID, cfg.OandaEnv)
		return &pricing.OandaStreamSource{
			AccountID: cfg.OandaAccountID,
			APIKey:    cfg.OandaAPIKey,
			Env:       cfg.OandaEnv,
		}
	}
	log.Printf("no OANDA credentials detected; falling back to synthetic ticks")
	return &pricing.SyntheticSource{
		Seed:           time.Now().UnixNano(),
		UpdateInterval: 200 * time.Millisecond,
		Spread:         0.0002,
	}
}

func attachEngineToPrices(ctx context.Context, mgr *pricing.Manager, engine *state.Engine) {
	id, ch := mgr.Subscribe(1024)
	defer mgr.Unsubscribe(id)
	for {
		select {
		case <-ctx.Done():
			return
		case tick := <-ch:
			engine.ProcessTick(tick)
		}
	}
}
