package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"localbroker/internal/config"
	"localbroker/internal/pricing"
	"localbroker/internal/state"
)

// Server exposes an OANDA-like REST interface backed by the Engine.
type Server struct {
	cfg    config.Config
	engine *state.Engine
	price  *pricing.Manager
}

// New wires dependencies.
func New(cfg config.Config, engine *state.Engine, price *pricing.Manager) *Server {
	return &Server{
		cfg:    cfg,
		engine: engine,
		price:  price,
	}
}

// Run starts serving HTTP until the context is cancelled.
func (s *Server) Run(ctx context.Context) error {
	router := s.routes()
	httpServer := &http.Server{
		Addr:    s.cfg.ListenAddr,
		Handler: router,
	}
	errCh := make(chan error, 1)
	go func() {
		log.Printf("local broker listening on %s", s.cfg.ListenAddr)
		errCh <- httpServer.ListenAndServe()
	}()
	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
		return nil
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}
}

func (s *Server) routes() http.Handler {
	r := chi.NewRouter()
	r.Get("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	r.Route("/v3", func(r chi.Router) {
		r.Route("/accounts", func(r chi.Router) {
			r.Get("/", s.handleAccounts)
			r.Post("/", s.handleCreateAccount)
			r.Route("/{accountID}", func(r chi.Router) {
				r.Get("/summary", s.handleSummary)
				r.Get("/openPositions", s.handleOpenPositions)
				r.Get("/transactions", s.handleTransactions)
				r.Get("/equity", s.handleEquityHistory)
				r.Get("/pricing", s.handlePricing)
				r.Get("/pricing/stream", s.handlePricingStream)
				r.Post("/orders", s.handleCreateOrder)
			})
		})
	})
	return r
}

func (s *Server) handleAccounts(w http.ResponseWriter, _ *http.Request) {
	payload := map[string]any{
		"accounts": s.engine.Accounts(),
	}
	writeJSON(w, http.StatusOK, payload)
}

func (s *Server) handleSummary(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	sum, err := s.engine.AccountSummary(accountID)
	if err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}
	resp := map[string]any{
		"account": map[string]any{
			"id":              sum.AccountID,
			"currency":        sum.Currency,
			"balance":         formatAmount(sum.Balance),
			"NAV":             formatAmount(sum.NAV),
			"unrealizedPL":    formatAmount(sum.UnrealizedPL),
			"openTradeCount":  sum.OpenTradeCount,
			"marginAvailable": formatAmount(sum.NAV), // placeholder until margin logic added
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleCreateAccount(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID       string  `json:"id"`
		Currency string  `json:"currency"`
		Balance  float64 `json:"balance"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid json: %w", err))
		return
	}
	info, err := s.engine.CreateAccount(req.ID, req.Currency, req.Balance)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"account": info})
}

func (s *Server) handleOpenPositions(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	positions, err := s.engine.OpenPositionsView(accountID)
	if err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}
	respPositions := make([]any, 0, len(positions))
	for _, pos := range positions {
		net := map[string]string{
			"units":        fmt.Sprintf("%d", pos.Units),
			"averagePrice": formatAmount(pos.AveragePrice),
			"unrealizedPL": formatAmount(pos.UnrealizedPL),
		}
		entry := map[string]any{
			"instrument":      pos.Instrument,
			"net":             net,
			"netUnrealizedPL": formatAmount(pos.UnrealizedPL),
		}
		if pos.TakeProfit != nil {
			entry["takeProfit"] = formatAmount(*pos.TakeProfit)
		}
		if pos.StopLoss != nil {
			entry["stopLoss"] = formatAmount(*pos.StopLoss)
		}
		if pos.TrailingStop != nil {
			entry["trailingStop"] = map[string]string{
				"distance": formatAmount(pos.TrailingStop.Distance),
				"stop":     formatAmount(pos.TrailingStop.Stop),
			}
		}
		respPositions = append(respPositions, entry)
	}
	writeJSON(w, http.StatusOK, map[string]any{"positions": respPositions})
}

func (s *Server) handleTransactions(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	limit := parseLimit(r, 100)
	txns, err := s.engine.TransactionHistory(accountID, limit)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	resp := make([]any, 0, len(txns))
	for _, tx := range txns {
		payload := formatTransactionPayload(&tx)
		resp = append(resp, payload)
	}
	writeJSON(w, http.StatusOK, map[string]any{"transactions": resp})
}

func (s *Server) handleEquityHistory(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	limit := parseLimit(r, 500)
	points, err := s.engine.EquityHistory(accountID, limit)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	resp := make([]any, 0, len(points))
	for _, pt := range points {
		resp = append(resp, map[string]string{
			"time":    pt.Time.Format(time.RFC3339Nano),
			"balance": formatAmount(pt.Balance),
			"nav":     formatAmount(pt.NAV),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"equity": resp})
}

func (s *Server) handlePricing(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	if _, err := s.engine.AccountSummary(accountID); err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}
	instruments := parseInstrumentsQuery(r.URL.Query().Get("instruments"))
	ticks := s.price.Snapshot(instruments)
	prices := make([]any, 0, len(ticks))
	for _, tick := range ticks {
		prices = append(prices, map[string]any{
			"instrument":  tick.Instrument,
			"status":      statusOrDefault(tick.Status),
			"time":        tick.Time.Format(time.RFC3339Nano),
			"bids":        formatBuckets(tick.Bids),
			"asks":        formatBuckets(tick.Asks),
			"closeoutBid": formatAmount(tick.CloseoutBid),
			"closeoutAsk": formatAmount(tick.CloseoutAsk),
		})
	}
	resp := map[string]any{
		"prices": prices,
		"time":   time.Now().UTC().Format(time.RFC3339Nano),
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handlePricingStream(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	if _, err := s.engine.AccountSummary(accountID); err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("streaming not supported"))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	subID, ch := s.price.Subscribe(256)
	defer s.price.Unsubscribe(subID)

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case tick, ok := <-ch:
			if !ok {
				return
			}
			payload := map[string]any{
				"instrument":  tick.Instrument,
				"status":      statusOrDefault(tick.Status),
				"time":        tick.Time.Format(time.RFC3339Nano),
				"bids":        formatBuckets(tick.Bids),
				"asks":        formatBuckets(tick.Asks),
				"closeoutBid": formatAmount(tick.CloseoutBid),
				"closeoutAsk": formatAmount(tick.CloseoutAsk),
			}
			data, err := json.Marshal(payload)
			if err != nil {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func (s *Server) handleCreateOrder(w http.ResponseWriter, r *http.Request) {
	accountID := chi.URLParam(r, "accountID")
	var payload state.OrderPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid json: %w", err))
		return
	}
	res, err := s.engine.PlaceOrder(accountID, payload)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	resp := map[string]any{
		"relatedTransactionIDs": []string{},
	}
	switch {
	case res.Filled != nil:
		resp["orderFillTransaction"] = formatTransactionPayload(res.Filled)
		resp["relatedTransactionIDs"] = []string{res.Filled.ID}
		resp["lastTransactionID"] = res.Filled.ID
		writeJSON(w, http.StatusCreated, resp)
	case res.Pending != nil:
		resp["pendingOrder"] = res.Pending
		resp["lastTransactionID"] = res.Pending.ID
		writeJSON(w, http.StatusAccepted, resp)
	default:
		writeJSON(w, http.StatusOK, resp)
	}
}

func parseInstrumentsQuery(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if inst := strings.ToUpper(strings.TrimSpace(p)); inst != "" {
			out = append(out, inst)
		}
	}
	return out
}

func parseLimit(r *http.Request, defaultVal int) int {
	val := strings.TrimSpace(r.URL.Query().Get("limit"))
	if val == "" {
		return defaultVal
	}
	if n, err := strconv.Atoi(val); err == nil && n > 0 {
		return n
	}
	return defaultVal
}

func formatAmount(val float64) string {
	return fmt.Sprintf("%.5f", val)
}

func statusOrDefault(status string) string {
	if status != "" {
		return status
	}
	return "tradeable"
}

func formatBuckets(buckets []pricing.PriceBucket) []map[string]any {
	out := make([]map[string]any, 0, len(buckets))
	for _, b := range buckets {
		out = append(out, map[string]any{
			"price":     formatAmount(b.Price),
			"liquidity": b.Liquidity,
		})
	}
	return out
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]any{
		"error": err.Error(),
	})
}

func formatTransactionPayload(tx *state.Transaction) map[string]any {
	if tx == nil {
		return nil
	}
	return map[string]any{
		"id":             tx.ID,
		"time":           tx.Time.Format(time.RFC3339Nano),
		"type":           tx.Type,
		"instrument":     tx.Instrument,
		"units":          fmt.Sprintf("%d", tx.Units),
		"price":          formatAmount(tx.Price),
		"pl":             formatAmount(tx.RealizedPL),
		"accountBalance": formatAmount(tx.AccountBalance),
		"reason":         tx.Reason,
	}
}
