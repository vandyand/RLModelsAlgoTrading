package state

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"localbroker/internal/config"
	"localbroker/internal/pricing"
)

// PriceFeed exposes read-only access to latest ticks.
type PriceFeed interface {
	Latest(instrument string) (pricing.Tick, bool)
}

// Persistor defines the persistence hooks used by the engine.
type Persistor interface {
	Load() (SerializedEngine, error)
	Save(SerializedEngine) error
	AppendEquity(accountID string, point EquityPoint) error
	ReadEquity(accountID string, limit int) ([]EquityPoint, error)
}

// Engine keeps track of accounts, orders, and risk automation.
type Engine struct {
	cfg       config.Config
	priceFeed PriceFeed
	persistor Persistor

	mu       sync.RWMutex
	accounts map[string]*Account
}

// NewEngine seeds the ledger, optionally restoring from persistence.
func NewEngine(cfg config.Config, feed PriceFeed, persistor Persistor) *Engine {
	engine := &Engine{
		cfg:       cfg,
		priceFeed: feed,
		persistor: persistor,
		accounts:  make(map[string]*Account),
	}
	if persistor != nil {
		if snapshot, err := persistor.Load(); err == nil && len(snapshot.Accounts) > 0 {
			engine.restore(snapshot)
		} else if err != nil {
			log.Printf("state: failed to load snapshot: %v", err)
		}
	}
	if len(engine.accounts) == 0 && cfg.AccountID != "" {
		engine.accounts[cfg.AccountID] = newAccount(cfg.AccountID, cfg.BaseCurrency, cfg.StartingBalance)
	}
	return engine
}

// Accounts returns lightweight metadata for every account.
func (e *Engine) Accounts() []AccountInfo {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]AccountInfo, 0, len(e.accounts))
	for _, acct := range e.accounts {
		out = append(out, AccountInfo{
			ID:       acct.ID,
			Currency: acct.Currency,
		})
	}
	return out
}

// CreateAccount adds a new account at runtime.
func (e *Engine) CreateAccount(id, currency string, balance float64) (AccountInfo, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return AccountInfo{}, errors.New("account id required")
	}
	if balance < 0 {
		return AccountInfo{}, errors.New("starting balance must be non-negative")
	}
	if currency == "" {
		currency = e.cfg.BaseCurrency
	}
	e.mu.Lock()
	if _, exists := e.accounts[id]; exists {
		e.mu.Unlock()
		return AccountInfo{}, fmt.Errorf("account %s already exists", id)
	}
	acct := newAccount(id, currency, balance)
	e.accounts[id] = acct
	snapshot := e.snapshotLocked()
	equity := e.buildEquitySnapshotsLocked()
	e.mu.Unlock()

	e.persistState(snapshot)
	e.persistEquitySnapshots(equity)
	return AccountInfo{ID: id, Currency: currency}, nil
}

// AccountSummary computes real-time balance + NAV info.
func (e *Engine) AccountSummary(accountID string) (AccountSummary, error) {
	e.mu.RLock()
	acct, ok := e.accounts[accountID]
	e.mu.RUnlock()
	if !ok {
		return AccountSummary{}, errors.New("account not found")
	}
	unrealized := e.computeUnrealized(accountID)
	return AccountSummary{
		AccountID:      acct.ID,
		Currency:       acct.Currency,
		Balance:        acct.Balance,
		UnrealizedPL:   unrealized,
		NAV:            acct.Balance + unrealized,
		OpenTradeCount: acct.openTradeCount(),
	}, nil
}

// OpenPositionsView returns normalized view objects for REST.
func (e *Engine) OpenPositionsView(accountID string) ([]PositionView, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	acct, ok := e.accounts[accountID]
	if !ok {
		return nil, errors.New("account not found")
	}
	views := make([]PositionView, 0, len(acct.Positions))
	for _, pos := range acct.Positions {
		if pos.Units == 0 {
			continue
		}
		unrealized := e.unrealizedForPosition(pos)
		view := PositionView{
			Instrument:   pos.Instrument,
			Units:        pos.Units,
			AveragePrice: pos.AveragePrice,
			UnrealizedPL: unrealized,
			TakeProfit:   valueOrNil(pos.TakeProfit),
			StopLoss:     valueOrNil(pos.StopLoss),
		}
		if pos.Trailing != nil {
			view.TrailingStop = &TrailingView{
				Distance: pos.Trailing.Distance,
				Stop:     pos.Trailing.StopPrice,
			}
		}
		views = append(views, view)
	}
	return views, nil
}

// TransactionHistory returns the most recent transactions for an account.
func (e *Engine) TransactionHistory(accountID string, limit int) ([]Transaction, error) {
	e.mu.RLock()
	acct, ok := e.accounts[accountID]
	if !ok {
		e.mu.RUnlock()
		return nil, errors.New("account not found")
	}
	total := len(acct.Transactions)
	if limit <= 0 || limit > total {
		limit = total
	}
	out := make([]Transaction, limit)
	copy(out, acct.Transactions[total-limit:])
	e.mu.RUnlock()
	return out, nil
}

// EquityHistory loads persisted equity samples for dashboards.
func (e *Engine) EquityHistory(accountID string, limit int) ([]EquityPoint, error) {
	if e.persistor == nil {
		return nil, errors.New("persistence disabled")
	}
	e.mu.RLock()
	_, ok := e.accounts[accountID]
	e.mu.RUnlock()
	if !ok {
		return nil, errors.New("account not found")
	}
	if limit <= 0 {
		limit = 500
	}
	return e.persistor.ReadEquity(accountID, limit)
}

func valueOrNil(ptr *float64) *float64 {
	if ptr == nil {
		return nil
	}
	val := *ptr
	return &val
}

// ProcessTick should be called whenever new market data arrives.
func (e *Engine) ProcessTick(tick pricing.Tick) {
	e.mu.Lock()
	dirty := false
	for _, acct := range e.accounts {
		pos, ok := acct.Positions[tick.Instrument]
		if !ok || pos.Units == 0 {
			continue
		}
		if e.evaluateRisk(acct, pos, tick) {
			dirty = true
		}
	}
	for _, acct := range e.accounts {
		if e.evaluatePendingOrdersLocked(acct, tick) {
			dirty = true
		}
	}
	var snapshot SerializedEngine
	var equities []equityRecord
	if e.persistor != nil {
		equities = e.buildEquitySnapshotsLocked()
		if dirty {
			snapshot = e.snapshotLocked()
		}
	}
	e.mu.Unlock()

	if dirty {
		e.persistState(snapshot)
	}
	e.persistEquitySnapshots(equities)
}

func (e *Engine) evaluateRisk(acct *Account, pos *Position, tick pricing.Tick) bool {
	if pos.Units > 0 {
		bid, ok := tick.BestBid()
		if !ok {
			return false
		}
		if pos.TakeProfit != nil && bid >= *pos.TakeProfit {
			return e.closePosition(acct, pos, bid, "TAKE_PROFIT")
		}
		if pos.StopLoss != nil && bid <= *pos.StopLoss {
			return e.closePosition(acct, pos, bid, "STOP_LOSS")
		}
		if pos.Trailing != nil {
			if bid > pos.Trailing.Reference {
				pos.Trailing.Reference = bid
			}
			pos.Trailing.StopPrice = pos.Trailing.Reference - pos.Trailing.Distance
			if bid <= pos.Trailing.StopPrice {
				return e.closePosition(acct, pos, bid, "TRAILING_STOP_LOSS")
			}
		}
		return false
	}
	// Short position.
	ask, ok := tick.BestAsk()
	if !ok {
		return false
	}
	if pos.TakeProfit != nil && ask <= *pos.TakeProfit {
		return e.closePosition(acct, pos, ask, "TAKE_PROFIT")
	}
	if pos.StopLoss != nil && ask >= *pos.StopLoss {
		return e.closePosition(acct, pos, ask, "STOP_LOSS")
	}
	if pos.Trailing != nil {
		if ask < pos.Trailing.Reference {
			pos.Trailing.Reference = ask
		}
		pos.Trailing.StopPrice = pos.Trailing.Reference + pos.Trailing.Distance
		if ask >= pos.Trailing.StopPrice {
			return e.closePosition(acct, pos, ask, "TRAILING_STOP_LOSS")
		}
	}
	return false
}

func (e *Engine) evaluatePendingOrdersLocked(acct *Account, tick pricing.Tick) bool {
	if len(acct.PendingOrders) == 0 {
		return false
	}
	remaining := acct.PendingOrders[:0]
	dirty := false
	for _, ord := range acct.PendingOrders {
		if ord == nil || ord.Instrument != tick.Instrument {
			remaining = append(remaining, ord)
			continue
		}
		if !shouldTriggerOrder(ord.Type, ord.Units, ord.Price, tick) {
			remaining = append(remaining, ord)
			continue
		}
		fillPrice, err := priceForSide(tick, ord.Units)
		if err != nil {
			remaining = append(remaining, ord)
			continue
		}
		if _, err := e.applyFillLocked(acct, ord.Instrument, ord.Units, fillPrice, ord.Risk, fmt.Sprintf("%s_ORDER", strings.ToUpper(ord.Type))); err != nil {
			remaining = append(remaining, ord)
			continue
		}
		dirty = true
	}
	acct.PendingOrders = remaining
	return dirty
}

// PlaceOrder currently supports MARKET, LIMIT, STOP orders with optional TP/SL/TSL instructions.
func (e *Engine) PlaceOrder(accountID string, payload OrderPayload) (OrderResult, error) {
	order := payload.Order
	if order.Type == "" {
		return OrderResult{}, errors.New("missing order type")
	}
	instrument := strings.ToUpper(order.Instrument)
	if instrument == "" {
		return OrderResult{}, errors.New("missing instrument")
	}
	units, err := strconv.ParseInt(order.Units, 10, 64)
	if err != nil || units == 0 {
		return OrderResult{}, errors.New("units must be non-zero integer string")
	}

	tick, ok := e.priceFeed.Latest(instrument)
	if !ok {
		return OrderResult{}, fmt.Errorf("instrument %s has no pricing data yet", instrument)
	}
	price, err := priceForSide(tick, units)
	if err != nil {
		return OrderResult{}, err
	}

	risk, err := extractRisk(order)
	if err != nil {
		return OrderResult{}, err
	}

	e.mu.Lock()
	acct, ok := e.accounts[accountID]
	if !ok {
		e.mu.Unlock()
		return OrderResult{}, errors.New("account not found")
	}

	orderType := strings.ToUpper(order.Type)
	switch orderType {
	case "MARKET":
		res, snap, equities, err := e.executeMarketOrderLocked(acct, instrument, units, price, risk)
		if err != nil {
			e.mu.Unlock()
			return OrderResult{}, err
		}
		e.mu.Unlock()
		e.persistState(snap)
		e.persistEquitySnapshots(equities)
		return res, nil
	case "LIMIT", "STOP":
		targetPrice, err := strconv.ParseFloat(strings.TrimSpace(order.Price), 64)
		if err != nil || targetPrice <= 0 {
			e.mu.Unlock()
			return OrderResult{}, errors.New("limit/stop orders require positive price field")
		}
		res, snap, equities, err := e.enqueuePendingOrderLocked(acct, orderType, instrument, units, targetPrice, risk, tick)
		if err != nil {
			e.mu.Unlock()
			return OrderResult{}, err
		}
		e.mu.Unlock()
		e.persistState(snap)
		e.persistEquitySnapshots(equities)
		return res, nil
	default:
		e.mu.Unlock()
		return OrderResult{}, fmt.Errorf("order type %s not supported", order.Type)
	}
}

func (e *Engine) executeMarketOrderLocked(acct *Account, instrument string, units int64, price float64, risk RiskSettings) (OrderResult, SerializedEngine, []equityRecord, error) {
	tx, err := e.applyFillLocked(acct, instrument, units, price, risk, "MARKET_ORDER")
	if err != nil {
		return OrderResult{}, SerializedEngine{}, nil, err
	}
	var snapshot SerializedEngine
	var equities []equityRecord
	if e.persistor != nil {
		snapshot = e.snapshotLocked()
		equities = e.buildEquitySnapshotsLocked()
	}
	return OrderResult{Filled: tx}, snapshot, equities, nil
}

func (e *Engine) enqueuePendingOrderLocked(acct *Account, orderType, instrument string, units int64, targetPrice float64, risk RiskSettings, tick pricing.Tick) (OrderResult, SerializedEngine, []equityRecord, error) {
	if shouldTriggerOrder(orderType, units, targetPrice, tick) {
		return e.executeMarketOrderLocked(acct, instrument, units, targetPrice, risk)
	}
	if err := e.ensureMarginLocked(acct, instrument, units, targetPrice); err != nil {
		return OrderResult{}, SerializedEngine{}, nil, err
	}
	pending := &PendingOrder{
		ID:         acct.nextOrderID(),
		Type:       strings.ToUpper(orderType),
		Instrument: instrument,
		Units:      units,
		Price:      targetPrice,
		Time:       time.Now().UTC(),
		Risk:       cloneRiskSettings(risk),
	}
	acct.PendingOrders = append(acct.PendingOrders, pending)
	var snapshot SerializedEngine
	if e.persistor != nil {
		snapshot = e.snapshotLocked()
	}
	return OrderResult{Pending: pending}, snapshot, nil, nil
}

func (e *Engine) applyFillLocked(acct *Account, instrument string, units int64, price float64, risk RiskSettings, reason string) (*Transaction, error) {
	if err := e.ensureMarginLocked(acct, instrument, units, price); err != nil {
		return nil, err
	}
	pos := acct.position(instrument)
	prevUnits := pos.Units
	transaction := Transaction{
		ID:         acct.nextTransactionID(),
		Time:       time.Now().UTC(),
		Type:       "ORDER_FILL",
		Instrument: instrument,
		Units:      units,
		Price:      price,
		Reason:     reason,
	}

	realized := pos.applyFill(units, price)
	acct.Balance += realized
	transaction.RealizedPL = realized
	transaction.AccountBalance = acct.Balance
	acct.Transactions = append(acct.Transactions, transaction)

	if pos.Units == 0 {
		pos.TakeProfit = nil
		pos.StopLoss = nil
		pos.Trailing = nil
	} else if sameSign(prevUnits, pos.Units) && sameSign(prevUnits, units) {
		pos.applyRiskSettings(risk, price)
	} else if !sameSign(prevUnits, pos.Units) {
		pos.applyRiskSettings(risk, price)
	}
	return &transaction, nil
}

func priceForSide(tick pricing.Tick, units int64) (float64, error) {
	if units > 0 {
		if price, ok := tick.BestAsk(); ok {
			return price, nil
		}
		if mid, ok := tick.MidPrice(); ok {
			return mid, nil
		}
		return 0, errors.New("missing ask price")
	}
	if price, ok := tick.BestBid(); ok {
		return price, nil
	}
	if mid, ok := tick.MidPrice(); ok {
		return mid, nil
	}
	return 0, errors.New("missing bid price")
}

func shouldTriggerOrder(orderType string, units int64, target float64, tick pricing.Tick) bool {
	orderType = strings.ToUpper(orderType)
	switch orderType {
	case "LIMIT":
		if units > 0 {
			if ask, ok := tick.BestAsk(); ok && ask <= target {
				return true
			}
		} else {
			if bid, ok := tick.BestBid(); ok && bid >= target {
				return true
			}
		}
	case "STOP":
		if units > 0 {
			if ask, ok := tick.BestAsk(); ok && ask >= target {
				return true
			}
		} else {
			if bid, ok := tick.BestBid(); ok && bid <= target {
				return true
			}
		}
	}
	return false
}

func (e *Engine) closePosition(acct *Account, pos *Position, price float64, reason string) bool {
	if pos.Units == 0 {
		return false
	}
	units := -pos.Units // closing trade direction
	realized := pos.applyFill(units, price)
	acct.Balance += realized
	tx := Transaction{
		ID:              acct.nextTransactionID(),
		Time:            time.Now().UTC(),
		Type:            reason,
		Instrument:      pos.Instrument,
		Units:           units,
		Price:           price,
		RealizedPL:      realized,
		AccountBalance:  acct.Balance,
		RiskCloseReason: reason,
	}
	acct.Transactions = append(acct.Transactions, tx)
	pos.TakeProfit = nil
	pos.StopLoss = nil
	pos.Trailing = nil
	return true
}

func (e *Engine) computeUnrealized(accountID string) float64 {
	e.mu.RLock()
	acct, ok := e.accounts[accountID]
	var total float64
	if ok {
		total = e.unrealizedForAccountLocked(acct)
	}
	e.mu.RUnlock()
	return total
}

func (e *Engine) unrealizedForAccountLocked(acct *Account) float64 {
	total := 0.0
	for _, pos := range acct.Positions {
		total += e.unrealizedForPosition(pos)
	}
	return total
}

func (e *Engine) unrealizedForPosition(pos *Position) float64 {
	if pos.Units == 0 {
		return 0
	}
	tick, ok := e.priceFeed.Latest(pos.Instrument)
	if !ok {
		return 0
	}
	if pos.Units > 0 {
		price, ok := tick.BestBid()
		if !ok {
			return 0
		}
		return float64(pos.Units) * (price - pos.AveragePrice)
	}
	price, ok := tick.BestAsk()
	if !ok {
		return 0
	}
	return float64(-pos.Units) * (pos.AveragePrice - price)
}

// Supporting models.

type Account struct {
	ID        string
	Currency  string
	Balance   float64
	Positions map[string]*Position

	Transactions      []Transaction
	PendingOrders     []*PendingOrder
	transactionSerial int64
	orderSerial       int64
}

func newAccount(id, currency string, balance float64) *Account {
	if currency == "" {
		currency = "USD"
	}
	return &Account{
		ID:            id,
		Currency:      currency,
		Balance:       balance,
		Positions:     make(map[string]*Position),
		PendingOrders: make([]*PendingOrder, 0),
	}
}

func (a *Account) position(instrument string) *Position {
	pos, ok := a.Positions[instrument]
	if !ok {
		pos = &Position{Instrument: instrument}
		a.Positions[instrument] = pos
	}
	return pos
}

func (a *Account) nextTransactionID() string {
	a.transactionSerial++
	return fmt.Sprintf("%d", a.transactionSerial)
}

func (a *Account) nextOrderID() string {
	a.orderSerial++
	return fmt.Sprintf("O-%d", a.orderSerial)
}

func (a *Account) openTradeCount() int {
	count := 0
	for _, pos := range a.Positions {
		if pos.Units != 0 {
			count++
		}
	}
	return count
}

type Position struct {
	Instrument   string
	Units        int64
	AveragePrice float64
	RealizedPL   float64
	TakeProfit   *float64
	StopLoss     *float64
	Trailing     *TrailingStop
}

func (p *Position) applyFill(units int64, price float64) float64 {
	if units == 0 {
		return 0
	}
	realized := 0.0
	if p.Units == 0 || sameSign(p.Units, units) {
		totalUnits := float64(abs64(p.Units) + abs64(units))
		weighted := p.AveragePrice*float64(abs64(p.Units)) + price*float64(abs64(units))
		p.AveragePrice = weighted / totalUnits
		p.Units += units
		return 0
	}

	closing := min(abs64(p.Units), abs64(units))
	if p.Units > 0 {
		realized += float64(closing) * (price - p.AveragePrice)
	} else {
		realized += float64(closing) * (p.AveragePrice - price)
	}
	p.Units += units
	if p.Units == 0 {
		p.AveragePrice = 0
	} else if (p.Units > 0 && units > 0) || (p.Units < 0 && units < 0) {
		// Increased exposure after partial close; average price becomes fill price.
		p.AveragePrice = price
	}
	return realized
}

func (p *Position) applyRiskSettings(risk RiskSettings, fillPrice float64) {
	if risk.TakeProfit != nil {
		val := *risk.TakeProfit
		p.TakeProfit = &val
	}
	if risk.StopLoss != nil {
		val := *risk.StopLoss
		p.StopLoss = &val
	}
	if risk.TrailingDistance != nil && *risk.TrailingDistance > 0 {
		dist := *risk.TrailingDistance
		p.Trailing = &TrailingStop{
			Distance:  dist,
			Reference: fillPrice,
		}
		if p.Units > 0 {
			p.Trailing.StopPrice = fillPrice - dist
		} else {
			p.Trailing.StopPrice = fillPrice + dist
		}
	} else if risk.TrailingDistance == nil {
		// leave existing trailing unchanged
	}
}

type TrailingStop struct {
	Distance  float64
	Reference float64
	StopPrice float64
}

type AccountInfo struct {
	ID       string `json:"id"`
	Currency string `json:"currency"`
}

type AccountSummary struct {
	AccountID      string  `json:"accountID"`
	Currency       string  `json:"currency"`
	Balance        float64 `json:"balance"`
	UnrealizedPL   float64 `json:"unrealizedPL"`
	NAV            float64 `json:"nav"`
	OpenTradeCount int     `json:"openTradeCount"`
}

type PositionView struct {
	Instrument   string        `json:"instrument"`
	Units        int64         `json:"units"`
	AveragePrice float64       `json:"averagePrice"`
	UnrealizedPL float64       `json:"unrealizedPL"`
	TakeProfit   *float64      `json:"takeProfit"`
	StopLoss     *float64      `json:"stopLoss"`
	TrailingStop *TrailingView `json:"trailingStop,omitempty"`
}

type TrailingView struct {
	Distance float64 `json:"distance"`
	Stop     float64 `json:"stop"`
}

type Transaction struct {
	ID              string    `json:"id"`
	Time            time.Time `json:"time"`
	Type            string    `json:"type"`
	Instrument      string    `json:"instrument"`
	Units           int64     `json:"units"`
	Price           float64   `json:"price"`
	RealizedPL      float64   `json:"realizedPL"`
	AccountBalance  float64   `json:"accountBalance"`
	Reason          string    `json:"reason"`
	RiskCloseReason string    `json:"riskCloseReason,omitempty"`
}

type PendingOrder struct {
	ID         string       `json:"id"`
	Type       string       `json:"type"`
	Instrument string       `json:"instrument"`
	Units      int64        `json:"units"`
	Price      float64      `json:"price"`
	Time       time.Time    `json:"time"`
	Risk       RiskSettings `json:"risk"`
}

type OrderResult struct {
	Filled  *Transaction
	Pending *PendingOrder
}

// OrderPayload matches the subset of OANDA POST /orders we need.
type OrderPayload struct {
	Order OrderDetails `json:"order"`
}

type OrderDetails struct {
	Type               string               `json:"type"`
	Instrument         string               `json:"instrument"`
	Units              string               `json:"units"`
	TimeInForce        string               `json:"timeInForce"`
	PositionFill       string               `json:"positionFill"`
	TakeProfitOnFill   *PriceInstruction    `json:"takeProfitOnFill"`
	StopLossOnFill     *PriceInstruction    `json:"stopLossOnFill"`
	TrailingStopOnFill *DistanceInstruction `json:"trailingStopLossOnFill"`
	Price              string               `json:"price"`
}

type PriceInstruction struct {
	Price string `json:"price"`
}

type DistanceInstruction struct {
	Distance string `json:"distance"`
}

type RiskSettings struct {
	TakeProfit       *float64 `json:"takeProfit,omitempty"`
	StopLoss         *float64 `json:"stopLoss,omitempty"`
	TrailingDistance *float64 `json:"trailingDistance,omitempty"`
}

func extractRisk(order OrderDetails) (RiskSettings, error) {
	var risk RiskSettings
	parsePrice := func(inst *PriceInstruction) (*float64, error) {
		if inst == nil || inst.Price == "" {
			return nil, nil
		}
		val, err := strconv.ParseFloat(inst.Price, 64)
		if err != nil {
			return nil, err
		}
		return &val, nil
	}
	parseDistance := func(inst *DistanceInstruction) (*float64, error) {
		if inst == nil || inst.Distance == "" {
			return nil, nil
		}
		val, err := strconv.ParseFloat(inst.Distance, 64)
		if err != nil {
			return nil, err
		}
		if val <= 0 {
			return nil, errors.New("trailing distance must be positive")
		}
		return &val, nil
	}
	var err error
	if risk.TakeProfit, err = parsePrice(order.TakeProfitOnFill); err != nil {
		return risk, err
	}
	if risk.StopLoss, err = parsePrice(order.StopLossOnFill); err != nil {
		return risk, err
	}
	if risk.TrailingDistance, err = parseDistance(order.TrailingStopOnFill); err != nil {
		return risk, err
	}
	return risk, nil
}

// Persistence + serialization helpers.

type EquityPoint struct {
	Time    time.Time `json:"time"`
	Balance float64   `json:"balance"`
	NAV     float64   `json:"nav"`
}

type SerializedEngine struct {
	Timestamp time.Time           `json:"timestamp"`
	Accounts  []SerializedAccount `json:"accounts"`
}

type SerializedAccount struct {
	ID                string               `json:"id"`
	Currency          string               `json:"currency"`
	Balance           float64              `json:"balance"`
	TransactionSerial int64                `json:"transactionSerial"`
	OrderSerial       int64                `json:"orderSerial"`
	Positions         []SerializedPosition `json:"positions"`
	Transactions      []Transaction        `json:"transactions"`
	PendingOrders     []SerializedOrder    `json:"pendingOrders"`
}

type SerializedPosition struct {
	Instrument   string        `json:"instrument"`
	Units        int64         `json:"units"`
	AveragePrice float64       `json:"averagePrice"`
	TakeProfit   *float64      `json:"takeProfit,omitempty"`
	StopLoss     *float64      `json:"stopLoss,omitempty"`
	Trailing     *TrailingStop `json:"trailing,omitempty"`
}

type SerializedOrder struct {
	ID         string       `json:"id"`
	Type       string       `json:"type"`
	Instrument string       `json:"instrument"`
	Units      int64        `json:"units"`
	Price      float64      `json:"price"`
	Time       time.Time    `json:"time"`
	Risk       RiskSettings `json:"risk"`
}

type equityRecord struct {
	AccountID string
	Point     EquityPoint
}

func (e *Engine) snapshotLocked() SerializedEngine {
	snap := SerializedEngine{
		Timestamp: time.Now().UTC(),
		Accounts:  make([]SerializedAccount, 0, len(e.accounts)),
	}
	for _, acct := range e.accounts {
		accountSnap := SerializedAccount{
			ID:                acct.ID,
			Currency:          acct.Currency,
			Balance:           acct.Balance,
			TransactionSerial: acct.transactionSerial,
			OrderSerial:       acct.orderSerial,
			Transactions:      append([]Transaction(nil), acct.Transactions...),
			Positions:         make([]SerializedPosition, 0, len(acct.Positions)),
			PendingOrders:     make([]SerializedOrder, 0, len(acct.PendingOrders)),
		}
		for _, pos := range acct.Positions {
			if pos == nil {
				continue
			}
			ser := SerializedPosition{
				Instrument:   pos.Instrument,
				Units:        pos.Units,
				AveragePrice: pos.AveragePrice,
			}
			ser.TakeProfit = copyFloatPtr(pos.TakeProfit)
			ser.StopLoss = copyFloatPtr(pos.StopLoss)
			if pos.Trailing != nil {
				ts := *pos.Trailing
				ser.Trailing = &ts
			}
			accountSnap.Positions = append(accountSnap.Positions, ser)
		}
		for _, ord := range acct.PendingOrders {
			if ord == nil {
				continue
			}
			accountSnap.PendingOrders = append(accountSnap.PendingOrders, SerializedOrder{
				ID:         ord.ID,
				Type:       ord.Type,
				Instrument: ord.Instrument,
				Units:      ord.Units,
				Price:      ord.Price,
				Time:       ord.Time,
				Risk:       cloneRiskSettings(ord.Risk),
			})
		}
		snap.Accounts = append(snap.Accounts, accountSnap)
	}
	return snap
}

func (e *Engine) restore(snapshot SerializedEngine) {
	e.accounts = make(map[string]*Account, len(snapshot.Accounts))
	for _, acctSnap := range snapshot.Accounts {
		acct := newAccount(acctSnap.ID, acctSnap.Currency, acctSnap.Balance)
		acct.transactionSerial = acctSnap.TransactionSerial
		acct.orderSerial = acctSnap.OrderSerial
		acct.Transactions = append([]Transaction(nil), acctSnap.Transactions...)
		for _, posSnap := range acctSnap.Positions {
			pos := &Position{
				Instrument:   posSnap.Instrument,
				Units:        posSnap.Units,
				AveragePrice: posSnap.AveragePrice,
			}
			pos.TakeProfit = copyFloatPtr(posSnap.TakeProfit)
			pos.StopLoss = copyFloatPtr(posSnap.StopLoss)
			if posSnap.Trailing != nil {
				ts := *posSnap.Trailing
				pos.Trailing = &ts
			}
			acct.Positions[pos.Instrument] = pos
		}
		if len(acctSnap.PendingOrders) > 0 {
			acct.PendingOrders = make([]*PendingOrder, 0, len(acctSnap.PendingOrders))
			for _, ord := range acctSnap.PendingOrders {
				orderCopy := ord
				acct.PendingOrders = append(acct.PendingOrders, &PendingOrder{
					ID:         orderCopy.ID,
					Type:       orderCopy.Type,
					Instrument: orderCopy.Instrument,
					Units:      orderCopy.Units,
					Price:      orderCopy.Price,
					Time:       orderCopy.Time,
					Risk:       cloneRiskSettings(orderCopy.Risk),
				})
			}
		}
		e.accounts[acct.ID] = acct
	}
}

func (e *Engine) buildEquitySnapshotsLocked() []equityRecord {
	if e.persistor == nil || len(e.accounts) == 0 {
		return nil
	}
	records := make([]equityRecord, 0, len(e.accounts))
	now := time.Now().UTC()
	for _, acct := range e.accounts {
		nav := acct.Balance + e.unrealizedForAccountLocked(acct)
		records = append(records, equityRecord{
			AccountID: acct.ID,
			Point: EquityPoint{
				Time:    now,
				Balance: acct.Balance,
				NAV:     nav,
			},
		})
	}
	return records
}

func (e *Engine) persistEquitySnapshots(records []equityRecord) {
	if e.persistor == nil || len(records) == 0 {
		return
	}
	for _, rec := range records {
		if err := e.persistor.AppendEquity(rec.AccountID, rec.Point); err != nil {
			log.Printf("state: failed to append equity: %v", err)
		}
	}
}

func (e *Engine) persistState(snapshot SerializedEngine) {
	if e.persistor == nil || len(snapshot.Accounts) == 0 {
		return
	}
	if err := e.persistor.Save(snapshot); err != nil {
		log.Printf("state: failed to save snapshot: %v", err)
	}
}

func copyFloatPtr(src *float64) *float64 {
	if src == nil {
		return nil
	}
	val := *src
	return &val
}

func cloneRiskSettings(risk RiskSettings) RiskSettings {
	return RiskSettings{
		TakeProfit:       copyFloatPtr(risk.TakeProfit),
		StopLoss:         copyFloatPtr(risk.StopLoss),
		TrailingDistance: copyFloatPtr(risk.TrailingDistance),
	}
}

func (e *Engine) ensureMarginLocked(acct *Account, instrument string, units int64, price float64) error {
	if e.cfg.MaxLeverage <= 0 {
		return nil
	}
	projected := e.projectedExposureLocked(acct, instrument, units, price)
	required := projected / e.cfg.MaxLeverage
	nav := acct.Balance + e.unrealizedForAccountLocked(acct)
	if nav+1e-9 < required {
		return fmt.Errorf("insufficient margin: require %.2f, nav %.2f", required, nav)
	}
	return nil
}

func (e *Engine) projectedExposureLocked(acct *Account, instrument string, deltaUnits int64, price float64) float64 {
	total := 0.0
	found := false
	for inst, pos := range acct.Positions {
		newUnits := pos.Units
		refPrice := positivePrice(pos.AveragePrice)
		if inst == instrument {
			newUnits += deltaUnits
			refPrice = positivePrice(price)
			found = true
		}
		total += math.Abs(float64(newUnits)) * refPrice
	}
	if !found {
		total += math.Abs(float64(deltaUnits)) * positivePrice(price)
	}
	return total
}

func positivePrice(val float64) float64 {
	if val <= 0 {
		return 1e-6
	}
	return val
}

// Utilities.

func sameSign(a, b int64) bool {
	return (a == 0 && b == 0) || (a > 0 && b > 0) || (a < 0 && b < 0)
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// MarshalJSON ensures time formatting matches RFC3339Nano.
func (t Transaction) MarshalJSON() ([]byte, error) {
	type Alias Transaction
	return json.Marshal(&struct {
		Time string `json:"time"`
		Alias
	}{
		Time:  t.Time.Format(time.RFC3339Nano),
		Alias: Alias(t),
	})
}
