package pricing

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// TickListener buffer default.
const defaultListenerBuf = 64

// Source emits ticks for each instrument requested.
type Source interface {
	Run(ctx context.Context, instruments []string, handler func(Tick)) error
}

// Manager fans out ticks to interested consumers and stores the latest snapshot.
type Manager struct {
	source      Source
	instruments []string

	mu          sync.RWMutex
	last        map[string]Tick
	subscribers map[int64]chan Tick

	nextSubID int64
}

// NewManager wires the given Source into a reusable tick cache.
func NewManager(src Source, instruments []string) *Manager {
	return &Manager{
		source:      src,
		instruments: instruments,
		last:        make(map[string]Tick),
		subscribers: make(map[int64]chan Tick),
	}
}

// Start launches the streaming loop. Safe to call once.
func (m *Manager) Start(ctx context.Context) {
	if m.source == nil {
		return
	}
	go func() {
		if err := m.source.Run(ctx, m.instruments, m.handleTick); err != nil && ctx.Err() == nil {
			log.Printf("pricing source stopped: %v", err)
		}
	}()
}

// Subscribe registers a buffered channel that receives ticks. Returns id and channel.
func (m *Manager) Subscribe(buffer int) (int64, <-chan Tick) {
	if buffer <= 0 {
		buffer = defaultListenerBuf
	}
	ch := make(chan Tick, buffer)
	id := atomic.AddInt64(&m.nextSubID, 1)

	m.mu.Lock()
	m.subscribers[id] = ch
	m.mu.Unlock()
	return id, ch
}

// Unsubscribe removes a listener.
func (m *Manager) Unsubscribe(id int64) {
	m.mu.Lock()
	ch, ok := m.subscribers[id]
	if ok {
		delete(m.subscribers, id)
		close(ch)
	}
	m.mu.Unlock()
}

// Latest returns the most recent tick for the instrument.
func (m *Manager) Latest(instrument string) (Tick, bool) {
	m.mu.RLock()
	t, ok := m.last[instrument]
	m.mu.RUnlock()
	return t, ok
}

// Snapshot returns the latest ticks for the requested instruments (wildcard when empty).
func (m *Manager) Snapshot(instruments []string) []Tick {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(instruments) == 0 {
		out := make([]Tick, 0, len(m.last))
		for _, t := range m.last {
			out = append(out, t)
		}
		return out
	}
	out := make([]Tick, 0, len(instruments))
	for _, inst := range instruments {
		if t, ok := m.last[inst]; ok {
			out = append(out, t)
		}
	}
	return out
}

func (m *Manager) handleTick(t Tick) {
	if t.Instrument == "" {
		return
	}
	m.mu.Lock()
	m.last[t.Instrument] = t
	for id, ch := range m.subscribers {
		select {
		case ch <- t:
		default:
			// Drop slow subscriber to preserve real-time characteristics.
			log.Printf("dropping slow pricing subscriber %d for %s", id, t.Instrument)
			close(ch)
			delete(m.subscribers, id)
		}
	}
	m.mu.Unlock()
}

// Tick represents a normalized best-bid-offer update.
type Tick struct {
	Instrument  string
	Time        time.Time
	Status      string
	Bids        []PriceBucket
	Asks        []PriceBucket
	CloseoutBid float64
	CloseoutAsk float64
	Source      string
}

// PriceBucket captures price + available liquidity for one side.
type PriceBucket struct {
	Price     float64
	Liquidity int64
}

// BestBid returns top-of-book bid.
func (t Tick) BestBid() (float64, bool) {
	if len(t.Bids) == 0 {
		return 0, false
	}
	return t.Bids[0].Price, true
}

// BestAsk returns top-of-book ask.
func (t Tick) BestAsk() (float64, bool) {
	if len(t.Asks) == 0 {
		return 0, false
	}
	return t.Asks[0].Price, true
}

// MidPrice returns midpoint when bid/ask available.
func (t Tick) MidPrice() (float64, bool) {
	bid, bok := t.BestBid()
	ask, aok := t.BestAsk()
	if !bok || !aok {
		return 0, false
	}
	return (bid + ask) / 2, true
}
