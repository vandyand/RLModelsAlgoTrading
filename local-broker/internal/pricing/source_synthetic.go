package pricing

import (
	"context"
	"math/rand"
	"time"
)

// SyntheticSource produces deterministic pseudo-random ticks for offline development.
type SyntheticSource struct {
	UpdateInterval time.Duration
	Seed           int64
	Spread         float64
}

// Run emits ticks until the context is cancelled.
func (s *SyntheticSource) Run(ctx context.Context, instruments []string, handler func(Tick)) error {
	interval := s.UpdateInterval
	if interval <= 0 {
		interval = 200 * time.Millisecond
	}
	rng := rand.New(rand.NewSource(s.Seed))

	prices := make(map[string]float64, len(instruments))
	for _, inst := range instruments {
		prices[inst] = 1.0 + rng.Float64()*0.5
	}
	spread := s.Spread
	if spread <= 0 {
		spread = 0.0001
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case t := <-ticker.C:
			for _, inst := range instruments {
				mid := prices[inst]
				// Random walk
				mid += (rng.Float64() - 0.5) * spread * 4
				if mid <= 0 {
					mid = 0.5
				}
				prices[inst] = mid
				bid := mid - spread/2
				ask := mid + spread/2
				handler(Tick{
					Instrument: inst,
					Time:       t.UTC(),
					Status:     "tradeable",
					Bids: []PriceBucket{
						{Price: bid, Liquidity: 1_000_000},
					},
					Asks: []PriceBucket{
						{Price: ask, Liquidity: 1_000_000},
					},
					CloseoutBid: bid,
					CloseoutAsk: ask,
					Source:      "synthetic",
				})
			}
		}
	}
}
