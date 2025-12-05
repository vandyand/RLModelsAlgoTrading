package pricing

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// OandaStreamSource streams real-time pricing from OANDA's PricingStream endpoint.
type OandaStreamSource struct {
	AccountID string
	APIKey    string
	Env       string // "practice" or "trade"

	HTTPClient *http.Client
}

// Run satisfies Source.
func (s *OandaStreamSource) Run(ctx context.Context, instruments []string, handler func(Tick)) error {
	if s.HTTPClient == nil {
		s.HTTPClient = &http.Client{
			Timeout: 0,
		}
	}
	base := "https://stream-fxpractice.oanda.com"
	if strings.EqualFold(s.Env, "trade") {
		base = "https://stream-fxtrade.oanda.com"
	}
	instCSV := strings.Join(instruments, ",")

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		err := s.streamOnce(ctx, base, instCSV, handler)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("oanda stream error: %v; retrying in 3s", err)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(3 * time.Second):
			}
		} else {
			// stream ended gracefully; short delay before restart
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Second):
			}
		}
	}
}

func (s *OandaStreamSource) streamOnce(ctx context.Context, baseURL, instruments string, handler func(Tick)) error {
	url := fmt.Sprintf("%s/v3/accounts/%s/pricing/stream", baseURL, s.AccountID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	q := req.URL.Query()
	q.Set("instruments", instruments)
	req.URL.RawQuery = q.Encode()
	req.Header.Set("Authorization", "Bearer "+s.APIKey)
	req.Header.Set("Accept-Datetime-Format", "RFC3339")

	resp, err := s.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("oanda stream status %d: %s", resp.StatusCode, string(body))
	}

	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		line = bytesTrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var msg streamMessage
		if err := json.Unmarshal(line, &msg); err != nil {
			log.Printf("oanda stream decode error: %v (payload=%s)", err, string(line))
			continue
		}
		if strings.EqualFold(msg.Type, "HEARTBEAT") {
			continue
		}
		if !strings.EqualFold(msg.Type, "PRICE") {
			continue
		}
		tick, err := msg.toTick()
		if err != nil {
			log.Printf("oanda tick parse error: %v", err)
			continue
		}
		handler(tick)
	}
}

type streamMessage struct {
	Type        string         `json:"type"`
	Instrument  string         `json:"instrument"`
	Time        string         `json:"time"`
	Status      string         `json:"status"`
	Bids        []streamBucket `json:"bids"`
	Asks        []streamBucket `json:"asks"`
	CloseoutBid string         `json:"closeoutBid"`
	CloseoutAsk string         `json:"closeoutAsk"`
}

type streamBucket struct {
	Price     string `json:"price"`
	Liquidity int64  `json:"liquidity"`
}

func (m streamMessage) toTick() (Tick, error) {
	parse := func(val string) (float64, error) {
		if val == "" {
			return 0, nil
		}
		return strconv.ParseFloat(val, 64)
	}
	var (
		bucketsBid = make([]PriceBucket, 0, len(m.Bids))
		bucketsAsk = make([]PriceBucket, 0, len(m.Asks))
	)
	for _, b := range m.Bids {
		price, err := parse(b.Price)
		if err != nil {
			return Tick{}, err
		}
		bucketsBid = append(bucketsBid, PriceBucket{Price: price, Liquidity: b.Liquidity})
	}
	for _, a := range m.Asks {
		price, err := parse(a.Price)
		if err != nil {
			return Tick{}, err
		}
		bucketsAsk = append(bucketsAsk, PriceBucket{Price: price, Liquidity: a.Liquidity})
	}
	closeoutBid, _ := parse(m.CloseoutBid)
	closeoutAsk, _ := parse(m.CloseoutAsk)
	ts, err := time.Parse(time.RFC3339Nano, m.Time)
	if err != nil {
		ts = time.Now().UTC()
	}
	return Tick{
		Instrument:  m.Instrument,
		Time:        ts,
		Status:      m.Status,
		Bids:        bucketsBid,
		Asks:        bucketsAsk,
		CloseoutBid: closeoutBid,
		CloseoutAsk: closeoutAsk,
		Source:      "oanda",
	}, nil
}

func bytesTrimSpace(b []byte) []byte {
	start := 0
	for ; start < len(b) && (b[start] == ' ' || b[start] == '\n' || b[start] == '\r' || b[start] == '\t'); start++ {
	}
	end := len(b) - 1
	for ; end >= start && (b[end] == ' ' || b[end] == '\n' || b[end] == '\r' || b[end] == '\t'); end-- {
	}
	if start >= len(b) {
		return nil
	}
	return b[start : end+1]
}
