package storage

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"localbroker/internal/state"
)

// FileStore persists broker state and equity history on disk.
type FileStore struct {
	statePath  string
	equityPath string

	mu sync.Mutex
}

// NewFileStore ensures the data directory exists and returns a store instance.
func NewFileStore(dataDir string) (*FileStore, error) {
	if dataDir == "" {
		dataDir = "./broker-data"
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, err
	}
	return &FileStore{
		statePath:  filepath.Join(dataDir, "state.json"),
		equityPath: filepath.Join(dataDir, "equity.jsonl"),
	}, nil
}

// Load reads the last saved engine snapshot from disk.
func (s *FileStore) Load() (state.SerializedEngine, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := os.ReadFile(s.statePath)
	if errors.Is(err, os.ErrNotExist) {
		return state.SerializedEngine{}, nil
	}
	if err != nil {
		return state.SerializedEngine{}, err
	}
	var snapshot state.SerializedEngine
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return state.SerializedEngine{}, err
	}
	return snapshot, nil
}

// Save writes a new snapshot atomically.
func (s *FileStore) Save(snapshot state.SerializedEngine) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if snapshot.Timestamp.IsZero() {
		snapshot.Timestamp = time.Now().UTC()
	}
	tmp := s.statePath + ".tmp"
	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.statePath)
}

// AppendEquity writes a single equity point for the given account as JSONL.
func (s *FileStore) AppendEquity(accountID string, point state.EquityPoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.equityPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if point.Time.IsZero() {
		point.Time = time.Now().UTC()
	}
	entry := struct {
		AccountID string            `json:"accountID"`
		Point     state.EquityPoint `json:"point"`
	}{
		AccountID: accountID,
		Point:     point,
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	_, err = f.Write(append(data, '\n'))
	return err
}

// ReadEquity returns the most recent equity points for an account (limit <=0 = all).
func (s *FileStore) ReadEquity(accountID string, limit int) ([]state.EquityPoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	file, err := os.Open(s.equityPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	points := make([]state.EquityPoint, 0)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var entry struct {
			AccountID string            `json:"accountID"`
			Point     state.EquityPoint `json:"point"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			continue
		}
		if accountID != "" && entry.AccountID != accountID {
			continue
		}
		points = append(points, entry.Point)
		if limit > 0 && len(points) > limit {
			points = points[1:]
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read equity history: %w", err)
	}
	return points, nil
}
