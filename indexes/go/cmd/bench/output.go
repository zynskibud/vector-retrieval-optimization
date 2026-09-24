package main

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
)

// result is the output JSON of CONTRACT.md section 3. Field tags fix the key names.
type result struct {
	ContractVersion int            `json:"contract_version"`
	Language        string         `json:"language"`
	Index           string         `json:"index"`
	DataDir         string         `json:"data_dir"`
	N               int            `json:"n"`
	Dim             int            `json:"dim"`
	Q               int            `json:"q"`
	K               int            `json:"k"`
	Threads         int            `json:"threads"`
	Seed            uint64         `json:"seed"`
	BuildParams     map[string]any `json:"build_params"`
	Build           buildInfo      `json:"build"`
	Searches        []searchRun    `json:"searches"`
	Machine         machine        `json:"machine"`
	Extra           map[string]any `json:"extra"`
}

type buildInfo struct {
	TrainS     float64 `json:"train_s"`
	AddS       float64 `json:"add_s"`
	TotalS     float64 `json:"total_s"`
	PeakRSSMB  float64 `json:"peak_rss_mb"`
	IndexBytes int64   `json:"index_bytes"`
}

type searchRun struct {
	SearchParams         map[string]any `json:"search_params"`
	IDs                  [][]int64      `json:"ids"`
	Scores               [][]score      `json:"scores"`
	LatencyMS            []float64      `json:"latency_ms"`
	TotalS               float64        `json:"total_s"`
	QPS                  float64        `json:"qps"`
	DistanceComputations *float64       `json:"distance_computations"` // nil -> null
	Extra                map[string]any `json:"extra"`
}

type machine struct {
	OS    string `json:"os"`
	Arch  string `json:"arch"`
	CPU   string `json:"cpu"`
	Cores int    `json:"cores"`
}

// score is a result score. -Inf (padding) serializes as null.
type score float32

// MarshalJSON writes the shortest float32 text that round-trips, or null for non-finite values.
func (s score) MarshalJSON() ([]byte, error) {
	f := float64(s)
	if math.IsInf(f, 0) || math.IsNaN(f) {
		return []byte("null"), nil
	}
	return strconv.AppendFloat(nil, f, 'g', -1, 32), nil
}

func writeJSON(path string, r *result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := json.NewEncoder(f).Encode(r); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
