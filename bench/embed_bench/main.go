// Package main is the ctx++ embedding throughput benchmark harness.
//
// It measures embedding throughput (symbols/sec, batches/sec, latency
// percentiles) across different batch sizes and backends (Ollama TCP, Ollama
// UDS, and TEI).
//
// Usage:
//
//	go run ./bench/embed_bench [flags]
//
//	  -url string        Ollama base URL (default: http://localhost:11434)
//	  -socket string     Unix domain socket path (optional; bypasses TCP for Ollama)
//	  -model string      Ollama model name (default: all-minilm)
//	  -tei-url string    TEI base URL (default: "" — TEI skipped if empty)
//	  -tei-model string  TEI model identifier (default: sentence-transformers/all-MiniLM-L6-v2)
//	  -tei-dims int      TEI embedding dimensions (default: 384)
//	  -n int             Total number of symbols to embed per run (default: 1000)
//	  -batches string    Comma-separated batch sizes to test (default: 1,8,32,128,512)
//	  -warmup int        Number of warmup embed calls before timing (default: 1)
//	  -out string        Output JSON report path (empty = stdout)
//
// Requires a running Ollama instance with the specified model pulled (unless -url is skipped).
// To run TEI locally with Docker and an NVIDIA GPU:
//
//	docker run --gpus all -p 8080:80 \
//	  ghcr.io/huggingface/text-embeddings-inference:turing-latest \
//	  --model-id sentence-transformers/all-MiniLM-L6-v2
//
// Then benchmark both backends:
//
//	go run ./bench/embed_bench -tei-url http://localhost:8080 -n 2000 -batches 32,128,512,1024
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
)

// ---- synthetic text generation ---------------------------------------------

// syntheticTexts returns n representative symbol embed texts of varying length,
// modelled after what buildEmbedText produces in the indexer.
func syntheticTexts(n int) []string {
	templates := []string{
		"func %s: func(%s string) error\n// %s handles incoming requests and validates parameters.",
		"method %s.%s: func(ctx context.Context, id int) (*Result, error)\n// %s performs a database lookup.",
		"type %s struct\n// %s represents a configuration object with multiple fields.",
		"func New%s: func(cfg *Config) *%s\n// New%s constructs a new instance with default settings.",
		"method %s.Close: func() error\n// Close releases all resources held by %s.",
		"func %s: func(a, b string) string\n// %s concatenates two strings with a separator.",
		"type %sHandler struct\n// %sHandler processes HTTP requests for the %s endpoint.",
		"func Parse%s: func(data []byte) (*%s, error)\n// Parse%s decodes a JSON payload into the struct.",
		"method %s.String: func() string\n// String returns a human-readable representation.",
		"func %sFromEnv: func() (*%s, error)\n// %sFromEnv reads configuration from environment variables.",
	}

	texts := make([]string, n)
	names := []string{
		"Handler", "Server", "Client", "Config", "Request", "Response",
		"Manager", "Worker", "Queue", "Buffer", "Processor", "Validator",
		"Parser", "Encoder", "Decoder", "Cache", "Store", "Index",
		"Router", "Middleware", "Scheduler", "Logger", "Monitor", "Registry",
	}

	for i := 0; i < n; i++ {
		tmpl := templates[i%len(templates)]
		name := names[i%len(names)]
		name2 := names[(i+1)%len(names)]
		texts[i] = fmt.Sprintf(tmpl, name, name2, name, name2, name)
	}
	return texts
}

// ---- result types ----------------------------------------------------------

// RunResult captures metrics for one (transport, batchSize) combination.
type RunResult struct {
	Transport   string  `json:"transport"` // "tcp" or "uds"
	BatchSize   int     `json:"batch_size"`
	TotalN      int     `json:"total_n"`
	Batches     int     `json:"batches"`
	TotalNs     int64   `json:"total_ns"`
	TotalS      string  `json:"total_s"`
	SymbolsPerS float64 `json:"symbols_per_s"`
	BatchesPerS float64 `json:"batches_per_s"`
	LatP50Ms    float64 `json:"lat_p50_ms"`
	LatP95Ms    float64 `json:"lat_p95_ms"`
	LatP99Ms    float64 `json:"lat_p99_ms"`
	LatMinMs    float64 `json:"lat_min_ms"`
	LatMaxMs    float64 `json:"lat_max_ms"`
}

// Report is the full benchmark output.
type Report struct {
	Timestamp string      `json:"timestamp"`
	GoVersion string      `json:"go_version"`
	GOOS      string      `json:"goos"`
	GOARCH    string      `json:"goarch"`
	NumCPU    int         `json:"num_cpu"`
	Model     string      `json:"model"`
	TotalN    int         `json:"total_n"`
	Results   []RunResult `json:"results"`
}

// ---- benchmark runner ------------------------------------------------------

func runBatch(
	ctx context.Context,
	embedder embed.BatchEmbedder,
	texts []string,
	batchSize int,
) (RunResult, error) {
	n := len(texts)
	numBatches := (n + batchSize - 1) / batchSize
	latencies := make([]float64, 0, numBatches)

	start := time.Now()
	for i := 0; i < n; i += batchSize {
		end := i + batchSize
		if end > n {
			end = n
		}
		batch := texts[i:end]

		bStart := time.Now()
		_, err := embedder.EmbedBatch(ctx, batch)
		elapsed := time.Since(bStart)

		if err != nil {
			return RunResult{}, fmt.Errorf("EmbedBatch at offset %d: %w", i, err)
		}
		latencies = append(latencies, float64(elapsed.Milliseconds()))
	}
	total := time.Since(start)

	sort.Float64s(latencies)

	return RunResult{
		BatchSize:   batchSize,
		TotalN:      n,
		Batches:     len(latencies),
		TotalNs:     total.Nanoseconds(),
		TotalS:      total.Round(time.Millisecond).String(),
		SymbolsPerS: float64(n) / total.Seconds(),
		BatchesPerS: float64(len(latencies)) / total.Seconds(),
		LatP50Ms:    percentile(latencies, 50),
		LatP95Ms:    percentile(latencies, 95),
		LatP99Ms:    percentile(latencies, 99),
		LatMinMs:    latencies[0],
		LatMaxMs:    latencies[len(latencies)-1],
	}, nil
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(len(sorted))*p/100)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// ---- main ------------------------------------------------------------------

func main() {
	var (
		ollamaURL  = flag.String("url", "http://localhost:11434", "Ollama base URL")
		socketPath = flag.String("socket", "", "Unix domain socket path (optional, Ollama only)")
		modelName  = flag.String("model", "all-minilm", "Ollama model name")
		teiURL     = flag.String("tei-url", "", "TEI base URL (e.g. http://localhost:8080); skipped if empty")
		teiModel   = flag.String("tei-model", "sentence-transformers/all-MiniLM-L6-v2", "TEI model identifier")
		teiDims    = flag.Int("tei-dims", 384, "TEI embedding dimensions")
		totalN     = flag.Int("n", 1000, "total symbols to embed per run")
		batchesStr = flag.String("batches", "1,8,32,128,512", "comma-separated batch sizes")
		warmup     = flag.Int("warmup", 1, "warmup embed calls (not timed)")
		outPath    = flag.String("out", "", "output JSON path (empty = stdout)")
	)
	flag.Parse()

	// Parse batch sizes.
	var batchSizes []int
	for _, s := range strings.Split(*batchesStr, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		v, err := strconv.Atoi(s)
		if err != nil || v <= 0 {
			log.Fatalf("invalid batch size %q", s)
		}
		batchSizes = append(batchSizes, v)
	}
	if len(batchSizes) == 0 {
		log.Fatal("no batch sizes specified")
	}

	ctx := context.Background()

	// Build Ollama TCP embedder and probe reachability.
	tcpEmbedder := embed.NewOllamaEmbedder(*ollamaURL, *modelName, "")
	ollamaReachable := false
	log.Printf("probing Ollama at %s ...", *ollamaURL)
	if err := tcpEmbedder.Ping(ctx); err != nil {
		log.Printf("Ollama unreachable (%v) — Ollama runs will be skipped", err)
	} else {
		ollamaReachable = true
		log.Printf("Ollama reachable — model=%s", *modelName)
	}

	// Build UDS embedder if socket path provided.
	var udsEmbedder *embed.OllamaEmbedder
	if *socketPath != "" && ollamaReachable {
		udsEmbedder = embed.NewOllamaEmbedder(*ollamaURL, *modelName, *socketPath)
		log.Printf("UDS transport configured: %s", *socketPath)
	}

	// Build TEI embedder if URL provided.
	var teiEmbedder *embed.TEIEmbedder
	if *teiURL != "" {
		teiEmbedder = embed.NewTEIEmbedder(*teiURL, *teiModel, *teiDims)
		log.Printf("probing TEI at %s ...", *teiURL)
		if err := teiEmbedder.Ping(ctx); err != nil {
			log.Printf("TEI unreachable (%v) — TEI runs will be skipped", err)
			teiEmbedder = nil
		} else {
			log.Printf("TEI reachable — model=%s dims=%d", *teiModel, *teiDims)
		}
	}

	if !ollamaReachable && teiEmbedder == nil {
		log.Fatal("no embedding backend reachable; provide a running Ollama or TEI instance")
	}

	// Generate synthetic embed texts.
	texts := syntheticTexts(*totalN)
	log.Printf("generated %d synthetic symbol texts", len(texts))

	// Warmup using first available backend, batch=1.
	if *warmup > 0 {
		log.Printf("warming up with %d single-embed calls...", *warmup)
		var warmer embed.BatchEmbedder
		if teiEmbedder != nil {
			warmer = teiEmbedder
		} else {
			warmer = tcpEmbedder
		}
		for i := 0; i < *warmup; i++ {
			if _, err := warmer.EmbedBatch(ctx, texts[:1]); err != nil {
				log.Printf("warmup call %d failed: %v", i+1, err)
			}
		}
		log.Printf("warmup done")
	}

	report := Report{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		GoVersion: runtime.Version(),
		GOOS:      runtime.GOOS,
		GOARCH:    runtime.GOARCH,
		NumCPU:    runtime.NumCPU(),
		Model:     *modelName,
		TotalN:    *totalN,
	}

	// --- Ollama TCP runs ---
	if ollamaReachable {
		log.Printf("--- Ollama TCP transport ---")
		for _, bs := range batchSizes {
			log.Printf("  batch_size=%d  n=%d ...", bs, *totalN)
			r, err := runBatch(ctx, tcpEmbedder, texts, bs)
			if err != nil {
				log.Printf("  ERROR: %v", err)
				continue
			}
			r.Transport = "ollama-tcp"
			report.Results = append(report.Results, r)
			log.Printf("  symbols/s=%.0f  batches/s=%.1f  p50=%.1fms  p95=%.1fms  p99=%.1fms  total=%s",
				r.SymbolsPerS, r.BatchesPerS, r.LatP50Ms, r.LatP95Ms, r.LatP99Ms, r.TotalS)
		}
	}

	// --- Ollama UDS runs (if socket configured) ---
	if udsEmbedder != nil {
		log.Printf("--- Ollama UDS transport (%s) ---", *socketPath)
		for _, bs := range batchSizes {
			log.Printf("  batch_size=%d  n=%d ...", bs, *totalN)
			r, err := runBatch(ctx, udsEmbedder, texts, bs)
			if err != nil {
				log.Printf("  ERROR: %v", err)
				continue
			}
			r.Transport = "ollama-uds"
			report.Results = append(report.Results, r)
			log.Printf("  symbols/s=%.0f  batches/s=%.1f  p50=%.1fms  p95=%.1fms  p99=%.1fms  total=%s",
				r.SymbolsPerS, r.BatchesPerS, r.LatP50Ms, r.LatP95Ms, r.LatP99Ms, r.TotalS)
		}
	}

	// --- TEI runs ---
	if teiEmbedder != nil {
		log.Printf("--- TEI backend (%s) ---", *teiURL)
		for _, bs := range batchSizes {
			log.Printf("  batch_size=%d  n=%d ...", bs, *totalN)
			r, err := runBatch(ctx, teiEmbedder, texts, bs)
			if err != nil {
				log.Printf("  ERROR: %v", err)
				continue
			}
			r.Transport = "tei"
			report.Results = append(report.Results, r)
			log.Printf("  symbols/s=%.0f  batches/s=%.1f  p50=%.1fms  p95=%.1fms  p99=%.1fms  total=%s",
				r.SymbolsPerS, r.BatchesPerS, r.LatP50Ms, r.LatP95Ms, r.LatP99Ms, r.TotalS)
		}
	}

	// Output JSON report.
	enc := json.NewEncoder(os.Stdout)
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			log.Fatalf("create output: %v", err)
		}
		defer f.Close()
		enc = json.NewEncoder(f)
	}
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatalf("encode report: %v", err)
	}

	// Print human-readable summary to stderr.
	if len(report.Results) > 0 {
		printSummary(os.Stderr, report)
	}
}

func printSummary(f *os.File, r Report) {
	fmt.Fprintf(f, "\n## Embedding Throughput Benchmark\n\n")
	fmt.Fprintf(f, "- **Model**: %s\n", r.Model)
	fmt.Fprintf(f, "- **Total symbols per run**: %d\n", r.TotalN)
	fmt.Fprintf(f, "- **System**: %s/%s, %d CPUs, %s\n\n", r.GOOS, r.GOARCH, r.NumCPU, r.GoVersion)

	fmt.Fprintf(f, "| Transport | Batch | Symbols/s | Batches/s | p50 (ms) | p95 (ms) | p99 (ms) | Total |\n")
	fmt.Fprintf(f, "|-----------|-------|-----------|-----------|----------|----------|----------|-------|\n")
	for _, res := range r.Results {
		fmt.Fprintf(f, "| %s | %d | %.0f | %.1f | %.1f | %.1f | %.1f | %s |\n",
			res.Transport,
			res.BatchSize,
			res.SymbolsPerS,
			res.BatchesPerS,
			res.LatP50Ms,
			res.LatP95Ms,
			res.LatP99Ms,
			res.TotalS,
		)
	}
	fmt.Fprintln(f)

	// Find best row.
	best := r.Results[0]
	for _, res := range r.Results[1:] {
		if res.SymbolsPerS > best.SymbolsPerS {
			best = res
		}
	}
	fmt.Fprintf(f, "**Best**: transport=%s batch_size=%d → %.0f symbols/s\n\n", best.Transport, best.BatchSize, best.SymbolsPerS)
}
