// Package embed defines the Embedder interface and its implementations.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

// Embedder converts text into a dense float32 vector.
type Embedder interface {
	// Model returns the model identifier string (included in stored embeddings).
	Model() string

	// Dims returns the embedding dimensionality.
	Dims() int

	// Embed returns the embedding vector for the given text.
	Embed(ctx context.Context, text string) ([]float32, error)
}

// BatchEmbedder is an optional interface that embedders can implement to
// support batched embedding calls. This is critical for performance: Ollama's
// /api/embed endpoint accepts an array of inputs, amortising per-call overhead
// and enabling GPU batching. Without this, embedding 181k symbols requires
// 181k HTTP round-trips.
type BatchEmbedder interface {
	Embedder
	// EmbedBatch returns embedding vectors for multiple texts in a single call.
	// The returned slice must have the same length as texts.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// ---- Ollama ----------------------------------------------------------------

const defaultOllamaURL = "http://localhost:11434"
const defaultOllamaModel = "bge-m3"
const ollamaDims = 1024

// OllamaEmbedder calls the Ollama /api/embed endpoint.
type OllamaEmbedder struct {
	baseURL string
	model   string
	client  *http.Client
}

// NewOllamaEmbedder constructs an OllamaEmbedder with a tuned HTTP client.
// If baseURL is empty, http://localhost:11434 is used.
// If model is empty, nomic-embed-text is used.
// If socketPath is non-empty, the client connects via Unix domain socket
// (baseURL is still used for the Host header but routing goes through the socket).
func NewOllamaEmbedder(baseURL, model, socketPath string) *OllamaEmbedder {
	if baseURL == "" {
		baseURL = defaultOllamaURL
	}
	if model == "" {
		model = defaultOllamaModel
	}

	transport := &http.Transport{
		MaxIdleConns:        64,
		MaxIdleConnsPerHost: 64,
		IdleConnTimeout:     120 * time.Second,
		DisableCompression:  true,
		ForceAttemptHTTP2:   false, // Ollama serves HTTP/1.1; skip ALPN overhead
	}

	if socketPath != "" {
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		}
	}

	return &OllamaEmbedder{
		baseURL: baseURL,
		model:   model,
		client: &http.Client{
			Timeout:   120 * time.Second, // batch calls can take a while
			Transport: transport,
		},
	}
}

func (e *OllamaEmbedder) Model() string { return e.model }
func (e *OllamaEmbedder) Dims() int     { return ollamaDims }

type ollamaEmbedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string for single, []string for batch
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed calls Ollama's /api/embed and returns the first embedding vector.
func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.doEmbed(ctx, text)
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, fmt.Errorf("ollama embed: empty response")
	}
	return vecs[0], nil
}

// EmbedBatch calls Ollama's /api/embed with an array of inputs, returning
// one embedding vector per input. This is dramatically faster than calling
// Embed in a loop because it amortises HTTP overhead and enables GPU batching.
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if len(texts) == 1 {
		vec, err := e.Embed(ctx, texts[0])
		if err != nil {
			return nil, err
		}
		return [][]float32{vec}, nil
	}
	vecs, err := e.doEmbed(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("ollama embed batch: got %d embeddings for %d inputs", len(vecs), len(texts))
	}
	return vecs, nil
}

// doEmbed is the shared HTTP call for both single and batch embedding.
// input should be a string (single) or []string (batch).
func (e *OllamaEmbedder) doEmbed(ctx context.Context, input any) ([][]float32, error) {
	body, err := json.Marshal(ollamaEmbedRequest{Model: e.model, Input: input})
	if err != nil {
		return nil, fmt.Errorf("ollama embed: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama embed: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embed: status %d", resp.StatusCode)
	}

	var result ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("ollama embed: decode: %w", err)
	}
	return result.Embeddings, nil
}

// Ping checks whether Ollama is reachable and the model is available.
func (e *OllamaEmbedder) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("ollama ping: %w", err)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("ollama ping: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama ping: status %d", resp.StatusCode)
	}
	return nil
}

// ---- Bundled (stub) --------------------------------------------------------

const bundledModel = "bundled-zero"

// BundledModel is the model name returned by the stub BundledEmbedder.
// Code that wants to skip embedding when no real model is available can
// compare embedder.Model() against this constant.
const BundledModel = bundledModel

// BundledEmbedder is a zero-vector stub used when Ollama is unavailable.
// Semantic search will be non-functional, but keyword search works normally.
// Replace with a real ONNX/WASM model in a future version.
type BundledEmbedder struct {
	dims int
}

// NewBundledEmbedder returns a stub embedder with the given dimensionality.
// Use dims=768 to match the Ollama nomic-embed-text model so vectors are
// schema-compatible if the user later enables Ollama.
func NewBundledEmbedder(dims int) *BundledEmbedder {
	if dims <= 0 {
		dims = ollamaDims
	}
	return &BundledEmbedder{dims: dims}
}

func (e *BundledEmbedder) Model() string { return bundledModel }
func (e *BundledEmbedder) Dims() int     { return e.dims }

// Embed returns a zero vector. Semantic search will rank all results equally.
func (e *BundledEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, e.dims), nil
}

// ---- Non-retryable errors --------------------------------------------------

// NonRetryableError wraps an error that should not be retried. When
// RetryingEmbedder sees a NonRetryableError it returns immediately without
// consuming any remaining retry budget.
type NonRetryableError struct {
	cause error
}

// NewNonRetryableError wraps cause so that RetryingEmbedder fast-fails.
func NewNonRetryableError(cause error) *NonRetryableError {
	return &NonRetryableError{cause: cause}
}

func (e *NonRetryableError) Error() string { return e.cause.Error() }
func (e *NonRetryableError) Unwrap() error { return e.cause }

// isNonRetryable returns true if err (or any error in its chain) is a
// NonRetryableError.
func isNonRetryable(err error) bool {
	var nre *NonRetryableError
	return errors.As(err, &nre)
}

// ---- Auto-detect -----------------------------------------------------------

// RetryConfig controls retry behavior for the RetryingEmbedder.
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts after the initial call.
	// Defaults to 3 if <= 0.
	MaxRetries int

	// BaseBackoff is the initial backoff duration before the first retry.
	// Subsequent retries use exponential backoff: base * 2^(attempt-1).
	// Defaults to 100ms if <= 0.
	BaseBackoff time.Duration
}

func (c *RetryConfig) setDefaults() {
	if c.MaxRetries <= 0 {
		c.MaxRetries = 3
	}
	if c.BaseBackoff <= 0 {
		c.BaseBackoff = 100 * time.Millisecond
	}
}

// RetryingEmbedder wraps an Embedder with exponential backoff and jitter on
// transient failures. It delegates Model() and Dims() directly to the inner
// embedder.
type RetryingEmbedder struct {
	inner Embedder
	cfg   RetryConfig
}

// NewRetryingEmbedder wraps inner with retry logic according to cfg.
func NewRetryingEmbedder(inner Embedder, cfg RetryConfig) *RetryingEmbedder {
	cfg.setDefaults()
	return &RetryingEmbedder{inner: inner, cfg: cfg}
}

func (r *RetryingEmbedder) Model() string { return r.inner.Model() }
func (r *RetryingEmbedder) Dims() int     { return r.inner.Dims() }

// Embed calls the inner embedder, retrying on error with exponential backoff
// and jitter. It respects context cancellation between retries.
func (r *RetryingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	var lastErr error
	for attempt := 0; attempt <= r.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := r.cfg.BaseBackoff * (1 << (attempt - 1))
			// Add jitter: ±25% of backoff. Guard against zero to avoid
			// Int63n(0) panic when BaseBackoff is very small (e.g. in tests).
			if half := int64(backoff / 2); half > 0 {
				jitter := time.Duration(rand.Int63n(half)) - backoff/4
				backoff += jitter
			}

			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("embed retry: context cancelled after %d attempts: %w", attempt, ctx.Err())
			case <-time.After(backoff):
			}
		}

		vec, err := r.inner.Embed(ctx, text)
		if err == nil {
			return vec, nil
		}
		lastErr = err

		// Non-retryable errors (e.g. Bedrock AccessDenied, ValidationException)
		// must not burn through the retry budget — fail immediately.
		if isNonRetryable(err) {
			return nil, err
		}

		// Check context between retries to fail fast.
		if ctx.Err() != nil {
			return nil, fmt.Errorf("embed retry: context cancelled after %d attempts: %w", attempt+1, ctx.Err())
		}
	}
	return nil, fmt.Errorf("embed retry: exhausted %d retries: %w", r.cfg.MaxRetries, lastErr)
}

// EmbedBatch delegates to the inner embedder's EmbedBatch if available,
// otherwise falls back to calling Embed in a loop. When the inner embedder
// supports batching, the entire batch is retried on failure. When falling
// back to single Embed calls, individual failures produce nil vectors
// (the caller must handle nils) rather than aborting the entire batch.
func (r *RetryingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	batcher, ok := r.inner.(BatchEmbedder)
	if !ok {
		// Fallback: call Embed one at a time. Individual failures produce
		// nil entries rather than aborting the whole batch — this is critical
		// for backends like Bedrock where a single oversized input should
		// not discard 2000 other valid embeddings.
		vecs := make([][]float32, len(texts))
		for i, t := range texts {
			v, err := r.Embed(ctx, t)
			if err != nil {
				// Leave vecs[i] as nil; caller skips nil entries.
				continue
			}
			vecs[i] = v
		}
		return vecs, nil
	}

	var lastErr error
	for attempt := 0; attempt <= r.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := r.cfg.BaseBackoff * (1 << (attempt - 1))
			if half := int64(backoff / 2); half > 0 {
				jitter := time.Duration(rand.Int63n(half)) - backoff/4
				backoff += jitter
			}
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("embed batch retry: context cancelled after %d attempts: %w", attempt, ctx.Err())
			case <-time.After(backoff):
			}
		}

		vecs, err := batcher.EmbedBatch(ctx, texts)
		if err == nil {
			return vecs, nil
		}
		lastErr = err
		if ctx.Err() != nil {
			return nil, fmt.Errorf("embed batch retry: context cancelled after %d attempts: %w", attempt+1, ctx.Err())
		}
	}
	return nil, fmt.Errorf("embed batch retry: exhausted %d retries: %w", r.cfg.MaxRetries, lastErr)
}

// ---- Auto-detect (Detect) --------------------------------------------------

// Detect probes the configured embedding backend and returns an appropriate
// Embedder. The returned bool is true when an external embedder (Ollama, TEI,
// or Bedrock) is active; false means the stub BundledEmbedder is in use.
//
// Environment variables:
//
//	CTXPP_EMBED_BACKEND    "auto" (default), "ollama", "tei", "bedrock", "openai", or "bundled"
//	CTXPP_OLLAMA_URL       Ollama base URL (default http://localhost:11434)
//	CTXPP_OLLAMA_MODEL     Ollama model name (default all-minilm)
//	CTXPP_OLLAMA_SOCKET    Unix domain socket path (optional; bypasses TCP)
//	CTXPP_TEI_URL          TEI base URL (default http://localhost:8080)
//	CTXPP_TEI_MODEL        TEI model identifier (default sentence-transformers/all-MiniLM-L6-v2)
//	CTXPP_TEI_DIMS         TEI embedding dimensions (default 384)
//	CTXPP_BEDROCK_REGION   AWS region (default us-east-1)
//	CTXPP_BEDROCK_MODEL    Bedrock model ID (default amazon.titan-embed-text-v2:0)
//	CTXPP_BEDROCK_DIMS     Bedrock embedding dimensions: 256, 512, or 1024 (default 1024)
//	CTXPP_OPENAI_URL       OpenAI-compatible base URL (default https://api.openai.com)
//	CTXPP_OPENAI_MODEL     OpenAI-compatible embedding model name (required when backend=openai)
//	CTXPP_OPENAI_API_KEY   Optional bearer token for OpenAI-compatible APIs
//	CTXPP_OPENAI_DIMS      Embedding dimensions for the selected model (required when backend=openai)
func Detect(ctx context.Context) (Embedder, bool) {
	backend := os.Getenv("CTXPP_EMBED_BACKEND")
	ollamaURL := os.Getenv("CTXPP_OLLAMA_URL")
	ollamaModel := os.Getenv("CTXPP_OLLAMA_MODEL")
	ollamaSocket := os.Getenv("CTXPP_OLLAMA_SOCKET")
	teiURL := os.Getenv("CTXPP_TEI_URL")
	teiModel := os.Getenv("CTXPP_TEI_MODEL")
	bedrockRegion := os.Getenv("CTXPP_BEDROCK_REGION")
	bedrockModel := os.Getenv("CTXPP_BEDROCK_MODEL")
	bedrockDimsStr := os.Getenv("CTXPP_BEDROCK_DIMS")
	openAIURL := os.Getenv("CTXPP_OPENAI_URL")
	openAIModel := os.Getenv("CTXPP_OPENAI_MODEL")
	openAIAPIKey := os.Getenv("CTXPP_OPENAI_API_KEY")
	openAIDimsStr := os.Getenv("CTXPP_OPENAI_DIMS")

	retryCfg := RetryConfig{} // uses defaults: 3 retries, 100ms base backoff

	switch backend {
	case "bundled":
		return NewBundledEmbedder(ollamaDims), false
	case "tei":
		e := NewTEIEmbedder(teiURL, teiModel, 0)
		// Forced TEI — return even if unreachable so callers can surface the error.
		return NewRetryingEmbedder(e, retryCfg), true
	case "ollama":
		e := NewOllamaEmbedder(ollamaURL, ollamaModel, ollamaSocket)
		// Forced ollama but unreachable — still return with retry wrapper.
		return NewRetryingEmbedder(e, retryCfg), true
	case "bedrock":
		bedrockDims := 0
		if bedrockDimsStr != "" {
			if v, err := strconv.Atoi(bedrockDimsStr); err == nil {
				bedrockDims = v
			}
		}
		e, err := NewBedrockEmbedder(ctx, bedrockRegion, bedrockModel, bedrockDims)
		if err != nil {
			// AWS config failed (e.g. no credentials). Fall back to bundled
			// so the caller gets a clear error path rather than a nil embedder.
			return NewBundledEmbedder(defaultBedrockDims), false
		}
		// Bedrock benefits from longer backoffs due to AWS throttle windows.
		bedrockRetryCfg := RetryConfig{
			MaxRetries:  5,
			BaseBackoff: 500 * time.Millisecond,
		}
		return NewRetryingEmbedder(e, bedrockRetryCfg), true
	case "openai":
		openAIDims, err := strconv.Atoi(openAIDimsStr)
		if err != nil || openAIDims <= 0 || openAIModel == "" {
			return NewBundledEmbedder(ollamaDims), false
		}
		e := NewOpenAIEmbedder(openAIURL, openAIModel, openAIAPIKey, openAIDims)
		return NewRetryingEmbedder(e, retryCfg), true
	default: // "auto" or empty: probe TEI first, then Ollama, then bundled
		tei := NewTEIEmbedder(teiURL, teiModel, 0)
		if err := tei.Ping(ctx); err == nil {
			return NewRetryingEmbedder(tei, retryCfg), true
		}
		ollama := NewOllamaEmbedder(ollamaURL, ollamaModel, ollamaSocket)
		if err := ollama.Ping(ctx); err == nil {
			return NewRetryingEmbedder(ollama, retryCfg), true
		}
		return NewBundledEmbedder(ollamaDims), false
	}
}

// ---- CachingEmbedder -------------------------------------------------------

// DefaultCacheSize is the maximum number of query vectors held in memory.
// At 1024 dims × 4 bytes each, 512 entries ≈ 2 MB.
const DefaultCacheSize = 512

// defaultCacheMissEmbedTimeout bounds backend work for a coalesced cache miss.
// It prevents runaway retries when callers cancel or time out.
const defaultCacheMissEmbedTimeout = 30 * time.Second

// CachingEmbedder wraps an Embedder and caches query vectors in memory up to
// a fixed capacity. When the cache is full the oldest entry (by insertion
// order) is evicted before the new one is stored — a simple FIFO policy
// implemented with a []string queue alongside the map.
//
// This is valuable at query time: the MCP server is long-lived and agents
// often repeat identical queries within a session.
//
// EmbedBatch: when the inner embedder implements BatchEmbedder, EmbedBatch
// delegates directly to it and bypasses the cache (batch paths are used during
// indexing, not at query time). When the inner embedder does not implement
// BatchEmbedder, EmbedBatch falls back to calling Embed per item, which does
// populate and use the cache.
type CachingEmbedder struct {
	inner   Embedder
	maxSize int
	mu      sync.RWMutex
	cache   map[string][]float32
	keys    []string // insertion-order queue for FIFO eviction
	sf      singleflight.Group
	// backendTimeout bounds the singleflight backend call context.
	backendTimeout time.Duration
}

// NewCachingEmbedder wraps inner with a bounded query-vector cache of
// DefaultCacheSize entries.
func NewCachingEmbedder(inner Embedder) *CachingEmbedder {
	return NewCachingEmbedderSize(inner, DefaultCacheSize)
}

// NewCachingEmbedderSize wraps inner with a bounded query-vector cache of
// maxSize entries. When the cache is full the oldest entry is evicted (FIFO).
// maxSize <= 0 is replaced with DefaultCacheSize.
func NewCachingEmbedderSize(inner Embedder, maxSize int) *CachingEmbedder {
	if maxSize <= 0 {
		maxSize = DefaultCacheSize
	}
	return &CachingEmbedder{
		inner:          inner,
		maxSize:        maxSize,
		cache:          make(map[string][]float32, maxSize),
		keys:           make([]string, 0, maxSize),
		backendTimeout: defaultCacheMissEmbedTimeout,
	}
}

func (c *CachingEmbedder) Model() string { return c.inner.Model() }
func (c *CachingEmbedder) Dims() int     { return c.inner.Dims() }

func cloneVec(vec []float32) []float32 {
	if vec == nil {
		return nil
	}
	out := make([]float32, len(vec))
	copy(out, vec)
	return out
}

// Embed returns a cached vector if available, otherwise calls the inner
// embedder and stores the result. If the cache is at capacity the oldest
// entry is evicted (FIFO) before the new entry is inserted.
//
// Concurrency: concurrent callers for the same text are coalesced via
// singleflight — only one backend call is made per unique in-flight key.
// The backend call is decoupled from any one caller's cancellation to avoid
// leader-cancel fan-out, but is still bounded by backendTimeout so abandoned
// requests cannot run indefinitely. Each caller waits on DoChan and may bail
// early on its own ctx.Done().
func (c *CachingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	// Fast path: read lock only.
	c.mu.RLock()
	if vec, ok := c.cache[text]; ok {
		c.mu.RUnlock()
		return cloneVec(vec), nil
	}
	c.mu.RUnlock()

	// Slow path: coalesce concurrent misses via DoChan so each caller can
	// independently respect its own ctx while the single backend call runs
	// under a context that is not tied to any one caller.
	ch := c.sf.DoChan(text, func() (any, error) {
		// Re-check cache inside singleflight in case it was populated while
		// we were waiting to enter the group.
		c.mu.RLock()
		if vec, ok := c.cache[text]; ok {
			c.mu.RUnlock()
			return cloneVec(vec), nil
		}
		c.mu.RUnlock()

		// Decouple the backend call from any single caller, but bound total
		// backend work duration so canceled/abandoned requests do not run
		// indefinitely.
		backendCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), c.backendTimeout)
		defer cancel()
		vec, err := c.inner.Embed(backendCtx, text)
		if err != nil {
			return nil, err
		}

		c.mu.Lock()
		if cached, ok := c.cache[text]; ok {
			c.mu.Unlock()
			return cached, nil
		}
		if len(c.cache) >= c.maxSize {
			oldest := c.keys[0]
			delete(c.cache, oldest)
			// Keep queue capacity stable by shifting in-place, then writing
			// the new key into the last slot.
			copy(c.keys, c.keys[1:])
			c.keys[len(c.keys)-1] = text
		} else {
			c.keys = append(c.keys, text)
		}
		c.cache[text] = cloneVec(vec)
		c.mu.Unlock()

		return vec, nil
	})

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("embed: context done waiting for result: %w", ctx.Err())
	case res := <-ch:
		if res.Err != nil {
			return nil, res.Err
		}
		return cloneVec(res.Val.([]float32)), nil
	}
}

// Len returns the number of entries currently in the cache.
func (c *CachingEmbedder) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache)
}

// EmbedBatch delegates to the inner embedder's EmbedBatch when available,
// preserving batch support for indexing paths. When the inner embedder does
// not implement BatchEmbedder, it falls back to calling Embed per item so
// that CachingEmbedder always satisfies the BatchEmbedder interface.
//
// Fallback error handling: individual item failures leave vecs[i] as nil and
// are recorded; processing continues for the remaining items. To preserve
// partial progress for batch callers, a partial-failure batch returns vecs with
// a nil error. An error is returned only when every item in the batch fails.
func (c *CachingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if batcher, ok := c.inner.(BatchEmbedder); ok {
		return batcher.EmbedBatch(ctx, texts)
	}
	vecs := make([][]float32, len(texts))
	var firstErr error
	successes := 0
	for i, text := range texts {
		vec, err := c.Embed(ctx, text)
		if err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("embed batch: item %d: %w", i, err)
			}
			continue // leave vecs[i] nil; caller skips nil entries
		}
		vecs[i] = vec
		successes++
	}
	if successes == 0 && firstErr != nil {
		return vecs, firstErr
	}
	return vecs, nil
}
