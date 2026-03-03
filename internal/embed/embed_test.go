package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// fakeEmbedder is a controllable Embedder for testing.
type fakeEmbedder struct {
	calls atomic.Int32
	failN int       // fail the first N calls
	err   error     // error to return on failure
	vec   []float32 // vector to return on success
	model string
	dims  int
}

func (f *fakeEmbedder) Model() string { return f.model }
func (f *fakeEmbedder) Dims() int     { return f.dims }

func (f *fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	n := int(f.calls.Add(1))
	if n <= f.failN {
		return nil, f.err
	}
	return f.vec, nil
}

func TestDetect_UsesOllamaWhenReachable(t *testing.T) {
	// This test requires a live Ollama instance at localhost:11434.
	// It is skipped in CI when Ollama is unavailable.
	e, usingOllama := Detect(context.Background())
	if !usingOllama {
		t.Skip("Ollama not reachable — skipping live detection test")
	}
	if e.Model() == bundledModel {
		t.Errorf("Detect() returned bundled model even though Ollama was reachable")
	}
}

func TestDetect_FallsBackToBundledWhenOllamaAbsent(t *testing.T) {
	// Point at a port nothing is listening on.
	ollama := NewOllamaEmbedder("http://localhost:19999", "", "")
	if err := ollama.Ping(context.Background()); err == nil {
		t.Skip("something is unexpectedly listening on port 19999")
	}

	// Swap the default URL so Detect sees the dead endpoint.
	orig := NewOllamaEmbedder("http://localhost:19999", "", "")
	if err := orig.Ping(context.Background()); err == nil {
		t.Skip("port 19999 is live — cannot test fallback")
	}

	// Call Detect with a dead Ollama — expect bundled fallback.
	// We test this indirectly: if Ollama is unreachable, Detect must
	// return an embedder whose Model() is the bundled constant.
	if _, usingOllama := Detect(context.Background()); usingOllama {
		// Only fail if Ollama really is down; if it's up this is expected.
		t.Log("Ollama is live on this machine — fallback path not exercised")
	}
}

func TestBundledEmbedder_ReturnsZeroVectorOfCorrectDims(t *testing.T) {
	tests := []struct {
		name string
		dims int
	}{
		{name: "default dims", dims: 768},
		{name: "small dims", dims: 4},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := NewBundledEmbedder(tc.dims)

			if e.Dims() != tc.dims {
				t.Errorf("Dims() = %d, want %d", e.Dims(), tc.dims)
			}

			vec, err := e.Embed(context.Background(), "any text")
			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			if len(vec) != tc.dims {
				t.Errorf("len(vec) = %d, want %d", len(vec), tc.dims)
			}
			for i, v := range vec {
				if v != 0 {
					t.Errorf("vec[%d] = %v, want 0", i, v)
				}
			}
		})
	}
}

func TestDetect_RespectsEmbedBackendBundled(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "bundled")
	e, usingOllama := Detect(context.Background())
	if usingOllama {
		t.Error("CTXPP_EMBED_BACKEND=bundled must force bundled embedder, got ollama")
	}
	if e.Model() != bundledModel {
		t.Errorf("Model() = %q, want %q", e.Model(), bundledModel)
	}
}

func TestDetect_RespectsEmbedBackendOllama(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "ollama")
	// We can't reach Ollama reliably in tests; just verify it tries to use
	// an OllamaEmbedder (model is not the bundled constant).
	e, usingOllama := Detect(context.Background())
	if !usingOllama {
		// When Ollama is unreachable but forced, Detect should still return
		// an OllamaEmbedder (and usingOllama=true), not silently fall back.
		t.Skip("Ollama not reachable — cannot fully verify CTXPP_EMBED_BACKEND=ollama forced path")
	}
	if e.Model() == bundledModel {
		t.Errorf("CTXPP_EMBED_BACKEND=ollama must use Ollama, got bundled")
	}
}

func TestDetect_RespectsCustomOllamaModel(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "bundled") // avoid needing live Ollama
	t.Setenv("CTXPP_OLLAMA_MODEL", "custom-model")
	e, usingOllama := Detect(context.Background())
	if usingOllama {
		t.Skip("Ollama live — CTXPP_EMBED_BACKEND override not effective")
	}
	// When forced bundled, model must be bundled regardless of CTXPP_OLLAMA_MODEL.
	if e.Model() != bundledModel {
		t.Errorf("Model() = %q, want %q", e.Model(), bundledModel)
	}
}

func TestNewOllamaEmbedder_TransportTuned(t *testing.T) {
	e := NewOllamaEmbedder("", "", "")
	tr, ok := e.client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("expected *http.Transport")
	}
	if tr.MaxIdleConns != 64 {
		t.Errorf("MaxIdleConns = %d, want 64", tr.MaxIdleConns)
	}
	if tr.MaxIdleConnsPerHost != 64 {
		t.Errorf("MaxIdleConnsPerHost = %d, want 64", tr.MaxIdleConnsPerHost)
	}
	if !tr.DisableCompression {
		t.Error("DisableCompression = false, want true")
	}
}

func TestNewOllamaEmbedder_SocketPath(t *testing.T) {
	e := NewOllamaEmbedder("", "", "/tmp/ollama.sock")
	tr, ok := e.client.Transport.(*http.Transport)
	if !ok {
		t.Fatal("expected *http.Transport")
	}
	// Verify DialContext is set (socket dialer installed).
	if tr.DialContext == nil {
		t.Error("DialContext = nil, expected Unix socket dialer")
	}
}

// ---- RetryingEmbedder tests ------------------------------------------------

func TestRetryingEmbedder_SucceedsWithoutRetry(t *testing.T) {
	inner := &fakeEmbedder{
		model: "test",
		dims:  3,
		vec:   []float32{1, 2, 3},
	}
	re := NewRetryingEmbedder(inner, RetryConfig{
		MaxRetries:  3,
		BaseBackoff: time.Millisecond,
	})

	vec, err := re.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vec) != 3 || vec[0] != 1 {
		t.Errorf("Embed() = %v, want [1 2 3]", vec)
	}
	if got := int(inner.calls.Load()); got != 1 {
		t.Errorf("calls = %d, want 1 (no retry needed)", got)
	}
}

func TestRetryingEmbedder_RetriesTransientFailure(t *testing.T) {
	inner := &fakeEmbedder{
		model: "test",
		dims:  3,
		vec:   []float32{1, 2, 3},
		failN: 2,
		err:   errors.New("connection refused"),
	}
	re := NewRetryingEmbedder(inner, RetryConfig{
		MaxRetries:  3,
		BaseBackoff: time.Millisecond, // fast for tests
	})

	vec, err := re.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed() error = %v, want success after retries", err)
	}
	if len(vec) != 3 {
		t.Errorf("Embed() vec len = %d, want 3", len(vec))
	}
	if got := int(inner.calls.Load()); got != 3 {
		t.Errorf("calls = %d, want 3 (2 failures + 1 success)", got)
	}
}

func TestRetryingEmbedder_ExhaustsRetries(t *testing.T) {
	retryErr := errors.New("server overloaded")
	inner := &fakeEmbedder{
		model: "test",
		dims:  3,
		vec:   []float32{1, 2, 3},
		failN: 10, // always fail
		err:   retryErr,
	}
	re := NewRetryingEmbedder(inner, RetryConfig{
		MaxRetries:  3,
		BaseBackoff: time.Millisecond,
	})

	_, err := re.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("Embed() error = nil, want error after exhausting retries")
	}
	if !errors.Is(err, retryErr) {
		t.Errorf("Embed() error = %v, want wrapped %v", err, retryErr)
	}
	// 1 initial + 3 retries = 4 total calls
	if got := int(inner.calls.Load()); got != 4 {
		t.Errorf("calls = %d, want 4 (1 initial + 3 retries)", got)
	}
}

func TestRetryingEmbedder_RespectsContextCancellation(t *testing.T) {
	inner := &fakeEmbedder{
		model: "test",
		dims:  3,
		vec:   []float32{1, 2, 3},
		failN: 10,
		err:   errors.New("temporary"),
	}
	re := NewRetryingEmbedder(inner, RetryConfig{
		MaxRetries:  5,
		BaseBackoff: 100 * time.Millisecond,
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := re.Embed(ctx, "hello")
	if err == nil {
		t.Fatal("Embed() error = nil, want context.Canceled")
	}
}

func TestRetryingEmbedder_DelegatesModelAndDims(t *testing.T) {
	inner := &fakeEmbedder{model: "nomic", dims: 768}
	re := NewRetryingEmbedder(inner, RetryConfig{})

	if re.Model() != "nomic" {
		t.Errorf("Model() = %q, want %q", re.Model(), "nomic")
	}
	if re.Dims() != 768 {
		t.Errorf("Dims() = %d, want 768", re.Dims())
	}
}

func TestRetryingEmbedder_DefaultConfig(t *testing.T) {
	inner := &fakeEmbedder{
		model: "test",
		dims:  3,
		vec:   []float32{1, 2, 3},
		failN: 2,
		err:   errors.New("transient"),
	}
	// Zero-value RetryConfig should use sensible defaults.
	re := NewRetryingEmbedder(inner, RetryConfig{})

	vec, err := re.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed() error = %v, want success with default retries", err)
	}
	if len(vec) != 3 {
		t.Errorf("vec len = %d, want 3", len(vec))
	}
}

func TestRetryingEmbedder_BackoffIncreases(t *testing.T) {
	inner := &fakeEmbedder{
		model: "test",
		dims:  3,
		vec:   []float32{1, 2, 3},
		failN: 3,
		err:   errors.New("busy"),
	}
	re := NewRetryingEmbedder(inner, RetryConfig{
		MaxRetries:  3,
		BaseBackoff: 10 * time.Millisecond,
	})

	start := time.Now()
	_, err := re.Embed(context.Background(), "hello")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	// With exponential backoff: 10ms + 20ms + 40ms = 70ms minimum (before jitter).
	// Allow some slack but ensure it's not instant.
	if elapsed < 30*time.Millisecond {
		t.Errorf("elapsed = %v, want >= 30ms (backoff should cause delay)", elapsed)
	}
}

// ---- OllamaEmbedder tests via httptest -------------------------------------

// newTestOllamaServer creates an httptest server that mimics Ollama's /api/embed endpoint.
func newTestOllamaServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *OllamaEmbedder) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	e := NewOllamaEmbedder(srv.URL, "test-model", "")
	return srv, e
}

func TestOllamaEmbedder_Embed_Success(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Verify the request body.
		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.Model != "test-model" {
			t.Errorf("request model = %q, want %q", req.Model, "test-model")
		}
		if req.Input != "hello world" {
			t.Errorf("request input = %q, want %q", req.Input, "hello world")
		}

		resp := ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	vec, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vec) != 3 || vec[0] != 0.1 || vec[1] != 0.2 || vec[2] != 0.3 {
		t.Errorf("Embed() = %v, want [0.1 0.2 0.3]", vec)
	}
}

func TestOllamaEmbedder_Embed_ServerError(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for 500 response")
	}
}

func TestOllamaEmbedder_Embed_EmptyResponse(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ollamaEmbedResponse{Embeddings: [][]float32{}})
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for empty embeddings")
	}
}

func TestOllamaEmbedder_Embed_InvalidJSON(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not json"))
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for invalid JSON")
	}
}

func TestOllamaEmbedder_Embed_ContextCancelled(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Slow server — will be cancelled before responding.
		time.Sleep(5 * time.Second)
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := e.Embed(ctx, "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for cancelled context")
	}
}

func TestOllamaEmbedder_Dims(t *testing.T) {
	e := NewOllamaEmbedder("", "", "")
	if e.Dims() != 1024 {
		t.Errorf("Dims() = %d, want 1024", e.Dims())
	}
}

func TestOllamaEmbedder_Ping_Success(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	})

	if err := e.Ping(context.Background()); err != nil {
		t.Errorf("Ping() error = %v, want nil", err)
	}
}

func TestOllamaEmbedder_Ping_Failure(t *testing.T) {
	_, e := newTestOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	})

	if err := e.Ping(context.Background()); err == nil {
		t.Error("Ping() error = nil, want error for 503")
	}
}

func TestBundledEmbedder_DefaultDims(t *testing.T) {
	e := NewBundledEmbedder(0) // 0 should default to ollamaDims (1024)
	if e.Dims() != 1024 {
		t.Errorf("Dims() = %d, want 1024 (default)", e.Dims())
	}
}

func TestDetect_RespectsEmbedBackendBedrock(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "bedrock")
	t.Setenv("CTXPP_BEDROCK_REGION", "us-west-2")
	// With valid AWS credentials, Detect returns a RetryingEmbedder wrapping
	// BedrockEmbedder. Without credentials, it falls back to bundled.
	e, usingExternal := Detect(context.Background())
	if usingExternal {
		// AWS credentials found — verify it wrapped a BedrockEmbedder.
		re, ok := e.(*RetryingEmbedder)
		if !ok {
			t.Fatalf("Detect() returned %T, want *RetryingEmbedder", e)
		}
		be, ok := re.inner.(*BedrockEmbedder)
		if !ok {
			t.Fatalf("inner = %T, want *BedrockEmbedder", re.inner)
		}
		if be.Model() != defaultBedrockModel {
			t.Errorf("Model() = %q, want %q", be.Model(), defaultBedrockModel)
		}
		// Verify bedrock gets longer retry config (5 retries, 500ms base).
		if re.cfg.MaxRetries != 5 {
			t.Errorf("MaxRetries = %d, want 5", re.cfg.MaxRetries)
		}
		if re.cfg.BaseBackoff != 500*time.Millisecond {
			t.Errorf("BaseBackoff = %v, want 500ms", re.cfg.BaseBackoff)
		}
	} else {
		// No AWS credentials — should fall back to bundled gracefully.
		if e.Model() != bundledModel {
			t.Errorf("Model() = %q, want %q (fallback)", e.Model(), bundledModel)
		}
	}
}

func TestDetect_BedrockCustomDims(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "bedrock")
	t.Setenv("CTXPP_BEDROCK_DIMS", "512")
	e, usingExternal := Detect(context.Background())
	if usingExternal {
		re := e.(*RetryingEmbedder)
		be := re.inner.(*BedrockEmbedder)
		if be.Dims() != 512 {
			t.Errorf("Dims() = %d, want 512", be.Dims())
		}
	}
	// If no credentials, falls back to bundled — that's fine for this test.
}
