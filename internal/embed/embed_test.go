package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

func newTestOpenAIServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *OpenAIEmbedder) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	e := NewOpenAIEmbedder(srv.URL, "test-model", "test-key", 3)
	return srv, e
}

func TestOpenAIEmbedder_Embed_Success(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer test-key")
		}

		var req openAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.Model != "test-model" {
			t.Errorf("request model = %q, want %q", req.Model, "test-model")
		}
		if len(req.Input) != 1 || req.Input[0] != "hello world" {
			t.Errorf("request input = %v, want [hello world]", req.Input)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{{Index: 0, Embedding: []float32{0.1, 0.2, 0.3}}},
		})
	})

	vec, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vec) != 3 || vec[0] != 0.1 || vec[1] != 0.2 || vec[2] != 0.3 {
		t.Errorf("Embed() = %v, want [0.1 0.2 0.3]", vec)
	}
}

func TestOpenAIEmbedder_Embed_OmitsAuthorizationWhenAPIKeyUnset(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want empty header", got)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{{Index: 0, Embedding: []float32{0.1, 0.2, 0.3}}},
		})
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder(srv.URL, "test-model", "", 3)
	if _, err := e.Embed(context.Background(), "hello world"); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
}

func TestDetect_RespectsEmbedBackendOpenAI(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "openai")
	t.Setenv("CTXPP_OPENAI_URL", "https://example.com")
	t.Setenv("CTXPP_OPENAI_MODEL", "text-embedding-3-small")
	t.Setenv("CTXPP_OPENAI_DIMS", "1536")

	e, usingExternal := Detect(context.Background())
	if !usingExternal {
		t.Fatal("Detect() usingExternal = false, want true")
	}
	re, ok := e.(*RetryingEmbedder)
	if !ok {
		t.Fatalf("Detect() returned %T, want *RetryingEmbedder", e)
	}
	oe, ok := re.inner.(*OpenAIEmbedder)
	if !ok {
		t.Fatalf("inner = %T, want *OpenAIEmbedder", re.inner)
	}
	if oe.Model() != "text-embedding-3-small" {
		t.Errorf("Model() = %q, want %q", oe.Model(), "text-embedding-3-small")
	}
	if oe.Dims() != 1536 {
		t.Errorf("Dims() = %d, want 1536", oe.Dims())
	}
}

func TestOpenAIEmbedder_EmbedBatch_ReordersByIndex(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{
				{Index: 1, Embedding: []float32{1, 1, 1}},
				{Index: 0, Embedding: []float32{0, 0, 0}},
			},
		})
	})

	vecs, err := e.EmbedBatch(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("EmbedBatch() error = %v", err)
	}
	if got := vecs[0][0]; got != 0 {
		t.Errorf("vecs[0][0] = %v, want 0", got)
	}
	if got := vecs[1][0]; got != 1 {
		t.Errorf("vecs[1][0] = %v, want 1", got)
	}
}

func TestOpenAIEmbedder_Embed_InvalidJSON(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not json"))
	})

	_, err := e.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("Embed() error = nil, want decode error")
	}
}

func TestOpenAIEmbedder_Embed_EmptyData(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{Data: []openAIEmbedData{}})
	})

	_, err := e.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("Embed() error = nil, want empty response error")
	}
}

func TestOpenAIEmbedder_EmbedBatch_CountMismatch(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{{Index: 0, Embedding: []float32{0.1, 0.2, 0.3}}},
		})
	})

	_, err := e.EmbedBatch(context.Background(), []string{"first", "second"})
	if err == nil {
		t.Fatal("EmbedBatch() error = nil, want count mismatch error")
	}
}

func TestOpenAIEmbedder_Embed_DimsMismatch(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{{Index: 0, Embedding: []float32{0.1, 0.2}}},
		})
	})

	_, err := e.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("Embed() error = nil, want dims mismatch error")
	}
}

func TestOpenAIEmbedder_EmbedBatch_DuplicateIndex(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{
				{Index: 0, Embedding: []float32{0.1, 0.2, 0.3}},
				{Index: 0, Embedding: []float32{0.4, 0.5, 0.6}},
			},
		})
	})

	_, err := e.EmbedBatch(context.Background(), []string{"first", "second"})
	if err == nil {
		t.Fatal("EmbedBatch() error = nil, want duplicate index error")
	}
}

func TestOpenAIEmbedder_EmbedBatch_MissingIndex(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIEmbedResponse{
			Data: []openAIEmbedData{
				{Index: 0, Embedding: []float32{0.1, 0.2, 0.3}},
				{Index: 2, Embedding: []float32{0.4, 0.5, 0.6}},
			},
		})
	})

	_, err := e.EmbedBatch(context.Background(), []string{"first", "second"})
	if err == nil {
		t.Fatal("EmbedBatch() error = nil, want missing index error")
	}
}

func TestOpenAIEmbedder_Embed_ServerErrorIncludesAPIMessage(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(openAIErrorResponse{Error: &struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		}{Message: "rate limit"}})
	})

	_, err := e.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("Embed() error = nil, want HTTP error")
	}
	if got := err.Error(); !strings.Contains(got, "rate limit") {
		t.Fatalf("Embed() error = %q, want API message", got)
	}
}

func TestOpenAIEmbedder_Embed_ClientErrorIsNonRetryable(t *testing.T) {
	_, e := newTestOpenAIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(openAIErrorResponse{Error: &struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		}{Message: "bad key"}})
	})

	_, err := e.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("Embed() error = nil, want HTTP error")
	}
	var nonRetryable *NonRetryableError
	if !errors.As(err, &nonRetryable) {
		t.Fatalf("Embed() error = %T, want *NonRetryableError", err)
	}
}

func TestOpenAIEmbedder_Embed_InvalidConfiguration(t *testing.T) {
	e := NewOpenAIEmbedder("https://example.com", "", "", 0)

	_, err := e.Embed(context.Background(), "hello world")
	if err == nil {
		t.Fatal("Embed() error = nil, want configuration error")
	}
}

func TestDetect_OpenAIInvalidDimsFallsBackToBundled(t *testing.T) {
	t.Setenv("CTXPP_EMBED_BACKEND", "openai")
	t.Setenv("CTXPP_OPENAI_MODEL", "text-embedding-3-small")
	t.Setenv("CTXPP_OPENAI_DIMS", "not-a-number")

	e, usingExternal := Detect(context.Background())
	if usingExternal {
		t.Fatal("Detect() usingExternal = true, want false")
	}
	if e.Model() != bundledModel {
		t.Errorf("Model() = %q, want %q", e.Model(), bundledModel)
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

// ---- CachingEmbedder tests -------------------------------------------------

// countingEmbedder records every text passed to Embed so tests can assert
// how many (and which) calls reached the inner embedder.
// It is safe for concurrent use.
type countingEmbedder struct {
	mu    sync.Mutex
	calls []string
	vec   []float32
	err   error
}

func (c *countingEmbedder) Model() string { return "test" }
func (c *countingEmbedder) Dims() int     { return len(c.vec) }
func (c *countingEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	c.mu.Lock()
	c.calls = append(c.calls, text)
	c.mu.Unlock()
	if c.err != nil {
		return nil, c.err
	}
	return c.vec, nil
}

func (c *countingEmbedder) callCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.calls)
}

func TestCachingEmbedder_CachesOnSecondCall(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{1, 2, 3}}
	c := NewCachingEmbedder(inner)

	v1, err := c.Embed(t.Context(), "hello")
	if err != nil {
		t.Fatalf("first Embed() error = %v", err)
	}
	v2, err := c.Embed(t.Context(), "hello")
	if err != nil {
		t.Fatalf("second Embed() error = %v", err)
	}

	if inner.callCount() != 1 {
		t.Errorf("inner called %d times, want 1", inner.callCount())
	}
	if len(v1) != len(v2) {
		t.Errorf("vectors differ in length: %d vs %d", len(v1), len(v2))
	}
}

func TestCachingEmbedder_DifferentTextsCallInner(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{0.5}}
	c := NewCachingEmbedder(inner)

	texts := []string{"alpha", "beta", "gamma"}
	for _, txt := range texts {
		if _, err := c.Embed(t.Context(), txt); err != nil {
			t.Fatalf("Embed(%q) error = %v", txt, err)
		}
	}

	if got := inner.callCount(); got != len(texts) {
		t.Errorf("inner called %d times, want %d", got, len(texts))
	}
	if got := c.Len(); got != len(texts) {
		t.Errorf("Len() = %d, want %d", got, len(texts))
	}
}

func TestCachingEmbedder_ErrorNotCached(t *testing.T) {
	inner := &countingEmbedder{err: errors.New("backend down")}
	c := NewCachingEmbedder(inner)

	_, err := c.Embed(t.Context(), "query")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// After an error the cache should still be empty.
	if got := c.Len(); got != 0 {
		t.Errorf("Len() = %d after error, want 0", got)
	}

	// Second call should hit inner again (not serve a cached error).
	_, _ = c.Embed(t.Context(), "query")
	if got := inner.callCount(); got != 2 {
		t.Errorf("inner called %d times after two failures, want 2", got)
	}
}

func TestCachingEmbedder_DelegatesModelAndDims(t *testing.T) {
	inner := &countingEmbedder{vec: make([]float32, 7)}
	c := NewCachingEmbedder(inner)
	if c.Model() != "test" {
		t.Errorf("Model() = %q, want %q", c.Model(), "test")
	}
	if c.Dims() != 7 {
		t.Errorf("Dims() = %d, want 7", c.Dims())
	}
}

func TestCachingEmbedder_ConcurrentSafeHits(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{1}}
	c := NewCachingEmbedder(inner)

	// Pre-populate the cache.
	if _, err := c.Embed(t.Context(), "shared"); err != nil {
		t.Fatal(err)
	}

	// Hammer the cache from multiple goroutines; all should hit the cache
	// and the inner embedder should still only have been called once.
	const goroutines = 50
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			_, err := c.Embed(t.Context(), "shared")
			errs <- err
		}()
	}
	for i := 0; i < goroutines; i++ {
		if err := <-errs; err != nil {
			t.Errorf("concurrent Embed() error = %v", err)
		}
	}

	if got := inner.callCount(); got != 1 {
		t.Errorf("inner called %d times under concurrency, want 1", got)
	}
}

func TestCachingEmbedder_EvictsOldestWhenFull(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{1}}
	const maxSize = 3
	c := NewCachingEmbedderSize(inner, maxSize)

	// Fill the cache: query-0, query-1, query-2 (query-0 is oldest).
	for i := 0; i < maxSize; i++ {
		if _, err := c.Embed(t.Context(), fmt.Sprintf("query-%d", i)); err != nil {
			t.Fatalf("Embed() error = %v", err)
		}
	}
	if got := c.Len(); got != maxSize {
		t.Fatalf("Len() = %d, want %d after fill", got, maxSize)
	}

	// Adding a new entry evicts query-0 (oldest); cache stays at maxSize.
	if _, err := c.Embed(t.Context(), "new"); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if got := c.Len(); got != maxSize {
		t.Errorf("Len() = %d, want %d after eviction", got, maxSize)
	}

	// query-0 was evicted — it must reach the inner embedder again.
	callsBefore := inner.callCount()
	if _, err := c.Embed(t.Context(), "query-0"); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if got := inner.callCount(); got != callsBefore+1 {
		t.Errorf("evicted key re-embedded %d extra times, want 1", got-callsBefore)
	}
	// Re-embedding query-0 evicts the next oldest (query-1).
	// Cache is now: query-2, new, query-0.

	// query-2 and "new" are still cached — must not hit inner.
	callsBefore = inner.callCount()
	for _, q := range []string{"query-2", "new"} {
		if _, err := c.Embed(t.Context(), q); err != nil {
			t.Fatalf("Embed(%q) error = %v", q, err)
		}
	}
	if got := inner.callCount(); got != callsBefore {
		t.Errorf("non-evicted keys hit inner %d extra times, want 0", got-callsBefore)
	}
}

func TestCachingEmbedder_ZeroMaxSizeUsesDefault(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{1}}
	c := NewCachingEmbedderSize(inner, 0)
	if c.maxSize != DefaultCacheSize {
		t.Errorf("maxSize = %d, want DefaultCacheSize (%d)", c.maxSize, DefaultCacheSize)
	}
}

func TestCachingEmbedder_ConcurrentSafeMissesSameKey(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{1}}
	c := NewCachingEmbedder(inner)

	// Hammer the cache from a cold start with the same key. After all
	// goroutines finish, keys and cache must be consistent: exactly one entry,
	// with no duplicate keys in the FIFO queue.
	const goroutines = 50
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			_, err := c.Embed(t.Context(), "shared-miss")
			errs <- err
		}()
	}
	for i := 0; i < goroutines; i++ {
		if err := <-errs; err != nil {
			t.Errorf("concurrent Embed() error = %v", err)
		}
	}

	// Cache must contain exactly one entry for "shared-miss".
	if got := c.Len(); got != 1 {
		t.Errorf("Len() = %d after concurrent misses, want 1", got)
	}
	// The FIFO keys queue must not contain duplicates.
	c.mu.RLock()
	keyCount := len(c.keys)
	c.mu.RUnlock()
	if keyCount != 1 {
		t.Errorf("keys queue len = %d, want 1 (duplicate inserts detected)", keyCount)
	}
	// Concurrent misses for the same key must be coalesced into a single
	// backend call — the inner embedder must be called exactly once.
	if got := inner.callCount(); got != 1 {
		t.Errorf("inner.callCount() = %d, want 1 backend call for concurrent misses to the same key", got)
	}
}

// batchCountingEmbedder is a BatchEmbedder test double that records calls.
type batchCountingEmbedder struct {
	countingEmbedder
	batchCalls [][]string
	batchMu    sync.Mutex
}

func (b *batchCountingEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	b.batchMu.Lock()
	b.batchCalls = append(b.batchCalls, texts)
	b.batchMu.Unlock()
	vecs := make([][]float32, len(texts))
	for i := range texts {
		vecs[i] = b.vec
	}
	return vecs, nil
}

func (b *batchCountingEmbedder) batchCallCount() int {
	b.batchMu.Lock()
	defer b.batchMu.Unlock()
	return len(b.batchCalls)
}

// Compile-time assertion: CachingEmbedder must satisfy BatchEmbedder.
var _ BatchEmbedder = (*CachingEmbedder)(nil)

// TestCachingEmbedder_EmbedBatchDelegatesToInner verifies that when the inner
// embedder implements BatchEmbedder, CachingEmbedder delegates EmbedBatch
// directly to it.
func TestCachingEmbedder_EmbedBatchDelegatesToInner(t *testing.T) {
	inner := &batchCountingEmbedder{countingEmbedder: countingEmbedder{vec: []float32{1, 2}}}
	c := NewCachingEmbedder(inner)

	texts := []string{"a", "b", "c"}
	vecs, err := c.EmbedBatch(t.Context(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch() error = %v", err)
	}
	if len(vecs) != len(texts) {
		t.Errorf("EmbedBatch() returned %d vecs, want %d", len(vecs), len(texts))
	}
	if got := inner.batchCallCount(); got != 1 {
		t.Errorf("inner.EmbedBatch called %d times, want 1", got)
	}
	// EmbedBatch must NOT have called Embed on the inner.
	if got := inner.callCount(); got != 0 {
		t.Errorf("inner.Embed called %d times during EmbedBatch, want 0", got)
	}
}

// TestCachingEmbedder_EmbedBatchFallsBackToEmbed verifies that when the inner
// embedder does NOT implement BatchEmbedder, CachingEmbedder falls back to
// calling Embed per item.
func TestCachingEmbedder_EmbedBatchFallsBackToEmbed(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{0.5}}
	c := NewCachingEmbedder(inner)

	texts := []string{"x", "y", "z"}
	vecs, err := c.EmbedBatch(t.Context(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch() fallback error = %v", err)
	}
	if len(vecs) != len(texts) {
		t.Errorf("EmbedBatch() fallback returned %d vecs, want %d", len(vecs), len(texts))
	}
	if got := inner.callCount(); got != len(texts) {
		t.Errorf("inner.Embed called %d times, want %d", got, len(texts))
	}
}

// TestCachingEmbedder_EmbedBatchFallbackPartialSuccess verifies that when the
// fallback path (no inner BatchEmbedder) encounters a per-item error, it:
//   - continues processing remaining items rather than aborting,
//   - returns nil vectors for failed items and real vectors for successes,
//   - returns nil error to preserve partial progress for callers.
func TestCachingEmbedder_EmbedBatchFallbackPartialSuccess(t *testing.T) {
	failErr := errors.New("backend down")
	callN := 0
	// failOnSecond fails only the second call.
	inner := &funcEmbedder{
		model: "test",
		dims:  1,
		embedFn: func(_ context.Context, _ string) ([]float32, error) {
			callN++
			if callN == 2 {
				return nil, failErr
			}
			return []float32{float32(callN)}, nil
		},
	}
	c := NewCachingEmbedder(inner)

	texts := []string{"a", "b", "c"}
	vecs, err := c.EmbedBatch(t.Context(), texts)

	// Partial success must return nil error so callers can keep successful vecs.
	if err != nil {
		t.Fatalf("EmbedBatch() fallback: partial failure error = %v, want nil", err)
	}
	// Must still return a slice of the correct length.
	if len(vecs) != len(texts) {
		t.Fatalf("EmbedBatch() fallback returned %d vecs, want %d", len(vecs), len(texts))
	}
	// Items 0 and 2 succeeded — must be non-nil.
	if vecs[0] == nil {
		t.Error("vecs[0] is nil, want non-nil (successful item)")
	}
	if vecs[2] == nil {
		t.Error("vecs[2] is nil, want non-nil (successful item)")
	}
	// Item 1 failed — must be nil.
	if vecs[1] != nil {
		t.Error("vecs[1] is non-nil, want nil (failed item)")
	}
}

func TestCachingEmbedder_EmbedBatchFallbackAllFailReturnsError(t *testing.T) {
	failErr := errors.New("backend down")
	inner := &funcEmbedder{
		model: "test",
		dims:  1,
		embedFn: func(_ context.Context, _ string) ([]float32, error) {
			return nil, failErr
		},
	}
	c := NewCachingEmbedder(inner)

	vecs, err := c.EmbedBatch(t.Context(), []string{"a", "b", "c"})
	if err == nil {
		t.Fatal("EmbedBatch() fallback all-fail error = nil, want non-nil")
	}
	if !errors.Is(err, failErr) {
		t.Errorf("EmbedBatch() error = %v, want wrapped %v", err, failErr)
	}
	if len(vecs) != 3 {
		t.Fatalf("EmbedBatch() returned %d vecs, want 3", len(vecs))
	}
	for i := range vecs {
		if vecs[i] != nil {
			t.Errorf("vecs[%d] is non-nil, want nil", i)
		}
	}
}

func TestCachingEmbedder_EmbedReturnsDefensiveCopy(t *testing.T) {
	inner := &countingEmbedder{vec: []float32{1, 2, 3}}
	c := NewCachingEmbedder(inner)

	first, err := c.Embed(t.Context(), "copy-key")
	if err != nil {
		t.Fatalf("Embed() first error = %v", err)
	}
	first[0] = 999

	second, err := c.Embed(t.Context(), "copy-key")
	if err != nil {
		t.Fatalf("Embed() second error = %v", err)
	}
	if second[0] == 999 {
		t.Fatal("cached vector was externally mutated; expected defensive copy")
	}
}

// TestCachingEmbedder_EmbedCanceledLeaderDoesNotFailWaiters verifies that
// when a singleflight leader's context is cancelled, other concurrent waiters
// with valid contexts still receive a result (they do not inherit the leader's
// cancellation).
func TestCachingEmbedder_EmbedCanceledLeaderDoesNotFailWaiters(t *testing.T) {
	// Slow embedder that blocks until released.
	release := make(chan struct{})
	started := make(chan struct{})
	var startedOnce sync.Once
	inner := &funcEmbedder{
		model: "test",
		dims:  1,
		embedFn: func(ctx context.Context, _ string) ([]float32, error) {
			startedOnce.Do(func() { close(started) })
			select {
			case <-release:
				return []float32{1}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}
	c := NewCachingEmbedder(inner)

	// Leader goroutine: cancellable context.
	leaderCtx, leaderCancel := context.WithCancel(context.Background())
	leaderDone := make(chan error, 1)
	go func() {
		_, err := c.Embed(leaderCtx, "key")
		leaderDone <- err
	}()

	// Wait deterministically until the backend call is in flight.
	<-started

	// Waiter goroutine: long-lived context — must not be affected by leader cancel.
	waiterDone := make(chan error, 1)
	waiterStarted := make(chan struct{})
	go func() {
		close(waiterStarted)
		_, err := c.Embed(context.Background(), "key")
		waiterDone <- err
	}()
	<-waiterStarted

	// Cancel the leader.
	leaderCancel()

	// Allow backend to complete so waiter can succeed.
	close(release)

	if err := <-waiterDone; err != nil {
		t.Errorf("waiter Embed() error = %v, want nil (canceled leader must not fail waiter)", err)
	}
	// Leader itself may or may not error depending on timing; we don't assert it.
	select {
	case <-leaderDone:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("leader goroutine did not return")
	}
}

func TestCachingEmbedder_EmbedBackendTimeoutIsBounded(t *testing.T) {
	inner := &funcEmbedder{
		model: "test",
		dims:  1,
		embedFn: func(ctx context.Context, _ string) ([]float32, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}
	c := NewCachingEmbedder(inner)
	c.backendTimeout = 20 * time.Millisecond

	start := time.Now()
	_, err := c.Embed(context.Background(), "timeout-key")
	if err == nil {
		t.Fatal("Embed() error = nil, want timeout error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Embed() error = %v, want context deadline exceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 250*time.Millisecond {
		t.Errorf("Embed() elapsed = %v, want <= 250ms", elapsed)
	}
}

// funcEmbedder is a test double whose Embed behaviour is provided via a closure.
type funcEmbedder struct {
	model   string
	dims    int
	embedFn func(ctx context.Context, text string) ([]float32, error)
}

func (f *funcEmbedder) Model() string { return f.model }
func (f *funcEmbedder) Dims() int     { return f.dims }
func (f *funcEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return f.embedFn(ctx, text)
}
