package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newTestTEIServer creates an httptest server that mimics TEI's POST /embed endpoint.
func newTestTEIServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *TEIEmbedder) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	e := NewTEIEmbedder(srv.URL, "sentence-transformers/all-MiniLM-L6-v2", 384)
	return srv, e
}

// teiEmbedHandler returns a handler that echoes back one zero-vector per input.
func teiEmbedHandler(t *testing.T, wantInputCount int) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req teiEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if wantInputCount > 0 && len(req.Inputs) != wantInputCount {
			t.Errorf("TEI handler: got %d inputs, want %d", len(req.Inputs), wantInputCount)
		}

		// Return one 3-element vector per input for test simplicity.
		resp := make([][]float32, len(req.Inputs))
		for i := range resp {
			resp[i] = []float32{0.1, 0.2, 0.3}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func TestTEIEmbedder_Model(t *testing.T) {
	e := NewTEIEmbedder("http://localhost:8080", "sentence-transformers/all-MiniLM-L6-v2", 384)
	if got := e.Model(); got != "sentence-transformers/all-MiniLM-L6-v2" {
		t.Errorf("Model() = %q, want %q", got, "sentence-transformers/all-MiniLM-L6-v2")
	}
}

func TestTEIEmbedder_Dims(t *testing.T) {
	e := NewTEIEmbedder("http://localhost:8080", "sentence-transformers/all-MiniLM-L6-v2", 384)
	if got := e.Dims(); got != 384 {
		t.Errorf("Dims() = %d, want 384", got)
	}
}

func TestTEIEmbedder_Embed_Success(t *testing.T) {
	_, e := newTestTEIServer(t, teiEmbedHandler(t, 1))

	vec, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("Embed() returned %d elements, want 3", len(vec))
	}
	if vec[0] != 0.1 || vec[1] != 0.2 || vec[2] != 0.3 {
		t.Errorf("Embed() = %v, want [0.1 0.2 0.3]", vec)
	}
}

func TestTEIEmbedder_Embed_EmptyText(t *testing.T) {
	_, e := newTestTEIServer(t, teiEmbedHandler(t, 0))

	// Empty string should still produce a valid call.
	_, err := e.Embed(context.Background(), "")
	if err != nil {
		t.Fatalf("Embed() with empty string error = %v", err)
	}
}

func TestTEIEmbedder_Embed_ServerError(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for 500 response")
	}
}

func TestTEIEmbedder_Embed_InvalidJSON(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not json"))
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for invalid JSON")
	}
}

func TestTEIEmbedder_Embed_EmptyResponse(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([][]float32{})
	})

	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for empty embeddings list")
	}
}

func TestTEIEmbedder_Embed_ContextCancelled(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Never respond — context should cancel.
		<-r.Context().Done()
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := e.Embed(ctx, "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for cancelled context")
	}
}

func TestTEIEmbedder_EmbedBatch_Success(t *testing.T) {
	texts := []string{"alpha", "beta", "gamma"}
	_, e := newTestTEIServer(t, teiEmbedHandler(t, len(texts)))

	vecs, err := e.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch() error = %v", err)
	}
	if len(vecs) != len(texts) {
		t.Fatalf("EmbedBatch() returned %d vecs, want %d", len(vecs), len(texts))
	}
	for i, v := range vecs {
		if len(v) != 3 {
			t.Errorf("vecs[%d] len = %d, want 3", i, len(v))
		}
	}
}

func TestTEIEmbedder_EmbedBatch_Empty(t *testing.T) {
	_, e := newTestTEIServer(t, teiEmbedHandler(t, 0))

	vecs, err := e.EmbedBatch(context.Background(), nil)
	if err != nil {
		t.Fatalf("EmbedBatch(nil) error = %v", err)
	}
	if vecs != nil {
		t.Errorf("EmbedBatch(nil) = %v, want nil", vecs)
	}
}

func TestTEIEmbedder_EmbedBatch_MismatchedResponse(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Return fewer vectors than requested.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([][]float32{{0.1, 0.2, 0.3}})
	})

	texts := []string{"a", "b", "c"} // 3 inputs, only 1 returned
	_, err := e.EmbedBatch(context.Background(), texts)
	if err == nil {
		t.Fatal("EmbedBatch() error = nil, want error for mismatched response length")
	}
}

func TestTEIEmbedder_EmbedBatch_SingleItem(t *testing.T) {
	_, e := newTestTEIServer(t, teiEmbedHandler(t, 1))

	vecs, err := e.EmbedBatch(context.Background(), []string{"solo"})
	if err != nil {
		t.Fatalf("EmbedBatch() error = %v", err)
	}
	if len(vecs) != 1 {
		t.Errorf("EmbedBatch() = %d vecs, want 1", len(vecs))
	}
}

func TestTEIEmbedder_Ping_Success(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	})

	if err := e.Ping(context.Background()); err != nil {
		t.Errorf("Ping() error = %v, want nil", err)
	}
}

func TestTEIEmbedder_Ping_Failure(t *testing.T) {
	_, e := newTestTEIServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	})

	if err := e.Ping(context.Background()); err == nil {
		t.Error("Ping() error = nil, want error for 503")
	}
}

func TestDetect_RespectsEmbedBackendTEI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)

	t.Setenv("CTXPP_EMBED_BACKEND", "tei")
	t.Setenv("CTXPP_TEI_URL", srv.URL)

	e, usingExternal := Detect(context.Background())
	if !usingExternal {
		t.Error("CTXPP_EMBED_BACKEND=tei must return usingExternal=true")
	}
	tei, ok := e.(*RetryingEmbedder)
	if !ok {
		t.Fatalf("Detect() returned %T, want *RetryingEmbedder wrapping TEIEmbedder", e)
	}
	inner, ok := tei.inner.(*TEIEmbedder)
	if !ok {
		t.Fatalf("RetryingEmbedder.inner = %T, want *TEIEmbedder", tei.inner)
	}
	if inner.Model() != defaultTEIModel {
		t.Errorf("Model() = %q, want %q", inner.Model(), defaultTEIModel)
	}
}
