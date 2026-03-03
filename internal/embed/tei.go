package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ---- TEI (Hugging Face Text Embeddings Inference) --------------------------

const defaultTEIURL = "http://localhost:8080"
const defaultTEIModel = "sentence-transformers/all-MiniLM-L6-v2"
const defaultTEIDims = 384

// teiEmbedRequest is the JSON body sent to TEI's POST /embed endpoint.
type teiEmbedRequest struct {
	Inputs []string `json:"inputs"`
}

// TEIEmbedder calls the Hugging Face Text Embeddings Inference (TEI) server.
// TEI is a purpose-built, high-throughput embedding server with GPU-optimised
// kernels (Flash Attention, cuBLASLt) and token-level dynamic batching.
//
// Endpoint: POST /embed  (accepts {"inputs": ["text1", "text2", ...]})
// Response: [][]float32
//
// Start TEI locally with Docker:
//
//	docker run --gpus all -p 8080:80 \
//	  ghcr.io/huggingface/text-embeddings-inference:turing-latest \
//	  --model-id sentence-transformers/all-MiniLM-L6-v2
type TEIEmbedder struct {
	baseURL string
	model   string
	dims    int
	client  *http.Client
}

// NewTEIEmbedder constructs a TEIEmbedder.
// If baseURL is empty, http://localhost:8080 is used.
// If model is empty, sentence-transformers/all-MiniLM-L6-v2 is used.
// If dims is <= 0, 384 (all-MiniLM-L6-v2 dimensionality) is used.
func NewTEIEmbedder(baseURL, model string, dims int) *TEIEmbedder {
	if baseURL == "" {
		baseURL = defaultTEIURL
	}
	if model == "" {
		model = defaultTEIModel
	}
	if dims <= 0 {
		dims = defaultTEIDims
	}
	return &TEIEmbedder{
		baseURL: baseURL,
		model:   model,
		dims:    dims,
		client: &http.Client{
			Timeout: 120 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        64,
				MaxIdleConnsPerHost: 64,
				IdleConnTimeout:     120 * time.Second,
				DisableCompression:  true,
				ForceAttemptHTTP2:   false,
			},
		},
	}
}

// Model returns the embedding model identifier.
func (e *TEIEmbedder) Model() string { return e.model }

// Dims returns the embedding dimensionality.
func (e *TEIEmbedder) Dims() int { return e.dims }

// Embed returns the embedding vector for a single text.
func (e *TEIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.doEmbed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("tei embed: empty response")
	}
	return vecs[0], nil
}

// EmbedBatch returns embedding vectors for multiple texts in a single HTTP call.
// The returned slice has the same length as texts. This is the primary interface
// for high-throughput GPU batching — send large batches (100–2000) for best
// GPU utilisation.
func (e *TEIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	vecs, err := e.doEmbed(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("tei embed batch: got %d embeddings for %d inputs", len(vecs), len(texts))
	}
	return vecs, nil
}

// doEmbed sends a POST /embed request with the given inputs and parses the response.
func (e *TEIEmbedder) doEmbed(ctx context.Context, inputs []string) ([][]float32, error) {
	body, err := json.Marshal(teiEmbedRequest{Inputs: inputs})
	if err != nil {
		return nil, fmt.Errorf("tei embed: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("tei embed: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tei embed: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tei embed: status %d", resp.StatusCode)
	}

	var result [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("tei embed: decode: %w", err)
	}
	return result, nil
}

// Ping checks whether the TEI server is reachable by calling GET /health.
func (e *TEIEmbedder) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("tei ping: %w", err)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("tei ping: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("tei ping: status %d", resp.StatusCode)
	}
	return nil
}
