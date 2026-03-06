package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultOpenAIURL = "https://api.openai.com"
const maxOpenAIErrorBodyBytes = 4096

type openAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []openAIEmbedData `json:"data"`
}

type openAIEmbedData struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type openAIErrorResponse struct {
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// OpenAIEmbedder calls an OpenAI-compatible /v1/embeddings API.
type OpenAIEmbedder struct {
	baseURL string
	model   string
	apiKey  string
	dims    int
	client  *http.Client
}

// NewOpenAIEmbedder constructs an OpenAI-compatible embedder.
// If baseURL is empty, https://api.openai.com is used.
func NewOpenAIEmbedder(baseURL, model, apiKey string, dims int) *OpenAIEmbedder {
	if baseURL == "" {
		baseURL = defaultOpenAIURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &OpenAIEmbedder{
		baseURL: baseURL,
		model:   model,
		apiKey:  apiKey,
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

func (e *OpenAIEmbedder) Model() string { return e.model }
func (e *OpenAIEmbedder) Dims() int     { return e.dims }

func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.doEmbed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("openai embed: empty response")
	}
	return vecs[0], nil
}

func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	vecs, err := e.doEmbed(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("openai embed batch: got %d embeddings for %d inputs", len(vecs), len(texts))
	}
	return vecs, nil
}

func (e *OpenAIEmbedder) Ping(ctx context.Context) error {
	_, err := e.Embed(ctx, "ping")
	if err != nil {
		return fmt.Errorf("openai ping: %w", err)
	}
	return nil
}

func (e *OpenAIEmbedder) doEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	if e.model == "" {
		return nil, fmt.Errorf("openai embed: model is required")
	}
	if e.dims <= 0 {
		return nil, fmt.Errorf("openai embed: dims must be > 0")
	}

	body, err := json.Marshal(openAIEmbedRequest{Model: e.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("openai embed: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai embed: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai embed: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		err := e.httpError(resp)
		if resp.StatusCode >= http.StatusBadRequest && resp.StatusCode < http.StatusInternalServerError && resp.StatusCode != http.StatusTooManyRequests {
			return nil, NewNonRetryableError(err)
		}
		return nil, err
	}

	var result openAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("openai embed: decode: %w", err)
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("openai embed: empty response")
	}
	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("openai embed: got %d embeddings for %d inputs", len(result.Data), len(texts))
	}

	vecs := make([][]float32, len(texts))
	seen := make([]bool, len(texts))
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(texts) {
			return nil, fmt.Errorf("openai embed: response index %d out of range for %d inputs", item.Index, len(texts))
		}
		if seen[item.Index] {
			return nil, fmt.Errorf("openai embed: duplicate response index %d", item.Index)
		}
		seen[item.Index] = true
		if len(item.Embedding) == 0 {
			return nil, fmt.Errorf("openai embed: empty embedding at index %d", item.Index)
		}
		if e.dims > 0 && len(item.Embedding) != e.dims {
			return nil, fmt.Errorf("openai embed: expected %d dims, got %d", e.dims, len(item.Embedding))
		}
		vecs[item.Index] = item.Embedding
	}
	for i := range vecs {
		if !seen[i] {
			return nil, fmt.Errorf("openai embed: missing embedding for response index %d", i)
		}
	}
	return vecs, nil
}

func (e *OpenAIEmbedder) httpError(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxOpenAIErrorBodyBytes+1))
	if err != nil {
		return fmt.Errorf("openai embed: status %d", resp.StatusCode)
	}
	truncated := len(body) > maxOpenAIErrorBodyBytes
	if truncated {
		body = body[:maxOpenAIErrorBodyBytes]
	}
	var apiErr openAIErrorResponse
	if err := json.Unmarshal(body, &apiErr); err == nil && apiErr.Error != nil && apiErr.Error.Message != "" {
		msg := apiErr.Error.Message
		if truncated {
			msg += "..."
		}
		return fmt.Errorf("openai embed: status %d: %s", resp.StatusCode, msg)
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		return fmt.Errorf("openai embed: status %d", resp.StatusCode)
	}
	if truncated {
		msg += "..."
	}
	return fmt.Errorf("openai embed: status %d: %s", resp.StatusCode, msg)
}
