package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// mockBedrockClient implements BedrockClientAPI for testing.
type mockBedrockClient struct {
	// handler is called for each InvokeModel call. Return the response body
	// bytes and an optional error.
	handler func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error)
}

func (m *mockBedrockClient) InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	body, err := m.handler(ctx, params)
	if err != nil {
		return nil, err
	}
	return &bedrockruntime.InvokeModelOutput{Body: body}, nil
}

// titanResponse is a helper to build a valid Titan response body.
func titanResponse(t *testing.T, embedding []float32, tokenCount int) []byte {
	t.Helper()
	resp := titanEmbedResponse{Embedding: embedding, InputTextTokenCount: tokenCount}
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestBedrockEmbedder_Model(t *testing.T) {
	e := NewBedrockEmbedderFromClient(&mockBedrockClient{}, "amazon.titan-embed-text-v2:0", 1024)
	if got := e.Model(); got != "amazon.titan-embed-text-v2:0" {
		t.Errorf("Model() = %q, want %q", got, "amazon.titan-embed-text-v2:0")
	}
}

func TestBedrockEmbedder_Dims(t *testing.T) {
	e := NewBedrockEmbedderFromClient(&mockBedrockClient{}, "", 1024)
	if got := e.Dims(); got != 1024 {
		t.Errorf("Dims() = %d, want 1024", got)
	}
}

func TestBedrockEmbedder_Defaults(t *testing.T) {
	e := NewBedrockEmbedderFromClient(&mockBedrockClient{}, "", 0)
	if got := e.Model(); got != defaultBedrockModel {
		t.Errorf("Model() = %q, want %q", got, defaultBedrockModel)
	}
	if got := e.Dims(); got != defaultBedrockDims {
		t.Errorf("Dims() = %d, want %d", got, defaultBedrockDims)
	}
}

func TestBedrockEmbedder_Embed_Success(t *testing.T) {
	wantVec := make([]float32, 1024)
	for i := range wantVec {
		wantVec[i] = float32(i) * 0.001
	}

	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			// Verify request body.
			var req titanEmbedRequest
			if err := json.Unmarshal(input.Body, &req); err != nil {
				return nil, fmt.Errorf("unmarshal request: %w", err)
			}
			if req.InputText != "hello world" {
				t.Errorf("InputText = %q, want %q", req.InputText, "hello world")
			}
			if req.Dimensions != 1024 {
				t.Errorf("Dimensions = %d, want 1024", req.Dimensions)
			}
			if !req.Normalize {
				t.Error("Normalize = false, want true")
			}
			if *input.ModelId != "amazon.titan-embed-text-v2:0" {
				t.Errorf("ModelId = %q, want %q", *input.ModelId, "amazon.titan-embed-text-v2:0")
			}
			return titanResponse(t, wantVec, 3), nil
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "amazon.titan-embed-text-v2:0", 1024)
	vec, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vec) != 1024 {
		t.Fatalf("Embed() returned %d dims, want 1024", len(vec))
	}
	if vec[0] != wantVec[0] || vec[1023] != wantVec[1023] {
		t.Errorf("Embed() vec mismatch: vec[0]=%v, vec[1023]=%v", vec[0], vec[1023])
	}
}

func TestBedrockEmbedder_Embed_InvokeError(t *testing.T) {
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			return nil, fmt.Errorf("ThrottlingException: rate exceeded")
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for invoke failure")
	}
}

func TestBedrockEmbedder_Embed_EmptyEmbedding(t *testing.T) {
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			return titanResponse(t, []float32{}, 0), nil
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for empty embedding")
	}
}

func TestBedrockEmbedder_Embed_DimensionMismatch(t *testing.T) {
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			// Return 512 dims when 1024 expected.
			vec := make([]float32, 512)
			return titanResponse(t, vec, 1), nil
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for dimension mismatch")
	}
}

func TestBedrockEmbedder_Embed_InvalidJSON(t *testing.T) {
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			return []byte("not json"), nil
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	_, err := e.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for invalid JSON response")
	}
}

func TestBedrockEmbedder_Embed_ContextCancelled(t *testing.T) {
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			return nil, ctx.Err()
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	_, err := e.Embed(ctx, "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error for cancelled context")
	}
}

func TestBedrockEmbedder_Ping_Success(t *testing.T) {
	vec := make([]float32, 1024)
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			return titanResponse(t, vec, 1), nil
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	if err := e.Ping(context.Background()); err != nil {
		t.Errorf("Ping() error = %v, want nil", err)
	}
}

func TestBedrockEmbedder_Ping_Failure(t *testing.T) {
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			return nil, fmt.Errorf("AccessDeniedException: not authorized")
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	if err := e.Ping(context.Background()); err == nil {
		t.Error("Ping() error = nil, want error for access denied")
	}
}

func TestBedrockEmbedder_DoesNotImplementBatchEmbedder(t *testing.T) {
	e := NewBedrockEmbedderFromClient(&mockBedrockClient{}, "", 0)
	if _, ok := interface{}(e).(BatchEmbedder); ok {
		t.Error("BedrockEmbedder implements BatchEmbedder, but should NOT (concurrency is handled by indexer fan-out)")
	}
}

func TestBedrockEmbedder_Embed_CustomDims(t *testing.T) {
	tests := []struct {
		name string
		dims int
	}{
		{name: "256 dims", dims: 256},
		{name: "512 dims", dims: 512},
		{name: "1024 dims", dims: 1024},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mock := &mockBedrockClient{
				handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
					var req titanEmbedRequest
					if err := json.Unmarshal(input.Body, &req); err != nil {
						return nil, err
					}
					if req.Dimensions != tc.dims {
						t.Errorf("Dimensions = %d, want %d", req.Dimensions, tc.dims)
					}
					vec := make([]float32, tc.dims)
					return titanResponse(t, vec, 1), nil
				},
			}

			e := NewBedrockEmbedderFromClient(mock, "", tc.dims)
			vec, err := e.Embed(context.Background(), "test")
			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			if len(vec) != tc.dims {
				t.Errorf("len(vec) = %d, want %d", len(vec), tc.dims)
			}
		})
	}
}

func TestBedrockEmbedder_RequestFormat(t *testing.T) {
	var capturedReq titanEmbedRequest
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			if err := json.Unmarshal(input.Body, &capturedReq); err != nil {
				return nil, err
			}
			// Verify HTTP-level fields.
			if *input.ContentType != "application/json" {
				t.Errorf("ContentType = %q, want %q", *input.ContentType, "application/json")
			}
			if *input.Accept != "application/json" {
				t.Errorf("Accept = %q, want %q", *input.Accept, "application/json")
			}

			vec := make([]float32, 1024)
			return titanResponse(t, vec, 5), nil
		},
	}

	e := NewBedrockEmbedderFromClient(mock, "amazon.titan-embed-text-v2:0", 1024)
	_, err := e.Embed(context.Background(), "test input text")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if capturedReq.InputText != "test input text" {
		t.Errorf("InputText = %q, want %q", capturedReq.InputText, "test input text")
	}
	if capturedReq.Dimensions != 1024 {
		t.Errorf("Dimensions = %d, want 1024", capturedReq.Dimensions)
	}
	if !capturedReq.Normalize {
		t.Error("Normalize = false, want true")
	}
}

func TestBedrockEmbedder_Embed_TruncatesLongInput(t *testing.T) {
	var capturedReq titanEmbedRequest
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			if err := json.Unmarshal(input.Body, &capturedReq); err != nil {
				return nil, err
			}
			vec := make([]float32, 1024)
			return titanResponse(t, vec, 1), nil
		},
	}

	// Create an input exceeding titanMaxInputChars (28,000 chars).
	longInput := make([]byte, 35000)
	for i := range longInput {
		longInput[i] = 'a'
	}

	e := NewBedrockEmbedderFromClient(mock, "", 1024)
	_, err := e.Embed(context.Background(), string(longInput))
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(capturedReq.InputText) != titanMaxInputChars {
		t.Errorf("InputText length = %d, want %d (truncated)", len(capturedReq.InputText), titanMaxInputChars)
	}
}

func TestIsBedrockNonRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "AccessDeniedException is non-retryable",
			err:  &brtypes.AccessDeniedException{},
			want: true,
		},
		{
			name: "ValidationException is non-retryable",
			err:  &brtypes.ValidationException{},
			want: true,
		},
		{
			name: "ResourceNotFoundException is non-retryable",
			err:  &brtypes.ResourceNotFoundException{},
			want: true,
		},
		{
			name: "ModelErrorException is non-retryable",
			err:  &brtypes.ModelErrorException{},
			want: true,
		},
		{
			name: "ServiceQuotaExceededException is non-retryable",
			err:  &brtypes.ServiceQuotaExceededException{},
			want: true,
		},
		{
			name: "ThrottlingException is retryable",
			err:  &brtypes.ThrottlingException{},
			want: false,
		},
		{
			name: "InternalServerException is retryable",
			err:  &brtypes.InternalServerException{},
			want: false,
		},
		{
			name: "ServiceUnavailableException is retryable",
			err:  &brtypes.ServiceUnavailableException{},
			want: false,
		},
		{
			name: "ModelNotReadyException is retryable",
			err:  &brtypes.ModelNotReadyException{},
			want: false,
		},
		{
			name: "ModelTimeoutException is retryable",
			err:  &brtypes.ModelTimeoutException{},
			want: false,
		},
		{
			name: "generic error is retryable",
			err:  fmt.Errorf("network timeout"),
			want: false,
		},
		{
			name: "wrapped AccessDeniedException is non-retryable",
			err:  fmt.Errorf("wrapped: %w", &brtypes.AccessDeniedException{}),
			want: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isBedrockNonRetryable(tc.err); got != tc.want {
				t.Errorf("isBedrockNonRetryable(%T) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestRetryingEmbedder_FastFailsOnNonRetryableBedrockError(t *testing.T) {
	calls := 0
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			calls++
			return nil, &brtypes.AccessDeniedException{}
		},
	}

	inner := NewBedrockEmbedderFromClient(mock, "", 1024)
	r := NewRetryingEmbedder(inner, RetryConfig{MaxRetries: 5, BaseBackoff: 1})

	_, err := r.Embed(context.Background(), "test")
	if err == nil {
		t.Fatal("Embed() error = nil, want error")
	}

	// Should have called InvokeModel exactly once — no retries for non-retryable errors.
	if calls != 1 {
		t.Errorf("InvokeModel called %d times, want 1 (no retries for AccessDeniedException)", calls)
	}

	// The error chain must still contain the NonRetryableError sentinel.
	var nre *NonRetryableError
	if !errors.As(err, &nre) {
		t.Errorf("err does not wrap NonRetryableError: %v", err)
	}
}

func TestRetryingEmbedder_RetriesOnThrottling(t *testing.T) {
	calls := 0
	wantVec := make([]float32, 1024)
	mock := &mockBedrockClient{
		handler: func(ctx context.Context, input *bedrockruntime.InvokeModelInput) ([]byte, error) {
			calls++
			if calls < 3 {
				return nil, &brtypes.ThrottlingException{}
			}
			return titanResponse(t, wantVec, 1), nil
		},
	}

	inner := NewBedrockEmbedderFromClient(mock, "", 1024)
	// Use 1ns base backoff so the test doesn't actually sleep.
	r := NewRetryingEmbedder(inner, RetryConfig{MaxRetries: 5, BaseBackoff: 1})

	vec, err := r.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if vec == nil {
		t.Fatal("Embed() returned nil vec, want non-nil")
	}
	if calls != 3 {
		t.Errorf("InvokeModel called %d times, want 3 (2 throttle failures then success)", calls)
	}
}
