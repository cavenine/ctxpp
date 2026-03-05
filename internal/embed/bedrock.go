package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// ---- AWS Bedrock -----------------------------------------------------------

const defaultBedrockRegion = "us-east-1"
const defaultBedrockModel = "amazon.titan-embed-text-v2:0"
const defaultBedrockDims = 1024

// titanMaxInputChars is the maximum number of characters to send to Titan
// Embed v2. The model's hard limit is 8,192 tokens; at ~4 characters per
// token for English/code text, 28,000 characters yields ~7,000 tokens
// with comfortable headroom. Inputs exceeding this are truncated.
const titanMaxInputChars = 28_000

// titanEmbedRequest is the JSON body sent to Bedrock's InvokeModel endpoint
// for the Amazon Titan Text Embeddings V2 model.
type titanEmbedRequest struct {
	InputText  string `json:"inputText"`
	Dimensions int    `json:"dimensions,omitempty"`
	Normalize  bool   `json:"normalize"`
}

// titanEmbedResponse is the JSON response from the Titan Text Embeddings V2 model.
type titanEmbedResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

// BedrockClientAPI is the subset of bedrockruntime.Client methods used by
// BedrockEmbedder. This enables test doubles without mocking the full SDK.
type BedrockClientAPI interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

// BedrockEmbedder calls AWS Bedrock's InvokeModel API for text embeddings.
// It uses the Amazon Titan Text Embeddings V2 model by default, which produces
// 1024-dimensional vectors (matching bge-m3 used by the Ollama backend).
//
// Authentication uses the standard AWS credential chain: environment variables,
// ~/.aws/credentials, IAM roles, SSO sessions. No custom auth configuration
// is required.
//
// BedrockEmbedder intentionally does NOT implement BatchEmbedder. Bedrock's
// InvokeModel is a single-text API. Concurrency is handled upstream by the
// indexer's fan-out with a semaphore (cfg.EmbedConcurrency), which should be
// set to 20-50 for Bedrock to exploit AWS's horizontal scaling.
type BedrockEmbedder struct {
	api   BedrockClientAPI
	model string
	dims  int
}

// NewBedrockEmbedder constructs a BedrockEmbedder using the default AWS
// credential chain. If region is empty, us-east-1 is used. If model is
// empty, amazon.titan-embed-text-v2:0 is used. If dims is <= 0, 1024 is used.
func NewBedrockEmbedder(ctx context.Context, region, model string, dims int) (*BedrockEmbedder, error) {
	if region == "" {
		region = defaultBedrockRegion
	}
	if model == "" {
		model = defaultBedrockModel
	}
	if dims <= 0 {
		dims = defaultBedrockDims
	}

	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("bedrock embed: load aws config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(cfg)
	return &BedrockEmbedder{api: client, model: model, dims: dims}, nil
}

// NewBedrockEmbedderFromClient constructs a BedrockEmbedder using a
// pre-configured BedrockClientAPI. This is the primary constructor for
// testing — pass a mock/stub that implements BedrockClientAPI.
func NewBedrockEmbedderFromClient(client BedrockClientAPI, model string, dims int) *BedrockEmbedder {
	if model == "" {
		model = defaultBedrockModel
	}
	if dims <= 0 {
		dims = defaultBedrockDims
	}
	return &BedrockEmbedder{api: client, model: model, dims: dims}
}

func (e *BedrockEmbedder) Model() string { return e.model }
func (e *BedrockEmbedder) Dims() int     { return e.dims }

// Embed calls Bedrock's InvokeModel with the Titan Text Embeddings V2 request
// format and returns the embedding vector. Inputs exceeding titanMaxInputChars
// are truncated to stay within Titan's 8,192 token limit.
func (e *BedrockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if len(text) > titanMaxInputChars {
		text = text[:titanMaxInputChars]
	}

	body, err := json.Marshal(titanEmbedRequest{
		InputText:  text,
		Dimensions: e.dims,
		Normalize:  true,
	})
	if err != nil {
		return nil, fmt.Errorf("bedrock embed: marshal: %w", err)
	}

	out, err := e.api.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(e.model),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        body,
	})
	if err != nil {
		if isBedrockNonRetryable(err) {
			return nil, NewNonRetryableError(fmt.Errorf("bedrock embed: invoke: %w", err))
		}
		return nil, fmt.Errorf("bedrock embed: invoke: %w", err)
	}

	var resp titanEmbedResponse
	if err := json.Unmarshal(out.Body, &resp); err != nil {
		return nil, fmt.Errorf("bedrock embed: decode: %w", err)
	}

	if len(resp.Embedding) == 0 {
		return nil, fmt.Errorf("bedrock embed: empty embedding in response")
	}
	if len(resp.Embedding) != e.dims {
		return nil, fmt.Errorf("bedrock embed: expected %d dims, got %d", e.dims, len(resp.Embedding))
	}

	return resp.Embedding, nil
}

// Ping verifies that AWS credentials are valid and the model is accessible
// by performing a minimal embedding call.
func (e *BedrockEmbedder) Ping(ctx context.Context) error {
	_, err := e.Embed(ctx, "ping")
	if err != nil {
		return fmt.Errorf("bedrock ping: %w", err)
	}
	return nil
}

// isBedrockNonRetryable returns true if err is a Bedrock error that should not
// be retried. Non-retryable errors indicate a permanent caller-side problem
// (bad credentials, missing permissions, invalid input, wrong model ID) and
// burning through the retry budget would only add latency without any chance
// of success.
//
// Retryable errors (ThrottlingException, InternalServerException,
// ServiceUnavailableException, ModelNotReadyException, ModelTimeoutException)
// are NOT matched here and will be retried by RetryingEmbedder as normal.
func isBedrockNonRetryable(err error) bool {
	var accessDenied *brtypes.AccessDeniedException
	if errors.As(err, &accessDenied) {
		return true
	}
	var validation *brtypes.ValidationException
	if errors.As(err, &validation) {
		return true
	}
	var notFound *brtypes.ResourceNotFoundException
	if errors.As(err, &notFound) {
		return true
	}
	var modelErr *brtypes.ModelErrorException
	if errors.As(err, &modelErr) {
		return true
	}
	var quota *brtypes.ServiceQuotaExceededException
	return errors.As(err, &quota)
}
