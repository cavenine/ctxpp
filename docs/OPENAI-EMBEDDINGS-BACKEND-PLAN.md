# OpenAI-Compatible Embeddings Backend Plan

This document scopes and sequences issue `#4` (`feat: OpenAI-compatible embeddings backend`).

The goal is to add a new embedding backend that targets the OpenAI `POST /v1/embeddings` API so ctx++ can work with OpenAI, Azure OpenAI-style proxies, vLLM, LiteLLM, LocalAI, and any other OpenAI-compatible server.

## Goals

- Add an opt-in `openai` embedding backend behind `CTXPP_EMBED_BACKEND=openai`.
- Support both single-text and batched embedding calls.
- Keep the existing auto-detect flow unchanged for Ollama, TEI, Bedrock, and bundled fallback.
- Preserve ctx++'s current portability and retry behavior.
- Document configuration clearly enough that users can switch providers without code changes.

## Non-Goals

- Replacing the native Ollama backend.
- Changing default backend detection order.
- Adding provider-specific auth flows beyond bearer-token support.
- Solving every provider-specific URL shape in v1 of this backend.

---

## User-Facing Design

### Activation

- Backend name: `openai`
- Opt-in only via `CTXPP_EMBED_BACKEND=openai`

### Environment Variables

- `CTXPP_OPENAI_URL`
  - Default: `https://api.openai.com`
  - Base URL for any OpenAI-compatible server
- `CTXPP_OPENAI_MODEL`
  - Required when backend is forced
  - Example: `text-embedding-3-small`
- `CTXPP_OPENAI_API_KEY`
  - Optional
  - If set, send `Authorization: Bearer <token>`
  - If unset, send no auth header to support local unauthenticated servers
- `CTXPP_OPENAI_DIMS`
  - Required when backend is forced
  - Used for schema compatibility and response validation

### Request/Response Contract

Request:

```json
{
  "model": "text-embedding-3-small",
  "input": ["text1", "text2"]
}
```

Response:

```json
{
  "data": [
    {"index": 0, "embedding": [0.1, 0.2]},
    {"index": 1, "embedding": [0.3, 0.4]}
  ]
}
```

### Compatibility Notes

- Native OpenAI-compatible endpoints should work directly.
- Ollama can be used through this path when exposed via `/v1/embeddings`.
- Azure-specific deployment paths are not a v1 requirement; treat them as follow-up work if the generic base URL approach is insufficient.

---

## Implementation Plan

### 1) Add a new embedder

Add `internal/embed/openai.go` with:

- `type OpenAIEmbedder struct`
- `NewOpenAIEmbedder(baseURL, model, apiKey string, dims int) *OpenAIEmbedder`
- `Model() string`
- `Dims() int`
- `Embed(ctx, text)`
- `EmbedBatch(ctx, texts)`
- `Ping(ctx)`

Implementation should mirror current backend patterns:

- HTTP client setup should follow the tuned transport style already used by `OllamaEmbedder` and `TEIEmbedder`.
- Batch support should be first-class, not an afterthought, because the indexer benefits heavily from `BatchEmbedder`.
- Errors should always be wrapped with context using `fmt.Errorf("context: %w", err)`.

### 2) Define request/response types

Add internal JSON structs for:

- request body with `model` and `input`
- response body with `data[].index` and `data[].embedding`
- optional error body for clearer HTTP failures when providers return JSON errors

The implementation should:

- preserve response ordering by `index`
- reject empty `data`
- reject embeddings whose length does not match configured dims
- reject batch responses whose count does not match request size

### 3) Integrate with backend detection

Update `internal/embed/embed.go`:

- extend the env var documentation block to include `openai`
- read `CTXPP_OPENAI_URL`, `CTXPP_OPENAI_MODEL`, `CTXPP_OPENAI_API_KEY`, `CTXPP_OPENAI_DIMS`
- add `case "openai": ...` in `Detect`

Recommended behavior:

- if forced and required vars are missing or invalid, return bundled fallback with a warning-compatible failure path rather than panic
- wrap the embedder with `NewRetryingEmbedder(...)`
- keep `usingExternal=true` only when the backend is actually reachable/usable

### 4) Ping behavior

`Ping` should perform a tiny embedding request instead of calling a provider-specific models endpoint.

Reasoning:

- the actual embed path is what ctx++ depends on
- OpenAI-compatible servers vary in what auxiliary endpoints they implement
- a minimal real request is the best compatibility check

### 5) Update user messaging

Update command warnings and setup text in:

- `cmd/index.go`
- `cmd/mcp.go`
- `cmd/backfill.go`

These messages should mention OpenAI-compatible backends as a supported alternative, not only Ollama and Bedrock.

### 6) Update docs

Update `README.md` with:

- backend overview including `openai`
- env var table or examples
- one hosted example
- one local OpenAI-compatible example

Suggested examples:

- OpenAI hosted embeddings
- Ollama using `/v1/embeddings`
- a generic self-hosted proxy example

---

## Testing Plan

Add httptest-based unit coverage similar to existing embedder tests.

### New tests

Add tests for:

- single embed success
- batch embed success
- auth header present when API key is set
- auth header absent when API key is unset
- non-200 response handling
- invalid JSON response
- empty `data` response
- dims mismatch
- response count mismatch for batch requests
- `Detect()` with `CTXPP_EMBED_BACKEND=openai`
- invalid `CTXPP_OPENAI_DIMS` handling

### Validation commands

- `go test ./...`
- `go vet ./...`

If the change is clean and time permits:

- `go test -race ./...`

---

## Open Questions

### Should dims be required?

Recommendation: yes for v1.

- ctx++ stores vectors with known dimensions
- explicit dims avoid silent schema drift
- probing dims at startup adds provider-specific behavior and ambiguity

### Should this backend auto-detect?

Recommendation: no for now.

- auto-detecting arbitrary OpenAI-compatible providers is unreliable
- opt-in keeps current local-first behavior intact

### Should Azure OpenAI be handled now?

Recommendation: not unless it falls out naturally from the base URL design.

- Azure's path and query conventions differ enough that it may deserve a dedicated follow-up backend or URL-template support

---

## Suggested Delivery Order

1. Implement `OpenAIEmbedder` with single and batch support.
2. Wire `openai` into `Detect` and env parsing.
3. Add focused unit tests.
4. Update CLI warnings and README examples.
5. Validate with `go test ./...` and `go vet ./...`.

## Success Criteria

- A user can set `CTXPP_EMBED_BACKEND=openai` and successfully index/search using an OpenAI-compatible embeddings API.
- Batch embedding works through the same backend.
- Misconfiguration fails clearly.
- Existing Ollama, TEI, Bedrock, and bundled behavior remains unchanged.
