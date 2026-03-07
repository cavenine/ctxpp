# ANN Vector Index Plan

This document scopes and sequences issue `#5` (`feat: ANN vector index`).

The goal is to keep semantic and hybrid search fast as ctx++ scales beyond the range where brute-force cosine scan is comfortably cheap.

## Goals

- Add an ANN-backed semantic retrieval path for large repositories.
- Preserve the existing `ctxpp_search` behavior and MCP tool surface.
- Keep brute-force cosine search as the correctness fallback.
- Preserve current ranking behavior: source-tier weighting, deduplication, and hybrid RRF should continue to work.
- Support incremental indexing and live-watch updates without requiring full reindex for normal edits.

## Non-Goals

- Replacing SQLite as the system of record.
- Introducing a required external vector database or service.
- Removing exact cosine search.
- Changing embedding backends or embedding storage format.
- Shipping a learned reranker in the same project.

---

## Why Now

Today semantic search performs a brute-force scan over all stored vectors in `SearchSemantic`. This is simple and correct, but it becomes one of the dominant search costs on very large repositories.

ANN should be treated as a scaling feature:

- under ~100k symbols, brute-force is often good enough
- around ~300k symbols, semantic scan cost is already material
- beyond that, ctx++ needs a faster candidate-generation path to protect search latency and hosted-agent usability

## Design Principles

### 1) Refactor first, optimize second

The current `SearchSemantic` function combines API surface, candidate generation, exact scoring, tier weighting, and symbol hydration. ANN should first split these responsibilities so brute-force becomes one implementation behind a stable search contract.

### 2) ANN is candidate generation, not final truth

The first ANN implementation should return a candidate set that is then rescored with exact cosine similarity using the stored vectors already persisted in SQLite.

This keeps result quality close to current behavior while still capturing most of the latency win.

### 3) Keep SQLite as source of truth

Embeddings should continue to be written to SQLite first. The ANN index should be a derived structure that can be rebuilt from SQLite data if needed.

### 4) Prefer in-process Go implementation

Do not start with a SQLite-native ANN extension or external vector store. A Go-side ANN index keeps startup, fallback, rebuild, and incremental-update behavior under ctx++'s control.

### 5) Pick the smallest viable ANN dependency first

The initial ANN spike should target an in-process HNSW-style Go library, with exact reranking still handled by ctx++ using the vectors persisted in SQLite.

---

## User-Facing Behavior

### Search

No MCP API changes are required for the first iteration.

- `ctxpp_search mode=semantic` should continue to return semantic nearest neighbors.
- `ctxpp_search mode=hybrid` should continue to merge semantic and keyword results via RRF.
- If the ANN index is unavailable, stale, or disabled, ctx++ should transparently fall back to brute-force search.

### Configuration

Recommended initial env vars:

- `CTXPP_VECTOR_INDEX`
  - Default: `auto`
  - Values: `auto`, `bruteforce`, `ann`
- `CTXPP_ANN_EF_SEARCH`
  - Optional ANN search tuning value
- `CTXPP_ANN_CANDIDATES`
  - Number of ANN candidates to exact-rerank before final ranking

`auto` should prefer ANN only when the index is available and healthy.

---

## Architecture Plan

### 1) Introduce a semantic retrieval abstraction

Refactor the store so the public search path delegates to an internal engine.

Conceptually:

```go
type SemanticSearcher interface {
    Search(ctx context.Context, queryVec []float32, limit int) ([]Candidate, error)
}
```

Where `Candidate` should carry enough information to preserve current ranking behavior:

- symbol ID or embedding row ID
- raw similarity score
- optional fast-path metadata needed for reranking

Initial implementations:

- `BruteForceSearcher`
- `ANNSearcher`

`SearchSemantic` should become an orchestration method that:

1. asks the configured searcher for candidates
2. applies exact scoring if needed
3. applies source-tier weighting
4. hydrates symbols
5. deduplicates by name/kind

### 2) Build a persisted derived ANN index

The ANN index should live under `.ctxpp/` as a sibling artifact to `index.db`.

Recommended first shape:

- `.ctxpp/index.db` remains the source of truth for symbols and vectors
- `.ctxpp/ann-hnsw.bin` stores the derived HNSW graph
- `.ctxpp/ann-hnsw.json` stores ANN metadata

If the ANN files are missing or incompatible with the current embeddings model/dims, ctx++ should rebuild them from SQLite.

### 3) Keep exact reranking in SQLite-backed data

The ANN path should retrieve more than `limit` candidates, then compute exact cosine similarity using the stored vectors.

Suggested flow:

1. ANN returns top `N` row IDs or symbol IDs
2. ctx++ fetches the corresponding vectors from SQLite
3. ctx++ computes exact cosine scores
4. ctx++ applies tier weights
5. ctx++ returns the final `limit`

This keeps ANN approximate only in recall, not in final score ordering.

### 4) Support incremental maintenance

Index lifecycle requirements:

- initial index build should add vectors to ANN as embeddings are written
- file updates should remove stale vectors and insert replacement vectors
- file deletes should remove corresponding vectors from ANN
- backend/model change should invalidate and rebuild the ANN index

If the chosen ANN library makes deletes expensive, the first version can support tombstones plus periodic rebuild/compaction.

---

## Persistence and Lifecycle

### Startup

At startup:

1. open SQLite store
2. inspect embedding metadata (model, dims, count)
3. inspect ANN artifact metadata
4. if compatible, open ANN index
5. if missing or incompatible, either rebuild eagerly or mark for background rebuild

Recommendation: start with eager rebuild only when `CTXPP_VECTOR_INDEX=ann`; otherwise degrade to brute-force and log a warning.

### Rebuild Triggers

Rebuild the ANN index when:

- embedding dimensions change
- embedding model changes
- ANN format version changes
- artifact corruption is detected

### Failure Behavior

ANN failures must degrade safely:

- search falls back to brute-force
- indexing continues writing embeddings to SQLite
- rebuild can be retried later

---

## Candidate Library / Data Structure Choice

The intended algorithm is HNSW-style ANN or a comparable in-process structure with:

- fast approximate nearest-neighbor search
- incremental insert support
- acceptable delete/update strategy
- Go integration that does not require a separate service

Selection criteria:

- search latency improvement on 300k+ symbols
- acceptable memory overhead
- rebuild time
- incremental update complexity
- portability and maintenance burden

### Recommendation

Use a pure-Go HNSW implementation for the first ANN spike.

Current recommendation: `github.com/coder/hnsw`

Why this is the best fit for ctx++ right now:

- pure Go, so it preserves the local CLI / `go install` deployment model better than CGo-heavy alternatives
- in-process and file-backed, which fits ctx++'s derived-artifact architecture
- natural fit for candidate generation followed by exact rerank from SQLite-backed vectors
- lower operational burden than introducing a second embedded database or a native SQLite vector extension

### Alternatives Considered

#### DuckDB VSS

Not recommended for the first implementation.

Reasons:

- it introduces a second embedded database or requires a much larger persistence migration away from SQLite
- the VSS extension is still experimental for persistence/recovery workflows
- it adds CGo/runtime packaging burden and makes symbol + FTS joins awkward if SQLite remains the source of truth

#### CGo-backed ANN libraries (for example, USEARCH)

Not recommended for the first implementation.

Reasons:

- stronger native dependency and packaging burden
- less aligned with ctx++'s current operational model
- still useful later if pure-Go HNSW proves insufficient

#### SIMD-accelerated brute-force scan

Worth benchmarking separately, but it is not a replacement for ANN.

It may still be a useful optimization if brute-force remains acceptable below the repo sizes where ANN becomes necessary.

---

## Refactor Impact

### `internal/store/store.go`

Primary refactor target.

- split current `SearchSemantic` into engine + orchestration pieces
- preserve symbol hydration and tier weighting logic
- update `SearchHybrid` to use the new semantic path without changing its external behavior

### `internal/indexer/indexer.go`

Needs ANN lifecycle hooks.

- update ANN state when new embeddings are written
- handle single-file watch path updates
- support rebuild after embedding invalidation

### `cmd/app.go` / app wiring

May need a new dependency for semantic search engine initialization if the ANN index is not owned directly by `Store`.

### `cmd/handlers.go`

Should remain nearly unchanged.

The goal is to keep `ctxpp_search` stable while changing internals only.

### Tests and Benchmarks

Significant additions expected in:

- `internal/store/store_test.go`
- `bench/`

Add contract tests that both brute-force and ANN implementations must satisfy.

---

## Delivery Plan

### Phase 1: Search-layer refactor

- introduce semantic search abstraction
- keep brute-force as the only implementation
- add shared contract tests

Success criteria:

- no user-visible behavior change
- existing tests pass

### Phase 2: Offline ANN prototype

- load embeddings from SQLite into ANN structure
- support ANN candidate generation at query time
- exact-rerank candidates using stored vectors
- benchmark quality and latency against brute-force

Success criteria:

- substantial semantic latency improvement on large corpora
- no material regression in top-k relevance on benchmark queries

### Phase 3: Persistence and startup lifecycle

- persist ANN artifact under `.ctxpp/`
- add compatibility metadata and rebuild logic
- support fallback when ANN artifact is unavailable

Success criteria:

- ANN survives process restart
- rebuild behavior is deterministic and safe

### Phase 4: Incremental update support

- apply inserts/updates/deletes during indexing and watch events
- add periodic compaction or rebuild strategy if needed

Success criteria:

- branch switches and normal edits remain self-healing
- ANN results remain consistent with SQLite embeddings

### Current implementation note

The current ANN integration now supports deferred artifact refresh during write-heavy flows.

- `Store.BeginDeferredANNSync()` starts a batching window
- `Store.EndDeferredANNSync()` flushes one ANN refresh when the outermost batching window closes
- index and backfill flows use this to avoid rebuilding HNSW artifacts after every embedding batch

This keeps correctness the same while reducing rebuild churn in hot paths.

The current prototype also supports background rebuild-and-swap for active ANN stores:

- mutating operations can schedule a background ANN rebuild instead of blocking on a full artifact refresh
- ctx++ keeps serving from the last known-good ANN graph while rebuild is in flight
- on success, the store swaps to the fresh ANN searcher atomically
- on failure, ctx++ keeps the old ANN graph and can retry later

---

## Testing Plan

### Correctness

Add tests for:

- semantic search contract shared by brute-force and ANN engines
- fallback to brute-force when ANN is missing or unhealthy
- exact-rerank preserving stable ordering on candidate sets
- rebuild on model/dims mismatch
- incremental insert, update, and delete behavior

### Benchmarks

Measure:

- brute-force vs ANN semantic latency at 10k, 100k, and 300k+ symbols
- hybrid search latency impact
- ANN memory overhead
- ANN rebuild time
- ANN recall before exact rerank

Initial local snapshot on a Linux workstation (`-benchtime=1x`, 768 dims):

- brute-force semantic search, 10k symbols: ~26.9 ms
- HNSW semantic search, 10k symbols: ~0.94 ms
- HNSW artifact build, 10k symbols: ~611 ms
- brute-force semantic search, 100k symbols: ~252.6 ms
- HNSW semantic search, 100k symbols: ~0.98 ms
- HNSW artifact build, 100k symbols: ~8.0 s
- batched embedding writes with deferred ANN sync, 10k symbols / 64-item batches: ~0.81 s
- batched embedding writes with immediate ANN sync, 10k symbols / 64-item batches: ~104 s

Interpretation:

- query latency improvement is large and stays nearly flat between 10k and 100k
- build cost is now the primary performance problem to manage
- batching/deferred refresh is necessary before wiring ANN deeply into all indexing paths
- immediate per-batch HNSW maintenance is not viable for large write-heavy flows with the current library/integration
- deferred refresh remains necessary; it is over 100x faster than immediate sync on the current 10k batched-write benchmark

### Validation Commands

- `go test ./...`
- `go vet ./...`
- `go test -race ./...`

And benchmark runs under `bench/` using the kubernetes corpus or a similarly large repository.

---

## Open Questions

### Should ANN be default in `auto` mode?

Recommendation: not initially.

Start with ANN as opt-in or auto-only when a valid ANN artifact already exists.

### Should ANN be exact-rerank only, or fully trusted for final order?

Recommendation: exact-rerank.

This protects current quality expectations and keeps brute-force as the oracle.

### Should ANN index live inside SQLite?

Recommendation: no for v1.

Use SQLite for truth, separate artifact for derived search structure.

### Do we need ANN for all corpora?

Recommendation: no.

Small repositories should continue to work well with brute-force search.

---

## Success Criteria

- ctx++ keeps semantic and hybrid search responsive on very large repositories.
- `ctxpp_search` API and user experience remain stable.
- ANN can be disabled or bypassed without breaking search.
- Search quality remains close to current brute-force behavior.
- Live indexing and branch-switch recovery continue to work.
