# ctx++ Architecture

ctx++ is an MCP server that gives AI coding agents deep understanding of a
codebase. It reads source files, breaks them down into meaningful pieces
(functions, types, methods), converts those pieces into searchable forms, and
serves results back to the agent over the MCP protocol. Think of it as a
search engine purpose-built for code, running entirely on your machine.

This document explains how it works, what decisions were made, and why.

---

## The Problem

When an AI agent is asked "where does pod scheduling happen?" in a large
codebase, it needs to find the right functions, types, and interfaces -- not
just files with the word "scheduling" in them. Naive text search returns too
much noise. File-level embeddings are too coarse. The agent needs
symbol-level precision: the actual function signatures, type definitions,
and API boundaries that matter.

ctx++ solves this by extracting individual symbols from source code, embedding
each one as a vector, and combining vector similarity with keyword matching
to return precise, ranked results.

---

## High-Level Flow

A request like "find code related to RBAC authorization" goes through these
stages:

```
Agent asks a question
        |
        v
  MCP Server receives the query (stdio)
        |
        v
  Query is processed two ways simultaneously:
        |                           |
        v                           v
  FTS5 keyword search       Semantic vector search
  (BM25 ranking on            (cosine similarity
   symbol names,                against all stored
   signatures, docs)            embedding vectors)
        |                           |
        v                           v
  Results merged via Reciprocal Rank Fusion (RRF)
        |
        v
  Call-graph re-ranking (boost connected symbols)
        |
        v
  Top results returned to agent
```

Before any of this can happen, the codebase must be indexed. Indexing is a
separate pipeline that walks every file, parses it, extracts symbols, and
generates embeddings.

---

## Repository Layout

```
cmd/ctxpp/              CLI entry point (MCP server + index command)
internal/
  types/                Shared data structures (Symbol, CallEdge, etc.)
  store/                SQLite persistence (FTS5, vectors, hybrid search)
  parser/               Language parsers (tree-sitter based)
  embed/                Embedding backends (Ollama, TEI, Bedrock, bundled stub)
  indexer/              File walking, parsing orchestration, live watching
bench/
  compare/              Performance benchmark harness
  quality_eval/         Search quality evaluation tool
```

Each `internal/` package has a focused responsibility and communicates with
others through the types defined in `internal/types/`. There are no circular
imports.

---

## Indexing: How Code Gets Into the Database

Indexing transforms source files into searchable symbols and vectors. It runs
as a five-stage pipeline, where each stage feeds the next through Go channels:

```
Stage 1: Walk       Stage 2: Parse      Stage 3: Store      Stage 4: Embed      Stage 5: Write
(1 goroutine)       (N workers)         (1 goroutine)       (M workers)         (1 goroutine)
                                                                                 
Walk the file    -> Read + hash +    -> Write symbols,   -> Call Ollama      -> Batch-write
tree, filter        parse with          edges to SQLite     to get vectors      vectors to
by extension        tree-sitter         in batches          for each symbol     SQLite
                                                                                 
    walkJobs ------>  parsed -------->  embedJobs -------->  embedResults --->
    (channel)         (channel)         (channel)            (channel)
```

### Why a pipeline?

Each stage has different performance characteristics. Parsing is CPU-bound.
Embedding is I/O-bound (HTTP calls to Ollama for GPU inference). SQLite
writes are single-threaded by design. The pipeline lets all three run
concurrently: while batch N is being embedded, batch N+1 is being parsed,
and batch N-1 is being written to disk.

### Stage 1: Walk

A single goroutine walks the project directory tree. It skips hidden
directories, `node_modules`, the `.ctxpp` index directory, and anything
matched by `.gitignore`. Files are routed to parsers by extension (`.go`,
`.proto`, `.sql`, etc.) or by exact filename (`Makefile`, `Dockerfile`).

### Stage 2: Parse (N workers)

N concurrent workers (defaulting to the number of CPU cores) each:

1. Read the file contents.
2. Compute a content hash (xxhash) and compare it against the stored hash.
   If it matches, the file is skipped -- this is how incremental indexing
   works. (The database column is named `sha256` for historical reasons,
   but the actual algorithm is xxhash for speed.)
3. Parse the file using tree-sitter to extract symbols.
4. Classify the file into a source tier (explained below).

Parsing produces a `Result` containing symbols, call edges (which function
calls which), and import edges (which file imports which package).

### Stage 3: Store (single writer)

A single goroutine receives parsed results and writes them to SQLite in
batches of 32 files per transaction. This matters because SQLite's WAL
(Write-Ahead Log) has per-commit overhead; batching amortizes it. Previously,
each file triggered ~5 separate commits. Now it's 1 commit per batch.

After writing, this stage prepares embedding jobs. It builds "enriched" text
for each symbol by combining:
- The file path (for domain signal -- `pkg/auth/handler.go` tells the model
  this is authentication code)
- The symbol kind and name
- The function signature
- The doc comment
- A truncated body snippet (first ~500 bytes)
- Names of functions this symbol calls (up to 10)
- Packages this file imports (up to 6)

This enriched text is what gets sent to the embedding model.

### Stage 4: Embed (M workers)

M concurrent workers (default 8) send enriched text to the embedding backend
(typically Ollama running locally with GPU acceleration). The key optimization
here is **batch embedding**: instead of one HTTP request per symbol, symbols
are collected into batches of up to 2,000 and sent in a single `/api/embed`
call. This reduces HTTP round-trips from ~180k to ~90 for a large codebase
and enables GPU batching inside Ollama.

Two concurrent batches are kept in flight simultaneously so that batch N+1 is
already being processed on the GPU while batch N's results are being written
to disk.

### Stage 5: Write Embeddings (single writer)

A single goroutine receives embedding vectors and batch-upserts them into
SQLite (64 vectors per transaction). The vectors are stored as little-endian
float32 BLOBs.

### Incremental Indexing

Files are tracked by their content hash (xxhash). On subsequent runs, only
files whose hash has changed are re-parsed and re-embedded. Unchanged files
are skipped entirely. This makes re-indexing after small edits near-instant.

### Live Watching

After the initial index, an fsnotify-based file watcher monitors the project
for changes. When a file is modified, it is debounced (default 500ms) and
then re-indexed through the single-file path (parse, store, embed). Deleted
files are removed from the database via CASCADE deletes.

---

## Parsing: Extracting Symbols from Source Code

Parsers implement a small interface:

```go
type Parser interface {
    Language() string
    Extensions() []string
    Parse(filePath string, src []byte) (Result, error)
}
```

A `Result` contains:
- **Symbols**: functions, methods, types, interfaces, constants, variables,
  etc. Each has an ID, file path, name, kind, signature, doc comment, line
  range, and optional receiver (for methods).
- **Call edges**: which symbol calls which other symbol, and on which line.
- **Import edges**: which file imports which package.

### Tree-sitter first

All language parsers that can use tree-sitter do. Tree-sitter provides a
real AST (abstract syntax tree), which means the parser can distinguish
a function declaration from a function call, extract receiver types for
methods, and capture doc comments accurately. Hand-rolled regex parsers
are error-prone and fragile; tree-sitter grammars handle edge cases
(multiline signatures, nested types, string literals containing keywords)
that regex cannot.

Custom regex parsers are only used as a fallback for file types without
tree-sitter grammars (like `.http` request files).

### Supported languages

| Parser | Language | Method | Extracts |
|--------|----------|--------|----------|
| Go | `.go` | tree-sitter | functions, methods, types, interfaces, consts, vars, call graph, imports |
| Protobuf | `.proto` | tree-sitter | services, RPCs, messages, enums |
| SQL | `.sql` | tree-sitter + regex fallback | tables, views, indexes, triggers, procedures |
| Markdown | `.md` | tree-sitter | headings as sections |
| HTML | `.html` | tree-sitter | elements |
| Shell | `.sh`, `.bash` | tree-sitter | commands |
| HTTP | `.http`, `.rest` | regex | request definitions |
| Text | `.txt`, `.cfg`, etc. | line scanner | document chunks |

### Why symbol-level, not file-level or chunk-level?

This is the most consequential design decision in ctx++. Competing tools
embed at different granularities:

- **File-level** (e.g., some MCP tools): one embedding per file. Fast to
  index, but a 2,000-line file about authentication, database access, and
  logging gets a single blurry vector that's vaguely about all three topics.
  Queries for "authentication" match weakly against the whole file.

- **Chunk-level** (e.g., codemogger): split files into fixed-size text
  chunks (~500-1000 tokens). Better than file-level, but chunk boundaries
  don't respect symbol boundaries. A function might span two chunks, or a
  chunk might contain the tail of one function and the head of another.

- **Symbol-level** (ctx++): one embedding per function, type, method, etc.
  Each vector represents exactly one semantic unit. The `HandleLogin`
  function gets its own vector that captures authentication logic. The
  `MigrateDatabase` function gets a separate vector about schema migration.
  Queries match against precisely the right abstraction level.

The trade-off is more vectors (a large codebase can have 300k+ symbols) and
more embedding work. We accept this because:
1. GPU-accelerated batch embedding makes the indexing cost manageable.
2. Brute-force cosine scan over 300k vectors takes ~600ms, which is
   acceptable for an interactive tool.
3. The quality improvement is significant -- benchmarks show symbol-level
   consistently outperforms chunk-level and file-level.

---

## Storage: SQLite as the Single Store

Everything lives in a single SQLite database at `.ctxpp/index.db`. SQLite
was chosen because:

- **Zero deployment**: no database server to install or configure.
- **Single-file portable**: the entire index is one file that can be copied
  or deleted freely.
- **WAL mode**: allows concurrent readers while one writer is active, which
  the pipeline exploits.
- **FTS5**: SQLite's built-in full-text search engine provides BM25 keyword
  ranking without external dependencies.

### Schema

```
files           Tracks indexed files (path, content hash, mod time, language)
symbols         Every extracted symbol (ID, file, name, kind, signature, etc.)
embeddings      One float32 vector BLOB per symbol
call_edges      Caller -> callee relationships with line numbers
import_edges    File -> imported package relationships
symbols_fts     FTS5 virtual table over symbol name, signature, doc comment
```

The FTS5 table is kept in sync with `symbols` via SQLite triggers. When a
symbol is inserted, updated, or deleted, the corresponding FTS entry is
automatically maintained.

### Dual connection pools

The store maintains two SQLite connection pools:

- **Write pool** (1 connection): handles all mutations. SQLite only supports
  one writer at a time, so this is capped at 1. Uses `_txlock=immediate` to
  acquire the write lock at transaction start rather than on first write.

- **Read pool** (4 connections): handles concurrent read-only queries. This
  is critical during indexing: while the single writer is flushing a batch of
  parsed files, parse workers need to check file hashes to decide whether to
  skip unchanged files. Without a separate read pool, those hash lookups
  would block on the write transaction.

### Vector storage and search

Embedding vectors are stored as little-endian float32 BLOBs in the
`embeddings` table. Search is brute-force: scan every row, compute cosine
similarity against the query vector, keep the top K.

This sounds expensive, and it is -- but it's adequate. For a codebase with
300k symbols, the scan takes ~600ms. For codebases under 100k symbols
(which covers most projects), it's well under 100ms. We chose brute-force
over approximate nearest neighbor (ANN) indexing because:

1. **Simplicity**: no native extensions, no index maintenance, no
   hyperparameter tuning.
2. **Correctness**: brute-force always returns the true top K. ANN methods
   trade accuracy for speed.
3. **Good enough**: the bottleneck for large codebases is actually the
   Ollama embedding call (~25ms per query), not the vector scan.

If a future version needs sub-100ms search on million-symbol corpora, the
path forward is HNSW or a similar ANN index. The interface is designed to
make this swap transparent.

### Optimizations in the vector scan

The scan avoids allocations on the hot path:

- **sync.Pool** for scratch buffers (top-K heap, argument slices, maps).
- **sql.RawBytes** for vector BLOBs to avoid copying each row's bytes.
- **Integer rowid** as the primary scan key (avoids string allocation for
  symbol IDs until we need them).
- **Two-phase scan**: Phase 1 scans all vectors and keeps top 3x candidates
  by rowid. Phase 2 batch-fetches symbol metadata and applies tier weights
  only for the candidates, then re-sorts and truncates to the final limit.

---

## Search: Finding the Right Symbols

ctx++ supports three search modes:

### Keyword search (FTS5 BM25)

Traditional full-text search. The query is matched against symbol names,
signatures, and doc comments using SQLite's FTS5 engine with BM25 ranking.
Fast (sub-millisecond) and precise for exact identifier lookups.

### Semantic search (vector similarity)

The query is embedded into a vector (by sending it to the same model used
during indexing), then compared against all stored vectors via cosine
similarity. This finds conceptually related symbols even when the query
uses different words than the code -- "authentication flow" can match a
function called `HandleLogin`.

**Query-vector caching**: because the MCP server is a long-lived process
and agents frequently repeat identical queries within a session, the
`CachingEmbedder` wrapper caches query vectors in memory. The cache holds
up to 512 entries (≈2 MB at 1024 dims) and evicts the oldest entry (FIFO)
when full. Cache hits skip the Ollama round-trip entirely. The cache is
per-process and not persisted; it is distinct from the symbol-embedding
cache in SQLite (which is keyed on source hash and persists across restarts).

### Hybrid search (default)

Combines both approaches using **Reciprocal Rank Fusion (RRF)**, a
well-established technique from information retrieval. Here's how it works:

1. Run keyword search and semantic search concurrently, each returning
   3x the requested result count (over-selection).
2. For each result, compute an RRF score:
   ```
   score = 0.6 / (60 + semantic_rank) + 0.4 / (60 + keyword_rank)
   ```
   If a symbol appears in only one list, it receives a penalty rank for the
   other.
3. Sort by RRF score and take the top K.

**Why RRF instead of score normalization?** Raw scores from different
retrieval systems aren't comparable -- cosine similarity ranges and BM25
scores have different distributions. RRF sidesteps this by working with
ranks (ordinal positions) rather than scores. It's simple, parameter-light
(just the smoothing constant k=60 from the original paper), and empirically
effective.

**Why 60/40 weighting toward semantic?** Semantic search handles natural
language queries ("where does pod scheduling happen?") better than keyword
search. But keyword search catches exact identifier matches that semantic
search may miss (especially for short, specific names like `Watch` or
`Admit`). The 60/40 split lets semantic lead while keyword provides a
safety net.

### FTS query preprocessing

Before the keyword query hits FTS5, it passes through a preprocessing step
that significantly improves BM25 quality for natural language queries:

1. Lowercase the input.
2. Tokenize into alphanumeric tokens (including hyphenated/dotted compounds
   like `pod-affinity` or `net.http`).
3. Remove English stopwords ("the", "and", "is", etc.) and common agent
   filler words ("help", "file", "code").
4. Deduplicate.
5. Filter by length (3-30 characters).
6. Cap at 12 terms.

This matters because FTS5 treats every word in the query as a search term.
A query like "etcd storage and watch mechanism" without preprocessing matches
the word "and" in almost every doc comment, diluting the ranking of the
actually relevant terms "etcd", "storage", and "watch".

### Call-graph re-ranking

After hybrid search returns results, an optional re-ranking pass promotes
symbols that are connected to each other via call edges. If the top 10
results include `provisionClaimOperation`, `setClaimProvisioner`, and
`bindVolumeToClaim`, and these functions call each other, they're likely
part of a cohesive feature and should rank higher together.

The boost is small and bounded: +2 positions per call-graph connection.
This prevents a single densely-connected utility function from dominating
results while still rewarding cohesive clusters.

---

## Source Tier Classification

Not all code is equally useful to an agent. A CHANGELOG file matching
the query "authentication" is noise. Generated protobuf stubs matching
"service" are low-signal. Vendor code matching "HTTP handler" is
relevant but less so than the project's own handlers.

ctx++ classifies every file into one of four tiers at index time:

| Tier | Weight | What goes here |
|------|--------|----------------|
| Code (1) | 1.0 | Project source code (the default) |
| Docs (2) | 0.85 | Tests, documentation, configs, markdown, YAML |
| Vendor (3) | 0.7 | `vendor/`, `node_modules/` |
| Low Signal (4) | 0.5 | CHANGELOGs, generated code (`*.pb.go`, `zz_generated*`) |

The tier weight multiplies the cosine similarity score during semantic
search. A vendor symbol with 0.95 cosine similarity effectively scores
0.95 * 0.7 = 0.665, while a project source symbol with 0.90 similarity
scores 0.90 * 1.0 = 0.90 and ranks higher.

### Decision: what counts as vendor?

Only `vendor/` and `node_modules/` -- directories unambiguously managed by
package managers.

The lesson: be conservative about what you penalize. False negatives
(missing a relevant result) are worse than false positives (including a
less relevant one).

---

## Embedding: Turning Code into Vectors

### What is an embedding?

An embedding model reads a piece of text and produces a fixed-length array
of floating-point numbers (a "vector") that captures its meaning. Texts with
similar meaning produce vectors that are close together in this
high-dimensional space. "cosine similarity" measures how close two vectors
are, on a scale from -1 (opposite) to 1 (identical).

### Backend detection

ctx++ auto-detects the embedding backend at startup:

1. Probe for a TEI (Text Embeddings Inference) server at `localhost:8080`.
2. If not found, probe for Ollama at `localhost:11434`.
3. If neither is available, fall back to a bundled stub that returns zero
   vectors (keyword search still works; semantic search is disabled).

The backend can also be explicitly selected via the `CTXPP_EMBED_BACKEND`
environment variable. Supported values: `auto` (default), `ollama`, `tei`,
`bedrock`, or `bundled`.

### Backend: Ollama (local GPU)

Ollama runs embedding models locally on your GPU. This matters for two
reasons:

1. **Privacy**: your code never leaves your machine.
2. **Speed**: GPU-accelerated embedding is 10-100x faster than CPU. Indexing
   Kubernetes (300k+ symbols) takes ~47 minutes with GPU vs. what would be
   many hours on CPU.

The default model is `bge-m3`, which produces 1024-dimensional vectors and
has strong multilingual and code understanding capabilities.

Symbols are collected into batches of up to 2,000 and sent in a single
`/api/embed` HTTP call, which reduces HTTP round-trips and enables GPU
batching inside Ollama. Two concurrent batches are kept in flight to hide
inter-batch latency.

Configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `CTXPP_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `CTXPP_OLLAMA_MODEL` | `bge-m3` | Embedding model name |
| `CTXPP_OLLAMA_SOCKET` | (none) | Unix domain socket path (bypasses TCP) |

### Backend: AWS Bedrock

AWS Bedrock provides embedding models as a managed service -- no local GPU
required. ctx++ uses Amazon Titan Text Embeddings V2, which produces
1024-dimensional vectors (matching bge-m3 used by the Ollama backend).

Bedrock's `InvokeModel` API is a single-text endpoint, so ctx++ uses
concurrent fan-out instead of batching: the indexer dispatches up to
`CTXPP_EMBED_CONCURRENCY` parallel `Embed()` calls, each making an
independent HTTP request to Bedrock. AWS routes these across its GPU fleet,
providing horizontal scaling that local Ollama cannot match.

**Retry behavior**: Bedrock uses a tuned retry configuration with 5 retries
(vs. 3 for local backends) and 500ms base backoff (vs. 100ms), because AWS
throttle windows are longer than local transient errors. Exponential backoff
with jitter handles `ThrottlingException` (HTTP 429) and
`ServiceUnavailableException` (503) gracefully.

**Authentication**: uses the standard AWS credential chain (environment
variables, `~/.aws/credentials`, IAM roles, SSO sessions). No custom auth
configuration required.

Configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `CTXPP_EMBED_BACKEND` | (must be `bedrock`) | Bedrock is explicit opt-in only |
| `CTXPP_BEDROCK_REGION` | `us-east-1` | AWS region |
| `CTXPP_BEDROCK_MODEL` | `amazon.titan-embed-text-v2:0` | Bedrock model ID |
| `CTXPP_BEDROCK_DIMS` | `1024` | Embedding dimensions (256, 512, or 1024) |
| `CTXPP_EMBED_CONCURRENCY` | `8` | Concurrent embed calls (set to 20-50 for Bedrock) |

**Performance considerations**: at 30-50 concurrent requests, Bedrock can
index Kubernetes in roughly 10-18 minutes (vs. 47 minutes with local Ollama
on a Ryzen 9 9950X). The trade-off is per-token cost (~$0.30-0.60 for a full
Kubernetes index) and network latency (~100ms per call vs. ~1-5ms local).
For incremental re-indexing (only changed symbols), both backends are
near-instant.

### Retry wrapper

All embedding calls go through a retry wrapper with exponential backoff and
jitter. Ollama occasionally returns transient errors under heavy batch load;
the retry layer handles these transparently (3 retries, starting at 100ms
backoff).

### One vector per symbol

Each symbol gets exactly one embedding vector. We don't create multiple
vectors per symbol (e.g., one for the signature and one for the body).
This keeps the design simple: the embeddings table has a 1:1 relationship
with the symbols table, storage is predictable, and search scans a single
vector per symbol.

The enriched embed text (combining file path, signature, doc comment,
snippet, calls, and imports into one string) ensures the single vector
captures multiple facets of the symbol's meaning.

---

## MCP Server: The Agent Interface

ctx++ runs as an MCP (Model Context Protocol) server over stdio. The host
application (Claude, Cursor, VS Code, etc.) spawns the `ctxpp mcp` process
and communicates via JSON-RPC over stdin/stdout.

### Tools exposed to the agent

| Tool | Purpose |
|------|---------|
| `ctxpp_search` | Search by keyword, semantic similarity, or hybrid (default) |
| `ctxpp_index` | Trigger a full or incremental reindex |
| `ctxpp_file_skeleton` | List all symbols in a file with signatures and line ranges |
| `ctxpp_feature_traverse` | BFS walk of the call graph outward from a symbol (callees) |
| `ctxpp_blast_radius` | Find everything that calls a given symbol (callers) |

`feature_traverse` and `blast_radius` are complementary: traverse walks
forward through callees ("what does this function use?"), while blast_radius
walks backward through callers ("what uses this function?"). Traverse starts
from a keyword search for exact name matches, then expands outward up to a
configurable depth (default 3 hops), deduplicating visited symbols at each
level.

### stdio pipe deadlock prevention

MCP servers communicate via stdin/stdout. If the server writes log output
to stderr, and the host application doesn't drain stderr fast enough, the
stderr pipe buffer (64KB on Linux) fills up and the server blocks -- a
classic pipe deadlock. ctx++ redirects all log output to a file at
`.ctxpp/server.log` to avoid this.

---

## Decisions and Trade-offs

### Why Go?

- Single binary with minimal runtime dependencies.
- Goroutines and channels map naturally to the pipeline architecture.
- Tree-sitter bindings via `smacker/go-tree-sitter` (cgo) for fast,
  accurate AST parsing.
- Pure-Go SQLite via `modernc.org/sqlite` (no cgo required for storage).
- Good performance for the CPU-bound parsing stage.

### Why SQLite instead of a dedicated vector database?

A vector database (Pinecone, Qdrant, Milvus, etc.) would provide faster
approximate nearest neighbor search. But it would also require:
- A separate server process to install and manage.
- Network overhead for every query.
- Configuration, authentication, and operational complexity.

SQLite gives us everything in a single file with zero setup. The brute-force
scan is fast enough for the target scale (developer workstations, not
million-document corpora).

### Why not use an existing embedding cache?

ctx++ stores embeddings in SQLite rather than a separate cache because:
- The embedding is keyed by symbol ID, not by text content. If a symbol's
  source changes, the old embedding is replaced atomically with the new
  symbol data.
- SQLite's transaction guarantees mean the symbols table and embeddings
  table are always consistent -- there's no window where a symbol exists
  without its embedding or vice versa.
- A single `DELETE FROM files WHERE path=?` cascades to symbols, embeddings,
  call edges, and import edges. No orphaned data.

### Why brute-force cosine instead of HNSW/ANN?

At the scale ctx++ targets (up to ~300k symbols), brute-force is fast
enough and always correct. HNSW introduces:
- Index build time and memory overhead.
- Approximate results (might miss the true top-K).
- Complexity in handling incremental updates (rebalancing the graph).

The door is open to add ANN later if needed. The `SearchSemantic` function
is the only code that needs to change.

### Why RRF instead of learned re-ranking?

Reciprocal Rank Fusion requires no training data, no model, and no tuning
beyond the smoothing constant k. A learned re-ranker (cross-encoder, etc.)
would need labeled relevance judgments for code search queries, which don't
exist at scale. RRF is a strong baseline that "just works."

### Why enrich embedding text with calls and imports?

A function's meaning isn't fully captured by its signature and doc comment
alone. A function called `Process` that calls `ValidateToken`, `FetchUser`,
and `IssueSession` is clearly about authentication -- but the embedding model
can't know that unless we tell it. By including call targets and import paths
in the embedding text, the vector captures the function's role in the broader
system.

The trade-off is slightly longer embedding text (and thus slightly higher
indexing time). We cap calls at 10 and imports at 6 to keep it bounded.

---

## Performance Characteristics

Measured on a Ryzen 9 9950X with 128 GiB RAM and local Ollama GPU:

| Metric | Private monorepo (9k files) | Kubernetes (28k files) |
|--------|----------------------------|----------------------|
| Index time | 56 seconds | 47 minutes |
| Symbols | ~52k | ~318k |
| DB size | 94.5 MiB | 1.9 GiB |
| Peak RSS | 354 MiB | 1.1 GiB |
| Search p50 | 412 us | ~640 ms |

The dominant cost in search latency for large codebases is the brute-force
vector scan (~615ms for 318k vectors). For codebases under 100k symbols,
search is well under 10ms. The per-query Ollama embedding call adds ~25ms
regardless of corpus size.

Memory usage scales linearly with the number of embedded symbols. Each
1024-dimensional float32 vector is 4 KiB, so 318k vectors consume ~1.2 GiB
in the SQLite mmap.

---

## What's Not Here (and Why)

- **HTTP transport**: MCP over stdio is simpler and avoids the need for
  port management, CORS, and authentication. HTTP transport can be added
  later without changing the search or indexing logic.

- **Spectral clustering**: grouping related symbols into clusters was
  explored but deferred. The call-graph re-ranking provides a lighter-weight
  version of the same benefit.

- **Multi-language parsers** (Java, JavaScript, Rust, Python): the parser
  interface is designed for this from day one. Adding a new language means
  implementing the `Parser` interface and registering it in `allParsers()`.
  Go and Protobuf are the v1 priorities.

- **Native vector extensions** (sqlite-vss, etc.): would speed up the
  vector scan but add a native dependency. Brute-force covers v1 scale.

- **Remote storage**: the index is local-only. A team-shared index would
  require conflict resolution and authentication, which is out of scope.
