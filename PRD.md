# ctx++ Product Requirements Document

## Overview

ctx++ is a fast, local MCP (Model Context Protocol) server that gives AI coding agents structured access to large codebases. It combines AST-aware symbol extraction, vector and full-text search, and call graph traversal to let agents find relevant code quickly and trace how a named symbol connects to the rest of the codebase — without reading raw files.

---

## Problem Statement

AI coding agents need to understand codebases to be effective. Today they have two strategies:

1. **Read files directly** -- expensive in tokens, noisy, and slow. Reading many files to answer one question floods the context window with irrelevant code.
2. **Use an indexing MCP tool** -- existing tools (Context+, Codemogger) improve on raw file reads but have limitations:
   - Context+ stores embeddings in a flat JSON file and builds them lazily on the first search call, so it has not fully indexed the corpus when you start querying. On the kubernetes codebase (28k files), the first search call took 2m 7s.
   - Codemogger is fast and reliable for search but provides no call graph data -- it cannot trace which symbols a given function calls, or what calls it.
   - Neither tool can answer "show me the full call graph of feature X" in a single query.

The result is that AI agents on large codebases either flood context with raw files or issue multiple search round-trips to piece together a feature's shape.

---

## Goals

**v1**

- Sub-100ms search response on a 10,000-symbol index after initial index build
- Zero timeout failures from the MCP server under normal operation
- Accurate symbol extraction for Go, C, C++, Java, JavaScript, TypeScript, Rust, SQL, Protobuf, Markdown, HTML, Shell, and HTTP request files
- Call graph traversal: given a symbol name, return the set of symbols it transitively calls (BFS, configurable depth, default 3 hops)
- Blast radius: given a symbol name, return all symbols that call it
- No required external dependencies for baseline operation (keyword search works without Ollama)
- Ollama integration for semantic search embeddings (default model: `bge-m3`)
- AWS Bedrock integration for cloud-hosted embedding (Titan Text Embeddings V2)
- Incremental reindex on file change -- branch switches self-heal automatically
- Single `.ctxpp/index.db` file per codebase, easy to `.gitignore`

**Non-goals for v1**

- Spectral clustering / semantic navigation (deferred to v2)
- Static analysis / linting (deferred to v2)
- Code writing, restore points, or version control integration (out of scope -- AI coding agents handle this)
- Remote/cloud index storage
- HTTP/SSE transport (stdio only in v1)

---

## Target Users

- Developers using AI coding agents (OpenCode, Claude Code, Cursor, Windsurf, VS Code) on medium-to-large codebases (1,000-100,000 files)
- Teams where the codebase is too large for the AI to read files directly without hitting context limits
- Developers who want the AI to understand cross-cutting features (authentication, billing, data pipelines) without manually curating documentation

---

## Architecture

### Language

Go. Reasons:
- Native tree-sitter bindings via `github.com/tree-sitter/go-tree-sitter` -- full C performance, no WASM overhead
- Single statically linked binary, no runtime dependencies
- Excellent SQLite support via `modernc.org/sqlite` (pure Go, no CGO required for storage)
- Strong concurrency primitives for parallel file indexing
- Easy cross-platform distribution

### Storage

SQLite via `modernc.org/sqlite` with FTS5 for full-text search and brute-force cosine similarity scan for vector search (adequate for codebases up to ~100k symbols; no native vector extension required).

Schema (conceptual):

```
files        -- path, language, sha256, indexed_at
symbols      -- file_id, name, kind, start_line, end_line, signature, docstring
embeddings   -- symbol_id, model, vector (stored as blob)
call_edges   -- caller_symbol, callee_symbol (short names, from AST analysis)
import_edges -- file_id, imported_path, resolved_file_id
```

FTS5 virtual table over `symbols(name, signature, docstring)` for keyword search.

Brute-force cosine similarity scan over `embeddings` for semantic search (adequate for v1 scale; no native vector extension required).

Both indexes live in `.ctxpp/index.db` at the project root.

### Parsing

Tree-sitter via `github.com/tree-sitter/go-tree-sitter` for languages with available grammars:

| Language | Grammar package |
|---|---|
| Go | `github.com/tree-sitter/tree-sitter-go` |
| C | `github.com/tree-sitter/tree-sitter-c` |
| C++ | `github.com/tree-sitter/tree-sitter-cpp` |
| Java | `github.com/tree-sitter/tree-sitter-java` |
| JavaScript | `github.com/tree-sitter/tree-sitter-javascript` |
| TypeScript | `github.com/tree-sitter/tree-sitter-typescript` |
| Rust | `github.com/tree-sitter/tree-sitter-rust` |
| SQL | `github.com/cavenine/tree-sitter-sql` |
| Protobuf | `github.com/tree-sitter/tree-sitter-proto` |
| Markdown | `github.com/tree-sitter/tree-sitter-markdown` |
| HTML | `github.com/tree-sitter/tree-sitter-html` |
| Shell | `github.com/tree-sitter/tree-sitter-bash` |

HTTP request files (`.http`/`.rest`) use a lightweight line-based parser (no tree-sitter grammar exists for this format).

Each language implements a `Parser` interface:

```go
type Parser interface {
    Language()   string
    Extensions() []string
    Parse(path string, src []byte) (Result, error)
}
```

This abstraction keeps language-specific logic isolated and makes adding new grammars a single-file addition.

### Embeddings

Two backends behind an `Embedder` interface:

```go
type Embedder interface {
    Model() string
    Dims()  int
    Embed(ctx context.Context, text string) ([]float32, error)
}
```

A `BatchEmbedder` extension interface is also defined for backends that support batched calls (Ollama's `/api/embed` endpoint), which amortises per-call overhead and enables GPU batching.

**Ollama** (default model `bge-m3`, 1024 dimensions): selected through head-to-head quality benchmarks against multiple models (all-MiniLM-L6-v2, nomic-embed-text, snowflake-arctic-embed variants, mxbai-embed-large) on real codebases. bge-m3 produced the most relevant search results for code symbol retrieval. Requires [Ollama](https://ollama.com) running locally; uses GPU when available.

**AWS Bedrock** (Titan Text Embeddings V2, 1024 dimensions): cloud-hosted alternative; no local GPU required. Benchmarked on kubernetes/kubernetes (318k symbols): 4.7/5 search quality vs 4.8/5 for Ollama/bge-m3, within noise. Higher per-query latency (95–460 ms vs ~25 ms for local GPU) offset by horizontal concurrency (tested at 200 parallel calls).

**Stub fallback**: when no backend is detected, the embedding pipeline is skipped. No zero-vector embeddings are stored. Keyword search via FTS5 remains fully functional; semantic and hybrid search are unavailable.

Backend selection at startup (controlled by `CTXPP_EMBED_BACKEND`):
1. `auto` (default): probe Ollama at `CTXPP_OLLAMA_URL`; if reachable, use it. Otherwise fall back to stub with a warning.
2. `ollama`: use Ollama unconditionally.
3. `bedrock`: use AWS Bedrock (requires `CTXPP_BEDROCK_REGION`, `CTXPP_BEDROCK_MODEL`).
4. `tei`: use a Text Embeddings Inference server (`CTXPP_TEI_URL`).

Embeddings are stored with the model name. If the model changes, affected embeddings are invalidated and re-computed incrementally.

### Call Graph

The call graph is built at index time from tree-sitter AST analysis:

- **Call edges**: which symbol names are called inside a given symbol's body. Stored as short name pairs (`caller_symbol`, `callee_symbol`); not fully scope-resolved.
- **Import edges**: which files import which other files. Stored but not currently used at query time.

At query time, call graph data is used in two ways:

1. **Re-ranking in hybrid search**: after RRF fusion, symbols that share call edges with other symbols already in the top-K results are boosted by position. This is a local re-ranking pass, not a graph traversal.

2. **BFS traversal in `ctxpp_feature_traverse`**: starting from an exact-name seed symbol, outward callee edges are followed hop-by-hop up to a configurable depth (default: 3). Returns a flat list of all reachable symbols ordered by BFS distance.

### Incremental Indexing

A file watcher monitors the project directory for changes (debounced, 500ms). On change:

1. Compute SHA-256 of changed file
2. Compare against stored hash in `files` table
3. If changed: re-parse symbols, re-embed, update call/import edges, update FTS index
4. Deleted files: cascade delete symbols, embeddings, edges

Initial index is parallelized across a worker pool (default: `runtime.NumCPU()` workers). Embedding concurrency is controlled separately by `CTXPP_EMBED_CONCURRENCY`.

---

## MCP Tools (v1)

### `ctxpp_index`
Index or reindex the target codebase. Accepts an optional path; defaults to current working directory. Returns indexing statistics (files processed, symbols extracted, duration).

### `ctxpp_search`
Search for symbols by keyword or natural language query. Supports three modes:
- `keyword` -- FTS5 full-text search over symbol names and signatures. Fastest, precise for known identifiers.
- `semantic` -- vector similarity search over embeddings. Best for natural language queries when you don't know the exact name.
- `hybrid` -- FTS + vector, results merged by Reciprocal Rank Fusion, then optionally re-ranked by call graph connectivity. Default mode.

Returns: symbol name, kind (function/method/struct/etc.), file path, line range, signature, and a short code snippet.

### `ctxpp_file_skeleton`
Given a file path, return all top-level symbols with their signatures and line ranges -- without reading the full file body. Lets the AI understand a file's API surface cheaply before deciding which functions to read in full.

### `ctxpp_feature_traverse`
Given an exact symbol name, walk the call graph outward (BFS, default depth 3) and return the flat list of reachable symbols ordered by hop distance. Useful for understanding what a function transitively depends on.

Note: takes an exact symbol name, not a natural language query. If you don't know the name, use `ctxpp_search` first.

### `ctxpp_blast_radius`
Given a symbol name, return all symbols in the index that directly call it. Answers "what calls this?" before making a change.

---

## Configuration

All configuration via environment variables (MCP servers cannot accept flags interactively):

| Variable | Default | Description |
|---|---|---|
| `CTXPP_PROJECT` | cwd | Path to the project root to index |
| `CTXPP_EMBED_BACKEND` | `auto` | Embedding backend: `auto`, `ollama`, `bedrock`, `tei` |
| `CTXPP_OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `CTXPP_OLLAMA_MODEL` | `bge-m3` | Ollama embedding model |
| `CTXPP_OLLAMA_SOCKET` | _(unset)_ | Unix socket path for Ollama (bypasses TCP) |
| `CTXPP_BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `CTXPP_BEDROCK_MODEL` | `amazon.titan-embed-text-v2:0` | Bedrock model ID |
| `CTXPP_BEDROCK_DIMS` | `1024` | Bedrock embedding dimensions (256, 512, or 1024) |
| `CTXPP_TEI_URL` | `http://localhost:8080` | Text Embeddings Inference server URL |
| `CTXPP_EMBED_CONCURRENCY` | `runtime.NumCPU()` | Max concurrent embedding calls |

---

## Non-Functional Requirements

- **Cold start**: MCP server must be ready to accept tool calls within 2 seconds of launch (index load, not index build)
- **Search latency**: p99 < 200ms for keyword and hybrid search on a 10,000-symbol index
- **Feature traverse latency**: p99 < 500ms at depth 3 on a 10,000-symbol index
- **Index size**: no hard target; expect larger than competitors because ctx++ stores symbol-level embeddings (vs file-level or chunk-level). On kubernetes (318k symbols): 1.9 GiB.
- **Memory**: RSS grows with corpus size due to in-process vector scan. On kubernetes (318k symbols, 1024d): ~1.1 GiB. Acceptable for developer workstations; monitor for larger corpora.
- **Reliability**: zero MCP timeout failures under normal operation (all tool calls complete or return a structured error within the MCP timeout window)

---

## v2 Considerations (Out of Scope for v1)

- Spectral clustering for codebase-wide semantic navigation
- Static analysis integration (go vet, tsc --noEmit, etc.)
- Cross-repository indexing
- HTTP/SSE transport for remote MCP use
- Language server protocol (LSP) integration for richer type-resolved call graphs
- Index sharing / team cache (shared `.ctxpp/index.db` via git LFS or similar)
- Import edge utilization at query time (currently stored but unused)
- Natural language seed for `ctxpp_feature_traverse` (currently requires exact symbol name)
