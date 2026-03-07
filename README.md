![ctx++](ctxpp.png)

# ctx++
Fast, accurate codebase intelligence for AI coding agents.

ctx++ is an MCP (Model Context Protocol) server that gives AI agents precise, structured understanding of large codebases. It extracts symbols using native tree-sitter parsing, indexes them in SQLite with both full-text and vector search, and traces call graphs to automatically map how features are assembled across files -- no hand-maintained documentation required.

---

## Why ctx++

ctx++ is built around three principles:

1. **Never time out.** SQLite with indexed vector and FTS search means queries are fast regardless of codebase size. The MCP server loads in under 2 seconds and all tool calls complete within the MCP timeout window.
2. **Return the right code, not just matching code.** The call graph traversal finds everything involved in a feature by walking real call relationships -- the same way a senior engineer would read the code.
3. **Minimal setup.** A single Go binary plus [Ollama](https://ollama.com) for embeddings. No cloud services, no API keys, no Docker.

---

## Tools

| Tool | Description |
|---|---|
| `ctxpp_index` | Index or reindex the codebase. Run once after install; incremental updates happen automatically. |
| `ctxpp_search` | Search by identifier name (keyword) or natural language (semantic). Returns symbol definitions with file paths and line numbers. |
| `ctxpp_file_skeleton` | Return all symbols in a file with signatures and line ranges, without reading the full body. Cheap way to understand a file's API surface. |
| `ctxpp_feature_traverse` | Given an exact symbol name, return related symbols by walking the call graph outward via BFS. The auto-generated feature hub. |
| `ctxpp_blast_radius` | Given a symbol, return every location in the codebase that references it. Answers "what breaks if I change this?" |
| `ctxpp_ann_status` | Return the current ANN search status, including whether ANN is active, rebuilding, missing, or disabled. |

---

## Supported Languages

| Language | Extensions | Symbols Extracted |
|---|---|---|
| Go | `.go` | functions, methods, structs, interfaces, types, constants, variables |
| Java | `.java` | classes, interfaces, enums, methods, constructors, fields |
| Kotlin | `.kt`, `.kts` | functions, methods, classes, interfaces, properties, imports |
| JavaScript | `.js`, `.mjs`, `.cjs`, `.jsx` | functions, classes, methods, arrow functions |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | functions, classes, interfaces, type aliases, enums |
| Rust | `.rs` | functions, structs, enums, traits, impl methods, type aliases |
| C# | `.cs` | classes, interfaces, methods, fields, imports |
| C | `.c`, `.h` | functions, structs, enums, typedefs, function-like macros |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx` | functions, methods, classes, structs, enums, namespaces, templates |
| SQL | `.sql` | tables, views, indexes, functions, procedures, triggers |
| Markdown | `.md`, `.mdx` | headings (as sections) |
| HTML | `.html`, `.htm` | headings, script/style blocks |
| Shell | `.sh`, `.bash`, `.zsh`, `.dash` | functions |
| Protobuf | `.proto` | messages, services, RPCs, enums |
| HTTP | `.http`, `.rest` | named requests |
| Text/Config | `.txt`, `.env`, `Makefile`, `Dockerfile`, `LICENSE`, etc. | file-level document symbol |

Want to add another language? See `docs/ADDING-LANGUAGE-SUPPORT.md` for a step-by-step implementation template and PR checklist.

---

## Prerequisites

- **Go 1.24+** for building from source
- **[Ollama](https://ollama.com)** for semantic search embeddings

```bash
# Install Ollama, then pull the default embedding model:
ollama pull bge-m3
```

Without Ollama, ctx++ still works but provides keyword search only. Semantic search and feature traversal quality depend on embeddings.

---

## Install

```bash
go install github.com/cavenine/ctxpp@latest
```

Or build from source:

```bash
git clone https://github.com/cavenine/ctxpp
cd ctxpp
make build
```

---

## Quick Start

### 1. Index your project

```bash
ctxpp index --path /path/to/your/project
```

This creates `.ctxpp/index.db` in the project root. Add it to `.gitignore`:

```
.ctxpp/
```

Subsequent runs only re-process changed files. Branch switches self-heal automatically via the file watcher.

If parser logic changes but your source files do not, force a full reparse of supported files:

```bash
ctxpp index --path /path/to/your/project --force
```

### 2. Add to your MCP config

The examples below use Ollama with `bge-m3` (the default). If Ollama is not running, omit `CTXPP_OLLAMA_*` — ctx++ will fall back to keyword search only.

To enable the ANN vector index explicitly, set `CTXPP_VECTOR_INDEX=ann`. To let ctx++ decide automatically, use the default `CTXPP_VECTOR_INDEX=auto`.

**OpenCode** (`opencode.json` in project root):

```json
{
  "mcp": {
    "ctxpp": {
      "type": "local",
      "command": ["ctxpp", "mcp"],
      "enabled": true,
      "environment": {
        "CTXPP_PROJECT": "/path/to/your/project",
        "CTXPP_VECTOR_INDEX": "ann",
        "CTXPP_OLLAMA_URL": "http://localhost:11434",
        "CTXPP_OLLAMA_MODEL": "bge-m3"
      }
    }
  }
}
```

**Claude Code** (`.mcp.json`):

```json
{
  "mcpServers": {
    "ctxpp": {
      "command": "ctxpp",
      "args": ["mcp"],
      "env": {
        "CTXPP_PROJECT": "/path/to/your/project",
        "CTXPP_VECTOR_INDEX": "ann",
        "CTXPP_OLLAMA_URL": "http://localhost:11434",
        "CTXPP_OLLAMA_MODEL": "bge-m3"
      }
    }
  }
}
```

**Cursor / Windsurf** (`.cursor/mcp.json` or `.windsurf/mcp.json`):

```json
{
  "mcpServers": {
    "ctxpp": {
      "command": "ctxpp",
      "args": ["mcp"],
      "env": {
        "CTXPP_PROJECT": "/path/to/your/project",
        "CTXPP_VECTOR_INDEX": "ann",
        "CTXPP_OLLAMA_URL": "http://localhost:11434",
        "CTXPP_OLLAMA_MODEL": "bge-m3"
      }
    }
  }
}
```

### 3. Use it

Ask your AI agent anything about the codebase:

```
use ctxpp to show me everything involved in account authentication
```

```
use ctxpp to find where FetchAccount is defined and what calls it
```

```
use ctxpp_blast_radius to tell me what breaks if I change the Account struct
```

```
use ctxpp_ann_status to check whether ANN search is healthy or rebuilding
```

### 4. ANN usage

ctx++ supports three vector-index modes:

- `CTXPP_VECTOR_INDEX=auto` — default; use ANN when healthy artifacts are present, otherwise fall back to brute-force search
- `CTXPP_VECTOR_INDEX=bruteforce` — always use SQLite-backed brute-force vector search
- `CTXPP_VECTOR_INDEX=ann` — prefer ANN and build/load ANN artifacts under `.ctxpp/`

Current status: ANN is implemented, fast, and usable for experimentation, but it is still **experimental**. The brute-force path remains the relevance baseline until ANN recall quality is improved on very large corpora.

ANN artifacts live next to the main SQLite index:

```
.ctxpp/
  index.db
  ann-hnsw.bin
  ann-hnsw.json
```

Recommended interactive setup with ANN enabled:

```bash
export CTXPP_PROJECT=/path/to/your/project
export CTXPP_VECTOR_INDEX=ann
export CTXPP_OLLAMA_MODEL=bge-m3
ctxpp index --path "$CTXPP_PROJECT"
ctxpp mcp
```

Check ANN health at any time:

```bash
ctxpp_ann_status
```

Or from an MCP client / agent prompt:

```
use ctxpp_ann_status to confirm ANN is healthy before running semantic search
```

Compare ANN against the brute-force baseline on an existing index:

```bash
go run ./bench/ann_eval -db .ctxpp/index.db -mode quality
go run ./bench/ann_eval -db .ctxpp/index.db -mode latency
```

Use `-format json` for machine-readable output and `-search semantic` to compare
semantic-only retrieval instead of the default hybrid path.

---

## Ollama Integration

ctx++ uses [Ollama](https://ollama.com) for embedding-based semantic search. The default model is `bge-m3` (BAAI's BGE-M3, 1024 dimensions), which was selected through head-to-head quality benchmarks against multiple models on real codebases.

```bash
ollama pull bge-m3
```

ctx++ auto-detects Ollama on `localhost:11434` at startup. If Ollama is not running, ctx++ falls back to keyword search only and prints a warning.

To use a different embedding model (e.g., `all-minilm` for faster indexing at the cost of some search quality):

```json
"environment": {
  "CTXPP_PROJECT": "/path/to/your/project",
  "CTXPP_OLLAMA_MODEL": "all-minilm"
}
```

---

## AWS Bedrock Integration

For environments without a local GPU, ctx++ can use [Amazon Titan Text Embeddings V2](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) via AWS Bedrock. Quality is comparable to the Ollama/bge-m3 default (4.7/5 vs 4.8/5 on the kubernetes benchmark).

**Prerequisites**: AWS credentials configured via `~/.aws/credentials`, `AWS_PROFILE`, or IAM role. The identity needs `bedrock:InvokeModel` permission on `amazon.titan-embed-text-v2:0`.

Set `CTXPP_EMBED_BACKEND=bedrock` and the following env vars:

**OpenCode** (`opencode.json` in project root):

```json
{
  "mcp": {
    "ctxpp": {
      "type": "local",
      "command": ["ctxpp", "mcp"],
      "enabled": true,
      "environment": {
        "CTXPP_PROJECT": "/path/to/your/project",
        "CTXPP_EMBED_BACKEND": "bedrock",
        "CTXPP_BEDROCK_REGION": "us-east-1",
        "CTXPP_BEDROCK_MODEL": "amazon.titan-embed-text-v2:0",
        "CTXPP_BEDROCK_DIMS": "1024",
        "CTXPP_EMBED_CONCURRENCY": "100"
      }
    }
  }
}
```

**Claude Code** (`.mcp.json`):

```json
{
  "mcpServers": {
    "ctxpp": {
      "command": "ctxpp",
      "args": ["mcp"],
      "env": {
        "CTXPP_PROJECT": "/path/to/your/project",
        "CTXPP_EMBED_BACKEND": "bedrock",
        "CTXPP_BEDROCK_REGION": "us-east-1",
        "CTXPP_BEDROCK_MODEL": "amazon.titan-embed-text-v2:0",
        "CTXPP_BEDROCK_DIMS": "1024",
        "CTXPP_EMBED_CONCURRENCY": "100"
      }
    }
  }
}
```

**Cursor / Windsurf** (`.cursor/mcp.json` or `.windsurf/mcp.json`):

```json
{
  "mcpServers": {
    "ctxpp": {
      "command": "ctxpp",
      "args": ["mcp"],
      "env": {
        "CTXPP_PROJECT": "/path/to/your/project",
        "CTXPP_EMBED_BACKEND": "bedrock",
        "CTXPP_BEDROCK_REGION": "us-east-1",
        "CTXPP_BEDROCK_MODEL": "amazon.titan-embed-text-v2:0",
        "CTXPP_BEDROCK_DIMS": "1024",
        "CTXPP_EMBED_CONCURRENCY": "100"
      }
    }
  }
}
```

Or for initial indexing from the command line:

```bash
export CTXPP_EMBED_BACKEND=bedrock
export CTXPP_BEDROCK_REGION=us-east-1
export CTXPP_BEDROCK_MODEL=amazon.titan-embed-text-v2:0
export CTXPP_BEDROCK_DIMS=1024
export CTXPP_EMBED_CONCURRENCY=100   # increase to 200 for large repos
ctxpp index --path /path/to/your/project
```

**Trade-offs vs Ollama:**

| | Ollama (local GPU) | Bedrock |
|--|--|--|
| Per-query embed latency | ~25ms | 100-460ms |
| Index time (kubernetes, 318K symbols) | 47m | ~7.5h |
| GPU required | Yes | No |
| Cost | Free (local) | AWS API pricing |
| Horizontal scaling | Limited by GPU | High (100-200 concurrent) |
| Quality (kubernetes benchmark) | 4.8/5 | 4.7/5 |

Bedrock is the right choice for CI/CD pipelines, cloud-hosted agents, or developer machines without a GPU. For interactive development with a GPU available, Ollama is faster.

---

## Configuration

All configuration is via environment variables.

| Variable | Default | Description |
|---|---|---|
| `CTXPP_PROJECT` | `.` | Path to the project root to index |
| `CTXPP_VECTOR_INDEX` | `auto` | Vector search engine: `auto`, `bruteforce`, or `ann` |
| `CTXPP_OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `CTXPP_OLLAMA_MODEL` | `bge-m3` | Ollama embedding model |
| `CTXPP_EMBED_BACKEND` | _(auto-detect)_ | Embedding backend: `ollama` or `bedrock` |
| `CTXPP_WORKERS` | number of CPUs | Parallel workers for initial indexing |
| `CTXPP_EMBED_CONCURRENCY` | `10` | Max concurrent embedding requests (Bedrock) |

---

## CLI Reference

```
ctxpp index [--path/-p <path>] [--force]  Index or reindex a project (default: $CTXPP_PROJECT or current directory)
ctxpp backfill [--path/-p <path>]  Re-embed symbols missing embedding vectors
ctxpp mcp                          Start the MCP server over stdio
ctxpp version                      Print version
```

---

## Architecture

ctx++ is written in Go and built on:

- **[go-tree-sitter](https://github.com/tree-sitter/go-tree-sitter)** -- native C tree-sitter bindings for fast, accurate AST parsing across all supported languages
- **SQLite** via `modernc.org/sqlite` (pure Go, no CGO) with FTS5 for full-text search and brute-force cosine similarity for vector search
- **[Ollama](https://ollama.com)** for embedding generation (default model: `bge-m3`)
- **MCP SDK** for stdio-based MCP transport

The index lives in a single `.ctxpp/index.db` file per project. The schema tracks files, symbols, embeddings, call edges, and import edges. All queries hit indexed columns -- no full-table scans, no loading the entire index into memory.

See [PRD.md](PRD.md) for full architecture and design decisions.

---

## How Feature Traversal Works

When you ask `ctxpp_feature_traverse` about a symbol (e.g. `"HandleLogin"`):

1. A keyword search finds symbols with that exact name in the index
2. The call graph is walked outward from each seed via BFS — what does this function call? What do those functions call?
3. Results are returned in BFS order (seed first, then direct callees, then transitive callees) up to the configured depth (default: 3 hops)

This gives you the full call tree rooted at a symbol — useful for understanding what a function orchestrates without reading every file manually. Use `ctxpp_blast_radius` for the reverse direction: what calls this function?

---

## License

MIT
