# ctx++ Agent Instructions

## Project

ctx++ (`github.com/cavenine/ctxpp`) is a fast, local MCP server written in Go that provides AI agents with deep codebase intelligence via symbol extraction, FTS+vector search, incremental indexing, and call-graph traversal.

## Repository Layout

```
cmd/ctxpp/          # main binary (MCP server entry point)
internal/
  types/            # shared data types (Symbol, CallEdge, ImportEdge, FileRecord)
  store/            # SQLite persistence layer (FTS5 + brute-force vector search)
  parser/           # Parser interface + per-language implementations (Go first)
  embed/            # Embedder interface + Ollama and bundled backends
  indexer/          # File walker, worker pool, incremental reindex, fsnotify watcher
```

## Go Conventions

- **Minimum Go version**: see `go.mod`.
- **Error handling**: always wrap errors with `fmt.Errorf("context: %w", err)`; never swallow errors silently.
- **Naming**: follow standard Go naming — exported types are `PascalCase`, unexported are `camelCase`; avoid stutter (`store.Store`, not `store.StoreStore`).
- **Interfaces**: keep interfaces small and defined at the consumer, not the implementer.
- **Contexts**: accept `context.Context` as the first parameter on any I/O or long-running function.
- **Concurrency**: prefer `sync.WaitGroup` + channels over goroutine leaks; always document goroutine ownership.
- **Packages**: keep packages focused; avoid circular imports; `internal/` packages are private to this module.
- **No global state**: pass dependencies explicitly; avoid `init()` side effects.
- **Logging**: use `log/slog` with structured key-value pairs; no `fmt.Println` in library code.
- **Formatting**: code must pass `gofmt` and `go vet` with zero warnings.
- **Linting**: target zero issues from `staticcheck ./...`.

## Testing

- **Style**: table-driven tests using `[]struct{ name string; ... }` slices with `t.Run(tc.name, ...)`.
- **Naming**: test files are `*_test.go` in the same package (prefer white-box) or `*_test` package (black-box where appropriate).
- **Assertions**: use only the standard library (`testing` package); no external assertion libraries.
- **Subtests**: always use `t.Run` for each table case so failures are individually identified.
- **Helpers**: extract repeated setup into `t.Helper()`-annotated helper functions.
- **Benchmarks**: add `Benchmark*` functions alongside tests for any hot path (search, embedding, parse).
- **Fixtures**: place small source-code fixtures under `testdata/`; load with `os.ReadFile` or `os.DirFS`.

### TDD: Red/Green Cycle

Follow strict vertical-slice TDD — one behavior at a time, never horizontal (all tests then all code):

1. **Write one failing test** for a single observable behavior. Run it and confirm it is RED before writing any implementation.
2. **Write the minimum implementation** to make that test pass. Run it and confirm GREEN.
3. **Repeat** for the next behavior. Do not write the next test until the current cycle is GREEN.
4. **Never write speculative tests** for behavior that doesn't exist yet. Tests describe what the system does, not what you imagine it will do.
5. **Refactor only while GREEN** — never while a test is failing.

### SQLite / FTS5 Known Gotcha

FTS5 content-table virtual tables must be joined on `rowid`, not on `UNINDEXED` columns:

```go
// WRONG — UNINDEXED columns cannot be referenced via the FTS table alias:
JOIN symbols s ON s.id = f.symbol_id

// CORRECT — join via rowid:
JOIN symbols s ON s.rowid = f.rowid
```

### Table-Driven Test Template

```go
func TestFoo(t *testing.T) {
    tests := []struct {
        name    string
        input   string
        want    string
        wantErr bool
    }{
        {name: "happy path", input: "x", want: "y"},
        {name: "empty input", input: "", wantErr: true},
    }
    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            got, err := Foo(tc.input)
            if (err != nil) != tc.wantErr {
                t.Fatalf("Foo() error = %v, wantErr %v", err, tc.wantErr)
            }
            if got != tc.want {
                t.Errorf("Foo() = %q, want %q", got, tc.want)
            }
        })
    }
}
```

## Parser Strategy

- **Tree-sitter first**: if a tree-sitter grammar exists for a language or file type, use it. Do not write a custom regex/line-based parser when tree-sitter support is available.
- **Custom parsers only as fallback**: only use lightweight regex or line-scanning parsers for file types that have no tree-sitter grammar (e.g., `.http`/`.rest` request files, plain text/config files).
- **Rationale**: tree-sitter grammars provide robust, AST-accurate extraction with far fewer edge cases than hand-rolled regex parsers.

## Key Design Decisions

- **Index location**: `.ctxpp/index.db` in the project root (configurable via `CTXPP_PROJECT` env var).
- **MCP transport**: stdio only (`server.NewStdioServer`).
- **Embedding backend**: auto-detect Ollama at `http://localhost:11434`; fall back to bundled model.
- **Embedding cache**: embeddings are persisted in SQLite; never re-embed a symbol whose source hash hasn't changed.
- **Incremental indexing**: skip files whose SHA-256 matches the stored record; only reparse changed/new files.
- **Parallelism**: indexer uses a bounded worker pool (default: `runtime.NumCPU()` workers).
- **Vector search**: brute-force cosine similarity scan in Go (adequate for ≤100k symbols); no native extension required.
- **v1 languages**: Go. Interfaces are designed for multi-language from day one.

## Commands

```bash
# Build
go build ./...

# Test (all packages)
go test ./...

# Test with race detector
go test -race ./...

# Vet
go vet ./...

# Run MCP server (stdio)
./ctxpp mcp

# Index a project
./ctxpp index [path]
```

## What to Defer (out of v1 scope)

- Spectral clustering, HTTP transport, remote storage, code-writing / restore points.
- Java, JavaScript/TypeScript, Rust parsers (interfaces exist; implementations are v2+).
- Native SQLite vector extension (brute-force scan covers v1 scale).
