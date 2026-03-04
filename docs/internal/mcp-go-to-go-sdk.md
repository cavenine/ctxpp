# Migration Plan: mcp-go → modelcontextprotocol/go-sdk

## Overview

Port ctx++ from the community MCP SDK (`github.com/mark3labs/mcp-go v0.44.1`) to the official Go SDK (`github.com/modelcontextprotocol/go-sdk v1.2.0`).

## Current State

- **Current dependency**: `github.com/mark3labs/mcp-go v0.44.1`
- **Target dependency**: `github.com/modelcontextprotocol/go-sdk v1.2.0`
- **Branch**: No migration work started. Current branch is `feature/caching-embedder`.

## Files Requiring Changes

| File | mcp-go Usage |
|------|-------------|
| `cmd/mcp.go` | Server creation, tool registration, stdio transport |
| `cmd/handlers.go` | Handler signatures (`mcp.CallToolRequest` / `*mcp.CallToolResult`), param extraction (`req.GetString`, `req.GetInt`), result construction (`mcp.NewToolResultText`) |
| `cmd/handlers_test.go` | Test helpers building `mcp.CallToolRequest`, extracting `mcp.TextContent` from results |
| `bench/compare/main.go` | MCP client (`client.NewStdioMCPClientWithOptions`, `transport.WithCommandFunc`, `mcp.InitializeRequest`, `c.CallTool`) |

## API Differences

| Concern | mcp-go (current) | go-sdk (target) |
|---------|-------------------|-----------------|
| **Server creation** | `server.NewMCPServer("name", "ver", opts...)` | `mcp.NewServer(&mcp.Implementation{Name, Version}, nil)` |
| **Tool definition** | Builder pattern: `mcp.NewTool("name", mcp.WithString(...))` | Struct + Go struct tags: `&mcp.Tool{Name, Description}` with typed `Input` struct using `json`/`jsonschema` tags |
| **Handler signature** | `func(ctx, mcp.CallToolRequest) (*mcp.CallToolResult, error)` | `func(ctx, *mcp.CallToolRequest, InputStruct) (*mcp.CallToolResult, OutputStruct, error)` |
| **Param extraction** | `req.GetString("key", "default")` | Automatic JSON deserialization into typed `Input` struct |
| **Result construction** | `mcp.NewToolResultText("...")` | Return `(nil, output, nil)` where output is a typed struct |
| **Tool registration** | `s.AddTool(tool, handler)` | `mcp.AddTool(server, tool, handler)` (package-level generic function) |
| **Stdio transport** | `server.NewStdioServer(s).Listen(ctx, stdin, stdout)` | `server.Run(ctx, &mcp.StdioTransport{})` |
| **Client (bench)** | `client.NewStdioMCPClientWithOptions(...)` | Different client API (needs research) |

## Discoveries

- `go-sdk` uses a typed generic handler signature which leverages Go struct tags for JSON Schema generation, replacing the builder pattern in `mcp-go`.
- `go-sdk` has built-in support for OAuth, session management, and task tools.
- `go-sdk` has a stable v1.2.0 release compared to `mcp-go`'s v0.44.1.
- The `go-sdk` tool handler returns a 3-tuple `(*mcp.CallToolResult, OutputStruct, error)`. When the first return value is `nil`, the SDK auto-marshals the `OutputStruct` into the result content.

## Migration Steps

### 1. Create branch

Create `feature/migrate-go-sdk` from `main` (merge `feature/caching-embedder` first if ready).

### 2. Update go.mod

```bash
go get github.com/modelcontextprotocol/go-sdk@v1.2.0
```

Remove `github.com/mark3labs/mcp-go` from `go.mod` (unless retained for the bench client).

### 3. Define typed input structs

Create input structs for each of the 5 tools in `cmd/handlers.go`:

```go
type IndexInput struct {
    Path string `json:"path" jsonschema:"Path to the project root to index. Defaults to CTXPP_PROJECT env var or current directory."`
}

type SearchInput struct {
    Query string  `json:"query" jsonschema:"required,Search query."`
    Mode  string  `json:"mode,omitempty" jsonschema:"enum=keyword|semantic|hybrid,Search mode."`
    Limit float64 `json:"limit,omitempty" jsonschema:"Maximum number of results to return. Default: 10."`
}

type FileSkeletonInput struct {
    Path string `json:"path" jsonschema:"required,Path to the source file, relative to the project root."`
}

type FeatureTraverseInput struct {
    Query string  `json:"query" jsonschema:"required,Symbol name to start traversal from."`
    Depth float64 `json:"depth,omitempty" jsonschema:"Maximum hops to traverse in the call graph. Default: 3."`
}

type BlastRadiusInput struct {
    Symbol string `json:"symbol" jsonschema:"required,The symbol name to find references for."`
}
```

### 4. Rewrite cmd/mcp.go

- Replace `server.NewMCPServer(...)` with `mcp.NewServer(&mcp.Implementation{...}, nil)`
- Replace `s.AddTool(mcp.NewTool(...), handler)` with `mcp.AddTool(server, &mcp.Tool{...}, handler)`
- Replace `server.NewStdioServer(s).Listen(ctx, os.Stdin, os.Stdout)` with `server.Run(ctx, &mcp.StdioTransport{})`

### 5. Rewrite cmd/handlers.go

- Update each handler signature from `func(ctx, mcp.CallToolRequest) (*mcp.CallToolResult, error)` to `func(ctx, *mcp.CallToolRequest, InputStruct) (*mcp.CallToolResult, OutputStruct, error)`
- Remove manual `req.GetString`/`req.GetInt` calls -- replaced by typed input struct fields
- Replace `mcp.NewToolResultText(...)` with returning a typed output struct or constructing the result directly

### 6. Rewrite cmd/handlers_test.go

- Update `makeToolRequest` helper to match new `mcp.CallToolRequest` structure
- Update `getResultText` helper to match new result types
- Verify all 20+ existing handler tests still pass

### 7. Migrate bench/compare/main.go

This is the most uncertain step. The `go-sdk` client API may not expose `WithCommandFunc` for subprocess management. Options:

- **Option A**: Rewrite bench client to use `go-sdk`'s client API
- **Option B**: Keep `mcp-go` as a dev dependency only for `bench/compare/`
- **Option C**: Rewrite bench client to use raw JSON-RPC over stdio pipes

### 8. Verify

```bash
go build ./...
go test ./...
go test -race ./...
go vet ./...
```

## Open Questions

1. **Bench client strategy**: Should we keep `mcp-go` as a dependency solely for `bench/compare/main.go`, or rewrite the bench client against the `go-sdk` client API? The subprocess management (`WithCommandFunc`, `SysProcAttr`) is custom enough that a raw JSON-RPC approach may be simplest.

2. **Branch strategy**: Should `feature/caching-embedder` be merged to `main` before starting the migration branch, or should the migration branch be based off `feature/caching-embedder`?

3. **Output struct pattern**: The `go-sdk` handler signature returns `(result, output, error)`. For our tools that return free-form text (JSON-serialized symbol lists), we need to determine whether to use a simple `TextOutput struct { Text string }` wrapper or construct `*mcp.CallToolResult` directly and return `(result, zero, nil)`.
