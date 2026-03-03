package cmd

import (
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/spf13/cobra"
)

func newMCPCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "mcp",
		Short: "Start the ctx++ MCP server (stdio transport)",
		Long:  `Start the Model Context Protocol server. Communicates via stdin/stdout.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			runMCP()
			return nil
		},
	}
}

func runMCP() {
	ctx := context.Background()

	// In MCP mode, the server communicates via stdin/stdout. If slog writes
	// thousands of log lines to stderr (e.g. one per indexed file), the stderr
	// pipe buffer fills up (64KB on Linux) and the server blocks — a classic
	// pipe deadlock. Redirect logs to a file inside .ctxpp/ or discard them.
	//
	// Resolve project root: CTXPP_PROJECT env > current directory.
	root := os.Getenv("CTXPP_PROJECT")
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			slog.Error("getwd", "err", err)
			os.Exit(1)
		}
	}
	root, _ = filepath.Abs(root)

	// Open (or create) the index database.
	dbDir := filepath.Join(root, ".ctxpp")
	if err := os.MkdirAll(dbDir, 0o755); err != nil {
		slog.Error("mkdir .ctxpp", "err", err)
		os.Exit(1)
	}

	// Redirect slog to a log file so stderr pipe doesn't fill up.
	// When run as an MCP server, stdout/stderr are pipes; writing thousands
	// of log lines to stderr causes a pipe buffer deadlock (64KB limit).
	logFile, err := os.OpenFile(filepath.Join(dbDir, "server.log"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	} else {
		defer logFile.Close()
		slog.SetDefault(slog.New(slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelInfo})))
	}

	st, err := store.Open(filepath.Join(dbDir, "index.db"))
	if err != nil {
		slog.Error("open store", "err", err)
		os.Exit(1)
	}

	// Auto-detect embedder (Ollama preferred, bundled fallback).
	// Wrap with CachingEmbedder so repeated identical query texts don't hit
	// the backend on every search call.
	baseEmbedder, usingOllama := embed.Detect(ctx)
	embedder := embed.NewCachingEmbedder(baseEmbedder)
	if usingOllama {
		slog.Info("embedder: active", "model", embedder.Model())
	} else {
		slog.Warn("No embedding backend detected -- semantic search disabled. Install Ollama (https://ollama.com) and run 'ollama pull bge-m3', or set CTXPP_EMBED_BACKEND=bedrock for AWS Bedrock.")
	}

	parsers := allParsers()

	idx := indexer.New(indexer.Config{ProjectRoot: root}, st, parsers, embedder)

	a := &app{
		store:    st,
		indexer:  idx,
		embedder: embedder,
		root:     root,
	}

	s := server.NewMCPServer(
		"ctx++",
		"0.0.1",
		server.WithToolCapabilities(true),
	)

	s.AddTool(mcp.NewTool("ctxpp_index",
		mcp.WithDescription("Index or reindex the project codebase. Run once after setup; incremental updates happen automatically via the file watcher."),
		mcp.WithString("path",
			mcp.Description("Path to the project root to index. Defaults to CTXPP_PROJECT env var or current directory."),
		),
	), a.handleIndex)

	s.AddTool(mcp.NewTool("ctxpp_search",
		mcp.WithDescription("Search for symbols by identifier name (keyword mode) or natural language (semantic mode). Returns symbol definitions with file paths and line numbers."),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Search query. Use an exact identifier name for keyword search, or a natural language description for semantic search."),
		),
		mcp.WithString("mode",
			mcp.Description("Search mode: 'keyword' (FTS, fastest), 'semantic' (vector similarity), or 'hybrid' (default, combines both)."),
			mcp.Enum("keyword", "semantic", "hybrid"),
		),
		mcp.WithNumber("limit",
			mcp.Description("Maximum number of results to return. Default: 10."),
		),
	), a.handleSearch)

	s.AddTool(mcp.NewTool("ctxpp_file_skeleton",
		mcp.WithDescription("Return all top-level symbols in a file with their signatures and line ranges, without reading the full file body."),
		mcp.WithString("path",
			mcp.Required(),
			mcp.Description("Path to the source file, relative to the project root."),
		),
	), a.handleFileSkeleton)

	s.AddTool(mcp.NewTool("ctxpp_feature_traverse",
		mcp.WithDescription("Given a symbol name, return the full map of related symbols by walking the call graph. This is the auto-generated feature hub."),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Symbol name to start traversal from (e.g. 'HandleLogin')."),
		),
		mcp.WithNumber("depth",
			mcp.Description("Maximum hops to traverse in the call graph. Default: 3."),
		),
	), a.handleFeatureTraverse)

	s.AddTool(mcp.NewTool("ctxpp_blast_radius",
		mcp.WithDescription("Given a symbol name, return every location that references it: callers, importers, type usages. Answers 'what breaks if I change this?'"),
		mcp.WithString("symbol",
			mcp.Required(),
			mcp.Description("The symbol name to find references for (e.g. 'FetchAccount')."),
		),
	), a.handleBlastRadius)

	// Start background watcher (non-blocking; errors are logged).
	watchCtx, cancelWatch := context.WithCancel(ctx)
	defer cancelWatch()
	go func() {
		if err := idx.Watch(watchCtx); err != nil {
			slog.Warn("watcher stopped", "err", err)
		}
	}()

	if err := server.NewStdioServer(s).Listen(ctx, os.Stdin, os.Stdout); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}
