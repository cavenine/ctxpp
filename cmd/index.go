package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/spf13/cobra"
)

func newIndexCmd() *cobra.Command {
	var path string

	cmd := &cobra.Command{
		Use:   "index",
		Short: "Index or reindex a project codebase",
		Long: `Walk the project root, parse all supported source files, extract symbols,
and store them in the local index database (.ctxpp/index.db).

Subsequent runs are incremental: files whose content has not changed since the
last index pass are skipped. Use 'backfill' to re-embed symbols after switching
embedding backends.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			runIndex(path)
			return nil
		},
	}

	cmd.Flags().StringVarP(&path, "path", "p", "", "Path to the project root (default: $CTXPP_PROJECT or current directory)")

	return cmd
}

// runIndex is the "ctxpp index" subcommand.
// It opens the store, detects the embedder, indexes, prints stats, and exits.
func runIndex(path string) {
	ctx := context.Background()

	root := path
	if root == "" {
		root = os.Getenv("CTXPP_PROJECT")
	}
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			slog.Error("getwd", "err", err)
			os.Exit(1)
		}
	}
	root, _ = filepath.Abs(root)

	dbDir := filepath.Join(root, ".ctxpp")
	if err := os.MkdirAll(dbDir, 0o755); err != nil {
		slog.Error("mkdir .ctxpp", "err", err)
		os.Exit(1)
	}
	st, err := store.Open(filepath.Join(dbDir, "index.db"))
	if err != nil {
		slog.Error("open store", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	embedder, usingOllama := embed.Detect(ctx)
	if usingOllama {
		slog.Info("embedder: active", "model", embedder.Model())
	} else {
		fmt.Fprintln(os.Stderr, "WARNING: No embedding backend detected -- indexing with keyword search only.")
		fmt.Fprintln(os.Stderr, "         Install Ollama (https://ollama.com) and run 'ollama pull bge-m3',")
		fmt.Fprintln(os.Stderr, "         or set CTXPP_EMBED_BACKEND=bedrock for AWS Bedrock.")
	}

	idx := indexer.New(indexer.Config{ProjectRoot: root}, st, allParsers(), embedder)
	stats, err := idx.Index(ctx)
	if err != nil {
		slog.Error("index", "err", err)
		os.Exit(1)
	}

	fmt.Printf("root:    %s\nindexed: %d files, %d symbols\nskipped: %d files (unchanged)\ntime:    %s\n",
		root, stats.FilesIndexed, stats.SymbolsIndexed, stats.FilesSkipped, stats.Duration)
}
