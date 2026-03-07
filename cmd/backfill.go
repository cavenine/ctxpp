package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/spf13/cobra"
)

func newBackfillCmd() *cobra.Command {
	var path string

	cmd := &cobra.Command{
		Use:   "backfill",
		Short: "Re-embed symbols that are missing embedding vectors",
		Long: `Scan the index for symbols that have no embedding vector and embed them
using the currently configured backend. No source files are re-parsed.

Useful after:
  - Switching embedding backends (e.g. Ollama → Bedrock)
  - Recovering from an interrupted index run
  - Fixing a bug in the embedder that left some symbols un-embedded

Partial backfills are resumable: re-run until the missing count reaches zero.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			runBackfill(path)
			return nil
		},
	}

	cmd.Flags().StringVarP(&path, "path", "p", "", "Path to the project root (default: $CTXPP_PROJECT or current directory)")

	return cmd
}

// runBackfill is the "ctxpp backfill" subcommand.
// It re-embeds all symbols in the index that are missing an embedding vector,
// without re-parsing any source files. This is useful after:
//   - Switching embedding backends (e.g. Ollama → Bedrock)
//   - Recovering from an interrupted index run
//   - Fixing a bug in the embedder that left some symbols un-embedded
//
// Progress is printed to stdout. Errors on individual symbols are logged and
// skipped (nil vectors are not stored). The function exits 0 even if some
// symbols fail, since partial backfills are resumable: re-run until the
// missing count reaches zero.
func runBackfill(path string) {
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
	storeOpts, err := parseStoreOpenOptionsFromEnv()
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse store options: %v\n", err)
		os.Exit(1)
	}
	st, err := store.OpenWithOptions(filepath.Join(dbDir, "index.db"), storeOpts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open store: %v\n", err)
		os.Exit(1)
	}
	defer st.Close()

	embedder, active := embed.Detect(ctx)
	if !active {
		fmt.Fprintln(os.Stderr, "WARNING: No embedding backend detected. Set CTXPP_EMBED_BACKEND=bedrock or start Ollama before running backfill.")
		fmt.Fprintln(os.Stderr, "         Backfill will store zero vectors (stub embedder).")
	} else {
		fmt.Printf("embedder: %s (%d dims)\n", embedder.Model(), embedder.Dims())
	}

	ids, err := st.SymbolIDsWithoutEmbeddings()
	if err != nil {
		fmt.Fprintf(os.Stderr, "query missing embeddings: %v\n", err)
		os.Exit(1)
	}
	if len(ids) == 0 {
		fmt.Println("backfill: nothing to do — all symbols are already embedded.")
		return
	}
	fmt.Printf("backfill: %d symbols missing embeddings\n", len(ids))

	const batchSize = 500
	batcher, isBatcher := embedder.(embed.BatchEmbedder)

	var (
		done    int
		skipped int
	)
	start := time.Now()
	st.BeginDeferredANNSync()
	defer func() {
		if err := st.EndDeferredANNSync(); err != nil {
			fmt.Fprintf(os.Stderr, "EndDeferredANNSync: %v\n", err)
			os.Exit(1)
		}
	}()

	for lo := 0; lo < len(ids); lo += batchSize {
		hi := lo + batchSize
		if hi > len(ids) {
			hi = len(ids)
		}
		chunk := ids[lo:hi]

		syms, err := st.GetSymbolsByIDs(chunk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GetSymbolsByIDs: %v\n", err)
			os.Exit(1)
		}

		texts := make([]string, len(syms))
		for i, s := range syms {
			texts[i] = indexer.BuildEmbedText(s)
		}

		var vecs [][]float32
		if isBatcher {
			vecs, err = batcher.EmbedBatch(ctx, texts)
			if err != nil {
				fmt.Fprintf(os.Stderr, "EmbedBatch error (chunk %d-%d): %v\n", lo, hi, err)
				skipped += len(syms)
				continue
			}
		} else {
			vecs = make([][]float32, len(texts))
			for i, t := range texts {
				v, eerr := embedder.Embed(ctx, t)
				if eerr != nil {
					slog.Warn("backfill embed failed", "id", syms[i].ID, "err", eerr)
					skipped++
					continue
				}
				vecs[i] = v
			}
		}

		// Build items, skipping nil vectors.
		items := make([]store.EmbeddingItem, 0, len(syms))
		for i, s := range syms {
			if i >= len(vecs) || vecs[i] == nil {
				continue
			}
			items = append(items, store.EmbeddingItem{
				SymbolID: s.ID,
				Model:    embedder.Model(),
				Vector:   vecs[i],
			})
		}

		if len(items) > 0 {
			if err := st.UpsertEmbeddingsBatch(items); err != nil {
				fmt.Fprintf(os.Stderr, "UpsertEmbeddingsBatch: %v\n", err)
				os.Exit(1)
			}
		}
		done += len(items)

		fmt.Printf("\r  embedded %d / %d (skipped %d)    ", done, len(ids), skipped)
	}

	elapsed := time.Since(start)
	fmt.Printf("\nbackfill complete: %d embedded, %d skipped, %s elapsed\n", done, skipped, elapsed.Round(time.Millisecond))
	if skipped > 0 {
		fmt.Println("  Re-run 'ctxpp backfill' to retry skipped symbols.")
	}
}
