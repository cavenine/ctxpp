// Package main compares search quality between embedding models.
// Usage: go run ./bench/embed_compare
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/cavenine/ctxpp/internal/types"
)

type modelConfig struct {
	name   string
	dbPath string
	model  string
}

func main() {
	configs := []modelConfig{
		{"all-minilm", "/tmp/nebula-retina-minilm/.ctxpp/index.db", "all-minilm"},
		{"mxbai-embed-large", "/tmp/nebula-retina-mxbai/.ctxpp/index.db", "mxbai-embed-large"},
		{"bge-m3", "/tmp/nebula-retina-bge/.ctxpp/index.db", "bge-m3"},
	}

	// Queries: mix of exact identifiers, natural language, and domain concepts
	queries := []struct {
		query string
		desc  string
	}{
		// Exact identifier lookups (keyword should ace these)
		{"CreateAccount", "exact function name"},
		{"HandleLogin", "exact handler name"},
		{"MachineStore", "exact type name"},

		// Natural language / conceptual queries (semantic should shine)
		{"authentication and login flow", "auth concept"},
		{"how are machines grouped", "grouping concept"},
		{"license validation and entitlements", "licensing concept"},
		{"error handling middleware", "error handling concept"},
		{"database migration", "migration concept"},
		{"firewall rule management", "firewall concept"},
		{"user role permissions and authorization", "RBAC concept"},
		{"threat detection and malware scanning", "security concept"},
		{"policy configuration for endpoints", "policy concept"},
	}

	limit := 5
	ctx := context.Background()

	for _, q := range queries {
		fmt.Printf("\n=== Query: %q (%s) ===\n", q.query, q.desc)
		fmt.Println(strings.Repeat("-", 80))

		for _, cfg := range configs {
			st, err := store.Open(cfg.dbPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "open %s: %v\n", cfg.dbPath, err)
				continue
			}

			// Semantic search only (that's where models differ)
			embedder := embed.NewOllamaEmbedder("http://localhost:11434", cfg.model, "")
			semStart := time.Now()
			vec, embedErr := embedder.Embed(ctx, q.query)
			var semResults []types.Symbol
			var semErr error
			if embedErr == nil {
				semResults, semErr = st.SearchSemantic(vec, limit)
			} else {
				semErr = embedErr
			}
			semDur := time.Since(semStart)

			fmt.Printf("  [%s] (%v):\n", cfg.name, semDur.Round(time.Microsecond))
			if semErr != nil {
				fmt.Printf("    ERROR: %v\n", semErr)
			} else {
				printResults(semResults)
			}

			st.Close()
		}
	}
}

func printResults(syms []types.Symbol) {
	if len(syms) == 0 {
		fmt.Println("      (no results)")
		return
	}
	for i, s := range syms {
		sig := s.Signature
		if len(sig) > 60 {
			sig = sig[:57] + "..."
		}
		fmt.Printf("      %d. %s.%s [%s] %s:%d\n", i+1, s.Package, s.Name, s.Kind, s.File, s.StartLine)
		if sig != "" {
			fmt.Printf("         sig: %s\n", sig)
		}
	}
}
