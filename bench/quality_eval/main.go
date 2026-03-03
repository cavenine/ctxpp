// quality_eval runs search queries against a ctx++ index and prints
// the top-5 results for each query, intended for manual quality grading.
//
// Usage:
//
//	go run ./bench/quality_eval -db /tmp/bench-k8s-ctxpp/.ctxpp/index.db
//	go run ./bench/quality_eval -db /tmp/bench-k8s-ctxpp/.ctxpp/index.db -mode hybrid
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/cavenine/ctxpp/internal/types"
)

var k8sQueries = []string{
	"pod scheduling and node affinity",
	"container runtime interface CRI",
	"service discovery and endpoint routing",
	"RBAC authorization and role binding",
	"persistent volume claim provisioning",
	"kubelet pod lifecycle management",
	"API server admission controller webhook",
	"horizontal pod autoscaler scaling logic",
	"etcd storage and watch mechanism",
	"network policy enforcement and filtering",
}

var queryPresets = map[string][]string{
	"k8s": k8sQueries,
}

var queries = k8sQueries

func main() {
	dbPath := flag.String("db", "", "path to ctx++ index.db")
	ollamaURL := flag.String("ollama", "http://localhost:11434", "Ollama base URL")
	model := flag.String("model", "nomic-embed-text", "embedding model")
	limit := flag.Int("limit", 5, "number of results per query")
	mode := flag.String("mode", "semantic", "search mode: semantic, hybrid")
	preset := flag.String("preset", "k8s", "query preset: k8s")
	flag.Parse()

	if qs, ok := queryPresets[*preset]; ok {
		queries = qs
	} else {
		log.Fatalf("unknown preset %q; valid presets: k8s", *preset)
	}

	if *dbPath == "" {
		log.Fatal("-db is required")
	}

	st, err := store.Open(*dbPath)
	if err != nil {
		log.Fatalf("store.Open: %v", err)
	}
	defer st.Close()

	ctx := context.Background()

	// Use embed.Detect() to pick up the configured backend (Ollama, TEI,
	// Bedrock, etc.) from CTXPP_EMBED_BACKEND and related env vars. This
	// ensures query-time embedding uses the same model as index-time.
	// The -ollama and -model flags serve as fallbacks when Detect() returns
	// a bundled (stub) embedder, for backward compatibility.
	embedder, usingExternal := embed.Detect(ctx)
	if !usingExternal {
		// No external backend detected via env vars — fall back to explicit
		// Ollama flags for backward compatibility with older invocations.
		embedder = embed.NewOllamaEmbedder(*ollamaURL, *model, "")
		log.Printf("Using Ollama embedder (%s) -- set CTXPP_EMBED_BACKEND for other backends", *model)
	} else {
		log.Printf("Using detected embedder: %s (%dd)", embedder.Model(), embedder.Dims())
	}

	var allLatencies []time.Duration

	fmt.Printf("Mode: %s\n", *mode)

	for i, q := range queries {
		fmt.Printf("\n=== Q%d: %s ===\n\n", i+1, q)

		start := time.Now()
		vec, err := embedder.Embed(ctx, q)
		if err != nil {
			log.Printf("  embed error: %v", err)
			continue
		}
		embedDur := time.Since(start)

		searchStart := time.Now()
		var results []types.Symbol
		switch *mode {
		case "hybrid":
			results, err = st.SearchHybrid(vec, q, *limit)
			if err == nil && len(results) > 1 {
				results, _ = st.RerankByCallGraph(results)
			}
		default: // semantic
			results, err = st.SearchSemantic(vec, *limit)
		}
		if err != nil {
			log.Printf("  search error: %v", err)
			continue
		}
		searchDur := time.Since(searchStart)
		allLatencies = append(allLatencies, searchDur)

		fmt.Printf("  (embed: %s, search: %s)\n\n", embedDur, searchDur)

		for j, sym := range results {
			fmt.Printf("  %d. [%s] %s", j+1, sym.Kind, sym.Name)
			if sym.Receiver != "" {
				fmt.Printf(" (receiver: %s)", sym.Receiver)
			}
			fmt.Printf("\n     file: %s:%d-%d\n", sym.File, sym.StartLine, sym.EndLine)
			if sym.Signature != "" {
				sig := sym.Signature
				if len(sig) > 100 {
					sig = sig[:100] + "..."
				}
				fmt.Printf("     sig:  %s\n", sig)
			}
			if sym.DocComment != "" {
				doc := sym.DocComment
				if len(doc) > 120 {
					doc = doc[:120] + "..."
				}
				fmt.Printf("     doc:  %s\n", doc)
			}
			fmt.Printf("     tier: %d\n", sym.SourceTier)
		}
	}

	// Summary latencies.
	if len(allLatencies) > 0 {
		var total time.Duration
		for _, d := range allLatencies {
			total += d
		}
		fmt.Printf("\n--- Search latency summary ---\n")
		fmt.Printf("  queries: %d\n", len(allLatencies))
		fmt.Printf("  total:   %s\n", total)
		fmt.Printf("  avg:     %s\n", total/time.Duration(len(allLatencies)))
	}
}
