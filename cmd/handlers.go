package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/types"
	"github.com/mark3labs/mcp-go/mcp"
)

func (a *app) handleIndex(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	root := a.root
	if p := req.GetString("path", ""); p != "" {
		root, _ = filepath.Abs(p)
	}

	ctx := context.Background()
	// Re-create indexer with potentially different root.
	idx := indexer.New(indexer.Config{ProjectRoot: root}, a.store,
		allParsers(), a.embedder)

	stats, err := idx.Index(ctx)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("index error: %v", err)), nil
	}
	return mcp.NewToolResultText(fmt.Sprintf(
		"indexed project at %s\nfiles indexed: %d, skipped: %d, symbols: %d, duration: %s",
		root, stats.FilesIndexed, stats.FilesSkipped, stats.SymbolsIndexed, stats.Duration,
	)), nil
}

func (a *app) handleSearch(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query := req.GetString("query", "")
	mode := req.GetString("mode", "hybrid")
	if mode == "" {
		mode = "hybrid"
	}
	limit := req.GetInt("limit", 10)
	if limit <= 0 {
		limit = 10
	}

	var syms []types.Symbol
	var err error

	switch mode {
	case "keyword":
		syms, err = a.store.SearchKeyword(query, limit)
	case "semantic":
		vec, eerr := a.embedder.Embed(ctx, query)
		if eerr != nil {
			return mcp.NewToolResultText(fmt.Sprintf("embed error: %v", eerr)), nil
		}
		syms, err = a.store.SearchSemantic(vec, limit)
	default: // hybrid
		vec, _ := a.embedder.Embed(ctx, query)
		syms, err = a.store.SearchHybrid(vec, query, limit)
	}

	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("search error: %v", err)), nil
	}

	// Apply call-graph re-ranking to boost densely connected results.
	if len(syms) > 1 {
		syms, _ = a.store.RerankByCallGraph(syms)
	}

	return mcp.NewToolResultText(marshalSymbols(syms)), nil
}

func (a *app) handleFileSkeleton(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path := req.GetString("path", "")
	if path == "" {
		return mcp.NewToolResultText("path is required"), nil
	}
	syms, err := a.store.GetSymbolsByFile(path)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}
	if len(syms) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("no symbols found for %s (has it been indexed?)", path)), nil
	}
	return mcp.NewToolResultText(marshalSymbols(syms)), nil
}

func (a *app) handleFeatureTraverse(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query := req.GetString("query", "")
	depth := req.GetInt("depth", 3)
	if depth <= 0 {
		depth = 3
	}

	// BFS: start with seed symbols matching the query name.
	seeds, err := a.store.SearchKeyword(query, 10)
	if err != nil || len(seeds) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("no symbols found for %q", query)), nil
	}

	// Filter to exact name matches.
	var current []types.Symbol
	visitedIDs := make(map[string]bool)
	for _, s := range seeds {
		if s.Name == query && !visitedIDs[s.ID] {
			visitedIDs[s.ID] = true
			current = append(current, s)
		}
	}
	if len(current) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("no exact match for %q", query)), nil
	}

	result := append([]types.Symbol{}, current...)

	// BFS: expand callees hop by hop.
	for hop := 1; hop <= depth && len(current) > 0; hop++ {
		// Collect all callee names from this frontier.
		calleeNameSet := make(map[string]bool)
		for _, sym := range current {
			// call_edges stores caller_symbol as the short function name,
			// not the full symbol ID.
			callees, _ := a.store.CalleeSymbols(sym.Name)
			for _, c := range callees {
				calleeNameSet[c] = true
			}
		}

		if len(calleeNameSet) == 0 {
			break
		}

		// Batch-resolve callee names to symbols.
		names := make([]string, 0, len(calleeNameSet))
		for n := range calleeNameSet {
			names = append(names, n)
		}
		resolved, _ := a.store.GetSymbolsByNames(names)

		// Deduplicate and build next frontier.
		var next []types.Symbol
		for _, s := range resolved {
			if !visitedIDs[s.ID] {
				visitedIDs[s.ID] = true
				next = append(next, s)
				result = append(result, s)
			}
		}
		current = next
	}

	return mcp.NewToolResultText(marshalSymbols(result)), nil
}

func (a *app) handleBlastRadius(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	symbol := req.GetString("symbol", "")
	if symbol == "" {
		return mcp.NewToolResultText("symbol is required"), nil
	}

	// callee_symbol is stored as the short name; CallerSymbols returns
	// caller short names (not full IDs), so resolve via GetSymbolsByNames.
	callerNames, err := a.store.CallerSymbols(symbol)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	type blastResult struct {
		Symbol  string         `json:"symbol"`
		Callers []types.Symbol `json:"callers"`
	}

	// Batch-resolve caller names to full symbols.
	callerSyms, err := a.store.GetSymbolsByNames(callerNames)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error resolving callers: %v", err)), nil
	}

	out, _ := json.MarshalIndent(blastResult{Symbol: symbol, Callers: callerSyms}, "", "  ")
	return mcp.NewToolResultText(string(out)), nil
}
