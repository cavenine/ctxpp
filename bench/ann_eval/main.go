// ann_eval compares ctx++ brute-force and ANN retrieval on an existing index.
//
// Usage:
//
//	go run ./bench/ann_eval -db /tmp/bench-k8s-ctxpp/.ctxpp/index.db
//	go run ./bench/ann_eval -db /tmp/bench-k8s-ctxpp/.ctxpp/index.db -mode latency
//	go run ./bench/ann_eval -db /tmp/bench-k8s-ctxpp/.ctxpp/index.db -format json
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/hnsw"
	"github.com/cavenine/ctxpp/internal/store"
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

type queryResult struct {
	Query            string                         `json:"query"`
	Mode             string                         `json:"mode"`
	EffectiveMode    string                         `json:"effective_mode"`
	CandidateLimit   int                            `json:"candidate_limit"`
	FallbackApplied  bool                           `json:"fallback_applied"`
	ANNStatusBefore  store.ANNStatus                `json:"ann_status_before"`
	ANNStatusAfter   store.ANNStatus                `json:"ann_status_after"`
	NilNeighborSkips uint64                         `json:"nil_neighbor_skips"`
	SearchCalls      uint64                         `json:"search_calls"`
	VisitedNodes     uint64                         `json:"visited_nodes"`
	ReturnedNodes    uint64                         `json:"returned_nodes"`
	LayerTraversals  uint64                         `json:"layer_traversals"`
	Candidates       []store.SemanticDebugCandidate `json:"candidates,omitempty"`
	IDs              []string                       `json:"ids"`
	Names            []string                       `json:"names"`
}

type queryDiff struct {
	Query                        string      `json:"query"`
	BruteForce                   queryResult `json:"bruteforce"`
	BruteForceWide               queryResult `json:"bruteforce_wide"`
	ANN                          queryResult `json:"ann"`
	IdenticalTopK                bool        `json:"identical_top_k"`
	BruteForceTop1InANNTop       bool        `json:"bruteforce_top_1_in_ann_top_k"`
	ExactTop1Match               bool        `json:"exact_top_1_match"`
	CandidateOverlapCount        int         `json:"candidate_overlap_count"`
	BruteForceTopKInANNCands     int         `json:"bruteforce_top_k_in_ann_candidates"`
	BruteForceWideTopKInANNCands int         `json:"bruteforce_wide_top_k_in_ann_candidates"`
}

type qualityReport struct {
	Preset                 string      `json:"preset"`
	SearchMode             string      `json:"search_mode"`
	Limit                  int         `json:"limit"`
	IdenticalTopKCount     int         `json:"identical_top_k_count"`
	BruteForceTop1InANNTop int         `json:"bruteforce_top_1_in_ann_top_k_count"`
	ExactTop1MatchCount    int         `json:"exact_top_1_match_count"`
	Comparisons            []queryDiff `json:"comparisons"`
}

type latencySample struct {
	Query            string          `json:"query"`
	EmbedDuration    time.Duration   `json:"embed_duration_ns"`
	SearchDuration   time.Duration   `json:"search_duration_ns"`
	FallbackApplied  bool            `json:"fallback_applied"`
	ANNStatusBefore  store.ANNStatus `json:"ann_status_before"`
	ANNStatusAfter   store.ANNStatus `json:"ann_status_after"`
	NilNeighborSkips uint64          `json:"nil_neighbor_skips"`
	SearchCalls      uint64          `json:"search_calls"`
	VisitedNodes     uint64          `json:"visited_nodes"`
	ReturnedNodes    uint64          `json:"returned_nodes"`
	LayerTraversals  uint64          `json:"layer_traversals"`
}

type latencyRun struct {
	Mode         string          `json:"mode"`
	Status       store.ANNStatus `json:"ann_status"`
	EmbedAvg     time.Duration   `json:"embed_avg_ns"`
	SearchAvg    time.Duration   `json:"search_avg_ns"`
	SearchP50    time.Duration   `json:"search_p50_ns"`
	OpenAndReady time.Duration   `json:"open_and_ready_ns,omitempty"`
	Samples      []latencySample `json:"samples"`
}

type latencyReport struct {
	Preset string       `json:"preset"`
	Limit  int          `json:"limit"`
	Runs   []latencyRun `json:"runs"`
}

func main() {
	var (
		dbPath     = flag.String("db", "", "path to ctx++ index.db")
		preset     = flag.String("preset", "k8s", "query preset")
		mode       = flag.String("mode", "quality", "report mode: quality or latency")
		searchMode = flag.String("search", "hybrid", "search mode for quality: semantic or hybrid")
		limit      = flag.Int("limit", 5, "number of results per query")
		format     = flag.String("format", "text", "output format: text or json")
	)
	flag.Parse()

	if *dbPath == "" {
		log.Fatal("-db is required")
	}
	queries, ok := queryPresets[*preset]
	if !ok {
		log.Fatalf("unknown preset %q", *preset)
	}

	ctx := context.Background()
	embedder, usingExternal := embed.Detect(ctx)
	if !usingExternal {
		log.Fatal("no embedding backend detected; set CTXPP_EMBED_BACKEND or Ollama env vars")
	}

	switch *mode {
	case "quality":
		report, err := runQuality(ctx, *dbPath, queries, *searchMode, *limit, *preset, embedder)
		if err != nil {
			log.Fatalf("run quality report: %v", err)
		}
		output(*format, report, printQualityReport)
	case "latency":
		report, err := runLatency(ctx, *dbPath, queries, *limit, *preset, embedder)
		if err != nil {
			log.Fatalf("run latency report: %v", err)
		}
		output(*format, report, printLatencyReport)
	default:
		log.Fatalf("unsupported -mode %q", *mode)
	}
}

func output(format string, v any, printText func(any)) {
	switch format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(v); err != nil {
			log.Fatalf("encode json: %v", err)
		}
	case "text":
		printText(v)
	default:
		log.Fatalf("unsupported -format %q", format)
	}
}

func runQuality(ctx context.Context, dbPath string, queries []string, searchMode string, limit int, preset string, embedder embed.Embedder) (*qualityReport, error) {
	bruteForce, err := runQuerySet(ctx, dbPath, store.SemanticModeBruteForce, queries, searchMode, limit, embedder)
	if err != nil {
		return nil, fmt.Errorf("run brute-force query set: %w", err)
	}
	ann, err := runQuerySet(ctx, dbPath, store.SemanticModeANN, queries, searchMode, limit, embedder)
	if err != nil {
		return nil, fmt.Errorf("run ann query set: %w", err)
	}
	bruteForceWide, err := runQuerySetWithCandidateLimit(ctx, dbPath, store.SemanticModeBruteForce, queries, searchMode, limit, 4096, embedder)
	if err != nil {
		return nil, fmt.Errorf("run brute-force wide query set: %w", err)
	}

	report := &qualityReport{
		Preset:      preset,
		SearchMode:  searchMode,
		Limit:       limit,
		Comparisons: make([]queryDiff, 0, len(queries)),
	}
	for i := range queries {
		diff := queryDiff{
			Query:          queries[i],
			BruteForce:     bruteForce[i],
			BruteForceWide: bruteForceWide[i],
			ANN:            ann[i],
		}
		diff.IdenticalTopK = sameStrings(diff.BruteForce.IDs, diff.ANN.IDs)
		diff.BruteForceTop1InANNTop = containsFirst(diff.BruteForce.IDs, diff.ANN.IDs)
		diff.ExactTop1Match = sameFirst(diff.BruteForce.IDs, diff.ANN.IDs)
		if diff.IdenticalTopK {
			report.IdenticalTopKCount++
		}
		if diff.BruteForceTop1InANNTop {
			report.BruteForceTop1InANNTop++
		}
		if diff.ExactTop1Match {
			report.ExactTop1MatchCount++
		}
		diff.CandidateOverlapCount = overlapCount(candidateRowIDs(diff.BruteForce.Candidates), candidateRowIDs(diff.ANN.Candidates))
		diff.BruteForceTopKInANNCands = overlapCount(candidateRowIDs(diff.BruteForce.Candidates[:min(len(diff.BruteForce.Candidates), report.Limit)]), candidateRowIDs(diff.ANN.Candidates))
		diff.BruteForceWideTopKInANNCands = overlapCount(candidateRowIDs(diff.BruteForceWide.Candidates[:min(len(diff.BruteForceWide.Candidates), report.Limit)]), candidateRowIDs(diff.ANN.Candidates))
		report.Comparisons = append(report.Comparisons, diff)
	}
	return report, nil
}

func runLatency(ctx context.Context, dbPath string, queries []string, limit int, preset string, embedder embed.Embedder) (*latencyReport, error) {
	runs := make([]latencyRun, 0, 2)
	bruteForce, err := runLatencyMode(ctx, dbPath, store.SemanticModeBruteForce, queries, limit, embedder)
	if err != nil {
		return nil, fmt.Errorf("run brute-force latency: %w", err)
	}
	runs = append(runs, bruteForce)

	openStart := time.Now()
	annStore, err := store.OpenWithOptions(dbPath, store.OpenOptions{SemanticMode: store.SemanticModeANN})
	if err != nil {
		return nil, fmt.Errorf("open ann store: %w", err)
	}
	for {
		status := annStore.ANNStatus()
		if status.State != store.ANNStateRebuilding {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if err := annStore.Close(); err != nil {
		return nil, fmt.Errorf("close ann store after warmup: %w", err)
	}
	ann, err := runLatencyMode(ctx, dbPath, store.SemanticModeANN, queries, limit, embedder)
	if err != nil {
		return nil, fmt.Errorf("run ann latency: %w", err)
	}
	ann.OpenAndReady = time.Since(openStart)
	runs = append(runs, ann)

	return &latencyReport{Preset: preset, Limit: limit, Runs: runs}, nil
}

func runQuerySet(ctx context.Context, dbPath string, mode store.SemanticMode, queries []string, searchMode string, limit int, embedder embed.Embedder) ([]queryResult, error) {
	return runQuerySetWithCandidateLimit(ctx, dbPath, mode, queries, searchMode, limit, 0, embedder)
}

func runQuerySetWithCandidateLimit(ctx context.Context, dbPath string, mode store.SemanticMode, queries []string, searchMode string, limit int, candidateLimit int, embedder embed.Embedder) ([]queryResult, error) {
	st, err := store.OpenWithOptions(dbPath, store.OpenOptions{SemanticMode: mode})
	if err != nil {
		return nil, fmt.Errorf("open store: %w", err)
	}
	defer st.Close()

	results := make([]queryResult, 0, len(queries))
	for _, query := range queries {
		vec, err := embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embed query %q: %w", query, err)
		}
		statusBefore := st.ANNStatus()
		hnsw.ResetDebugCounters()
		debug, err := debugSearch(st, vec, query, searchMode, limit, candidateLimit)
		if err != nil {
			return nil, fmt.Errorf("search query %q: %w", query, err)
		}
		result := queryResult{
			Query:            query,
			Mode:             string(mode),
			EffectiveMode:    string(debug.EffectiveMode),
			CandidateLimit:   debug.CandidateLimit,
			FallbackApplied:  debug.FallbackApplied,
			ANNStatusBefore:  statusBefore,
			ANNStatusAfter:   st.ANNStatus(),
			NilNeighborSkips: hnsw.NilNeighborSkips(),
			SearchCalls:      hnsw.SearchCalls(),
			VisitedNodes:     hnsw.SearchVisitedNodes(),
			ReturnedNodes:    hnsw.SearchReturnedNodes(),
			LayerTraversals:  hnsw.SearchLayerTraversals(),
			Candidates:       debug.Candidates,
		}
		for _, sym := range debug.Results {
			result.IDs = append(result.IDs, sym.ID)
			result.Names = append(result.Names, sym.Name)
		}
		results = append(results, result)
	}
	return results, nil
}

func runLatencyMode(ctx context.Context, dbPath string, mode store.SemanticMode, queries []string, limit int, embedder embed.Embedder) (latencyRun, error) {
	st, err := store.OpenWithOptions(dbPath, store.OpenOptions{SemanticMode: mode})
	if err != nil {
		return latencyRun{}, fmt.Errorf("open store: %w", err)
	}
	defer st.Close()

	samples := make([]latencySample, 0, len(queries))
	var embedDurations []time.Duration
	var searchDurations []time.Duration
	for _, query := range queries {
		embedStart := time.Now()
		vec, err := embedder.Embed(ctx, query)
		if err != nil {
			return latencyRun{}, fmt.Errorf("embed query %q: %w", query, err)
		}
		embedDuration := time.Since(embedStart)

		statusBefore := st.ANNStatus()
		hnsw.ResetDebugCounters()
		searchStart := time.Now()
		debug, err := st.DebugSemanticQuery(vec, limit)
		if err != nil {
			return latencyRun{}, fmt.Errorf("search query %q: %w", query, err)
		}
		searchDuration := time.Since(searchStart)

		embedDurations = append(embedDurations, embedDuration)
		searchDurations = append(searchDurations, searchDuration)
		samples = append(samples, latencySample{
			Query:            query,
			EmbedDuration:    embedDuration,
			SearchDuration:   searchDuration,
			FallbackApplied:  debug.FallbackApplied,
			ANNStatusBefore:  statusBefore,
			ANNStatusAfter:   st.ANNStatus(),
			NilNeighborSkips: hnsw.NilNeighborSkips(),
			SearchCalls:      hnsw.SearchCalls(),
			VisitedNodes:     hnsw.SearchVisitedNodes(),
			ReturnedNodes:    hnsw.SearchReturnedNodes(),
			LayerTraversals:  hnsw.SearchLayerTraversals(),
		})
	}

	return latencyRun{
		Mode:      string(mode),
		Status:    st.ANNStatus(),
		EmbedAvg:  averageDuration(embedDurations),
		SearchAvg: averageDuration(searchDurations),
		SearchP50: percentile50(searchDurations),
		Samples:   samples,
	}, nil
}

func debugSearch(st *store.Store, vec []float32, query string, searchMode string, limit int, candidateLimit int) (store.SemanticDebugResult, error) {
	switch searchMode {
	case "semantic":
		if candidateLimit > 0 {
			return st.DebugSemanticQueryWithCandidateLimit(vec, limit, candidateLimit)
		}
		return st.DebugSemanticQuery(vec, limit)
	case "hybrid":
		syms, err := st.SearchHybrid(vec, query, limit)
		if err != nil {
			return store.SemanticDebugResult{}, err
		}
		if len(syms) > 1 {
			reranked, err := st.RerankByCallGraph(syms)
			if err == nil {
				syms = reranked
			}
		}
		var debug store.SemanticDebugResult
		if candidateLimit > 0 {
			debug, err = st.DebugSemanticQueryWithCandidateLimit(vec, limit, candidateLimit)
		} else {
			debug, err = st.DebugSemanticQuery(vec, limit)
		}
		if err != nil {
			return store.SemanticDebugResult{}, err
		}
		debug.Results = syms
		return debug, nil
	default:
		return store.SemanticDebugResult{}, fmt.Errorf("unsupported search mode %q", searchMode)
	}
}

func candidateRowIDs(candidates []store.SemanticDebugCandidate) []int64 {
	out := make([]int64, 0, len(candidates))
	for _, candidate := range candidates {
		out = append(out, candidate.RowID)
	}
	return out
}

func overlapCount(a, b []int64) int {
	seen := make(map[int64]struct{}, len(a))
	for _, value := range a {
		seen[value] = struct{}{}
	}
	count := 0
	for _, value := range b {
		if _, ok := seen[value]; ok {
			count++
		}
	}
	return count
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func averageDuration(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	var total time.Duration
	for _, value := range values {
		total += value
	}
	return total / time.Duration(len(values))
}

func percentile50(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	copyValues := append([]time.Duration(nil), values...)
	sort.Slice(copyValues, func(i, j int) bool {
		return copyValues[i] < copyValues[j]
	})
	return copyValues[len(copyValues)/2]
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func containsFirst(a, b []string) bool {
	if len(a) == 0 {
		return false
	}
	for _, value := range b {
		if value == a[0] {
			return true
		}
	}
	return false
}

func sameFirst(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	return a[0] == b[0]
}

func printQualityReport(v any) {
	report, ok := v.(*qualityReport)
	if !ok {
		log.Fatalf("unexpected quality report type %T", v)
	}
	fmt.Printf("Preset: %s\n", report.Preset)
	fmt.Printf("Search mode: %s\n", report.SearchMode)
	fmt.Printf("Limit: %d\n\n", report.Limit)
	fmt.Printf("Identical top-%d: %d / %d\n", report.Limit, report.IdenticalTopKCount, len(report.Comparisons))
	fmt.Printf("Brute-force top-1 present in ANN top-%d: %d / %d\n", report.Limit, report.BruteForceTop1InANNTop, len(report.Comparisons))
	fmt.Printf("Exact top-1 match: %d / %d\n", report.ExactTop1MatchCount, len(report.Comparisons))
	for i, comparison := range report.Comparisons {
		fmt.Printf("\nQ%d: %s\n", i+1, comparison.Query)
		if comparison.ANN.FallbackApplied {
			fmt.Printf("  ann fallback: yes (%s -> %s)\n", comparison.ANN.ANNStatusBefore.Mode, comparison.ANN.ANNStatusAfter.Mode)
		} else {
			fmt.Printf("  ann fallback: no\n")
		}
		fmt.Printf("  ann nil-neighbor skips: %d\n", comparison.ANN.NilNeighborSkips)
		fmt.Printf("  ann search calls: %d, layers: %d, visited: %d, returned: %d\n", comparison.ANN.SearchCalls, comparison.ANN.LayerTraversals, comparison.ANN.VisitedNodes, comparison.ANN.ReturnedNodes)
		fmt.Printf("  candidate overlap: %d\n", comparison.CandidateOverlapCount)
		fmt.Printf("  brute-force top-%d in ANN candidates: %d\n", report.Limit, comparison.BruteForceTopKInANNCands)
		fmt.Printf("  brute-force wide top-%d in ANN candidates: %d\n", report.Limit, comparison.BruteForceWideTopKInANNCands)
		fmt.Printf("  brute-force: %v\n", comparison.BruteForce.Names)
		fmt.Printf("  ann:         %v\n", comparison.ANN.Names)
	}
}

func printLatencyReport(v any) {
	report, ok := v.(*latencyReport)
	if !ok {
		log.Fatalf("unexpected latency report type %T", v)
	}
	fmt.Printf("Preset: %s\n", report.Preset)
	fmt.Printf("Limit: %d\n", report.Limit)
	for _, run := range report.Runs {
		fmt.Printf("\n%s\n", run.Mode)
		fmt.Printf("  ann_status: mode=%s state=%s artifacts=%v\n", run.Status.Mode, run.Status.State, run.Status.ArtifactsPresent)
		if run.OpenAndReady > 0 {
			fmt.Printf("  open_and_ready: %s\n", run.OpenAndReady)
		}
		fmt.Printf("  embed_avg:  %s\n", run.EmbedAvg)
		fmt.Printf("  search_avg: %s\n", run.SearchAvg)
		fmt.Printf("  search_p50: %s\n", run.SearchP50)
		fallbacks := 0
		nilSkips := uint64(0)
		searchCalls := uint64(0)
		visitedNodes := uint64(0)
		returnedNodes := uint64(0)
		layerTraversals := uint64(0)
		for _, sample := range run.Samples {
			if sample.FallbackApplied {
				fallbacks++
			}
			nilSkips += sample.NilNeighborSkips
			searchCalls += sample.SearchCalls
			visitedNodes += sample.VisitedNodes
			returnedNodes += sample.ReturnedNodes
			layerTraversals += sample.LayerTraversals
		}
		fmt.Printf("  fallbacks:  %d\n", fallbacks)
		fmt.Printf("  nil-neighbor skips: %d\n", nilSkips)
		fmt.Printf("  search calls: %d, layers: %d, visited: %d, returned: %d\n", searchCalls, layerTraversals, visitedNodes, returnedNodes)
	}
}
