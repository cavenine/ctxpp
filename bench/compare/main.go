// Package main implements the ctx++ comparison benchmark harness.
//
// It spawns MCP servers (ctx++, codemogger, contextplus) as stdio subprocesses,
// sends index and search tool calls via the MCP protocol, measures latencies,
// collects DB file sizes and process memory (peak RSS), and produces a
// structured JSON report with a human-readable markdown summary.
//
// Usage:
//
//	go run ./bench/compare [flags]
//
//	  -repo string       Path to local repo to benchmark (default: generates synthetic)
//	  -repo-url string   Git URL to clone as benchmark target (e.g. https://github.com/golang/go)
//	  -repo-ref string   Git ref to checkout after clone (default: HEAD)
//	  -repo-sub string   Subdirectory within cloned repo to index (e.g. "src/net/http")
//	  -files int         Number of synthetic files to generate (default: 200)
//	  -queries int       Number of search queries to run (default: 10)
//	  -runs int          Number of search runs per query for statistics (default: 3)
//	  -warmup int        Number of warmup search runs (not recorded) (default: 1)
//	  -out string        Output JSON report path (default: stdout)
//	  -tools string      Comma-separated tool list: ctxpp,codemogger,contextplus (default: ctxpp)
//	  -ctxpp-cmd string  Override ctxpp command (default: auto-detect ./ctxpp or "go run ./cmd/ctxpp")
//	  -seed int          Seed for synthetic repo generation (default: 42)
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// ToolConfig describes how to invoke an MCP server.
type ToolConfig struct {
	Name       string                // display name
	Command    string                // executable
	Args       []string              // command arguments (used when BuildArgs is nil)
	BuildArgs  func(string) []string // optional: builds args from repo path at runtime
	Env        []string              // additional env vars
	IndexTool  string                // MCP tool name for indexing
	IndexArgs  func(string) any      // builds index tool arguments from repo path
	SearchTool string                // MCP tool name for searching
	SearchArgs func(string) any      // builds search tool arguments from query
	// DBPath returns the expected DB file path for a given repo root,
	// or "" if DB size measurement is not applicable.
	DBPath func(string) string
}

// resolveArgs returns the command arguments, using BuildArgs(repoPath) if
// available, otherwise the static Args slice.
func (tc ToolConfig) resolveArgs(repoPath string) []string {
	if tc.BuildArgs != nil {
		return tc.BuildArgs(repoPath)
	}
	return tc.Args
}

// BenchResult captures metrics for one tool.
type BenchResult struct {
	Tool            string         `json:"tool"`
	IndexDuration   time.Duration  `json:"index_duration_ns"`
	IndexDurationS  string         `json:"index_duration"`
	DBSizeBytes     int64          `json:"db_size_bytes"`
	DBSizeHuman     string         `json:"db_size"`
	PeakRSSBytes    int64          `json:"peak_rss_bytes,omitempty"`
	PeakRSSHuman    string         `json:"peak_rss,omitempty"`
	SearchLatencies []SearchResult `json:"search_latencies"`
	SearchP50       time.Duration  `json:"search_p50_ns"`
	SearchP50S      string         `json:"search_p50"`
	SearchP95       time.Duration  `json:"search_p95_ns"`
	SearchP95S      string         `json:"search_p95"`
	SearchP99       time.Duration  `json:"search_p99_ns"`
	SearchP99S      string         `json:"search_p99"`
	Error           string         `json:"error,omitempty"`
}

// SearchResult captures one search query's timing.
type SearchResult struct {
	Query    string        `json:"query"`
	Duration time.Duration `json:"duration_ns"`
	Results  int           `json:"results"`
}

// Report is the full benchmark output.
type Report struct {
	Timestamp    string        `json:"timestamp"`
	RepoPath     string        `json:"repo_path"`
	RepoURL      string        `json:"repo_url,omitempty"`
	RepoRef      string        `json:"repo_ref,omitempty"`
	RepoFiles    int           `json:"repo_files"`
	Synthetic    bool          `json:"synthetic"`
	QueryCount   int           `json:"query_count"`
	RunsPerQuery int           `json:"runs_per_query"`
	WarmupRuns   int           `json:"warmup_runs"`
	GoVersion    string        `json:"go_version"`
	GOOS         string        `json:"goos"`
	GOARCH       string        `json:"goarch"`
	NumCPU       int           `json:"num_cpu"`
	Results      []BenchResult `json:"results"`
}

var defaultQueries = []string{
	"Method0",
	"Type1",
	"Func3",
	"performs operation",
	"generated function",
	"Func0",
	"Method2",
	"Type0",
	"string formatting",
	"data processing",
}

// realRepoQueries are used when benchmarking against a real (non-synthetic) repo.
// They are general enough to return results in most Go codebases.
var realRepoQueries = []string{
	"func",
	"error",
	"handler",
	"context",
	"string",
	"Read",
	"Write",
	"Close",
	"New",
	"Config",
	"Server",
	"Client",
	"Test",
	"request",
	"response",
	"parse",
	"buffer",
	"interface",
	"timeout",
	"connection",
}

func main() {
	var (
		repoPath   = flag.String("repo", "", "path to local repo to benchmark (empty = generate synthetic)")
		repoURL    = flag.String("repo-url", "", "git URL to clone as benchmark target")
		repoRef    = flag.String("repo-ref", "", "git ref to checkout after clone (default: HEAD)")
		repoSub    = flag.String("repo-sub", "", "subdirectory within cloned repo to index")
		numFiles   = flag.Int("files", 200, "number of synthetic files")
		numQueries = flag.Int("queries", 10, "number of search queries")
		numRuns    = flag.Int("runs", 3, "search runs per query")
		warmupRuns = flag.Int("warmup", 1, "warmup search runs (not recorded)")
		outPath    = flag.String("out", "", "output JSON path (empty = stdout)")
		toolsList  = flag.String("tools", "ctxpp", "comma-separated tools")
		ctxppCmd   = flag.String("ctxpp-cmd", "", "override ctxpp command path")
		idxTimeout = flag.Duration("index-timeout", 60*time.Minute, "max time to wait for index")
		seed       = flag.Int64("seed", 42, "seed for synthetic repo generation")
	)
	flag.Parse()

	ctx := context.Background()

	// Determine repo source.
	root := *repoPath
	synthetic := root == "" && *repoURL == ""
	clonedURL := ""

	switch {
	case *repoURL != "":
		// Clone repo.
		var err error
		root, err = cloneRepo(ctx, *repoURL, *repoRef)
		if err != nil {
			log.Fatalf("clone repo: %v", err)
		}
		defer os.RemoveAll(root)
		clonedURL = *repoURL

		if *repoSub != "" {
			sub := filepath.Join(root, *repoSub)
			if info, err := os.Stat(sub); err != nil || !info.IsDir() {
				log.Fatalf("-repo-sub %q does not exist in cloned repo", *repoSub)
			}
			root = sub
		}
		log.Printf("cloned repo %s to %s", *repoURL, root)

	case root != "":
		root, _ = filepath.Abs(root)
		log.Printf("using local repo: %s", root)

	default:
		var err error
		root, err = generateSyntheticRepo(*numFiles, *seed)
		if err != nil {
			log.Fatalf("generate repo: %v", err)
		}
		defer os.RemoveAll(root)
		log.Printf("generated synthetic repo: %s (%d files, seed=%d)", root, *numFiles, *seed)
	}

	// Count .go files.
	fileCount := countGoFiles(root)
	log.Printf("Go files in repo: %d", fileCount)

	// Build tool configs.
	tools := buildToolConfigs(*ctxppCmd)
	selectedTools := strings.Split(*toolsList, ",")

	// Select queries.
	queries := defaultQueries
	if !synthetic {
		queries = realRepoQueries
	}
	if *numQueries < len(queries) {
		queries = queries[:*numQueries]
	}

	goVer := runtime.Version()

	report := Report{
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		RepoPath:     root,
		RepoURL:      clonedURL,
		RepoRef:      *repoRef,
		RepoFiles:    fileCount,
		Synthetic:    synthetic,
		QueryCount:   len(queries),
		RunsPerQuery: *numRuns,
		WarmupRuns:   *warmupRuns,
		GoVersion:    goVer,
		GOOS:         runtime.GOOS,
		GOARCH:       runtime.GOARCH,
		NumCPU:       runtime.NumCPU(),
	}

	for _, name := range selectedTools {
		name = strings.TrimSpace(name)
		tc, ok := tools[name]
		if !ok {
			log.Printf("unknown tool %q, skipping", name)
			continue
		}

		log.Printf("--- benchmarking %s ---", tc.Name)
		result := benchmarkTool(ctx, tc, root, queries, *numRuns, *warmupRuns, *idxTimeout)
		report.Results = append(report.Results, result)
	}

	// Output.
	enc := json.NewEncoder(os.Stdout)
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			log.Fatalf("create output: %v", err)
		}
		defer f.Close()
		enc = json.NewEncoder(f)
	}
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatalf("encode report: %v", err)
	}

	// Print markdown summary to stderr.
	printMarkdownSummary(os.Stderr, report)
}

func buildToolConfigs(ctxppOverride string) map[string]ToolConfig {
	// Find ctxpp binary.
	ctxppBin := ctxppOverride
	if ctxppBin == "" {
		ctxppBin = "./ctxpp"
		if _, err := os.Stat(ctxppBin); err != nil {
			// Try go run instead.
			ctxppBin = "go"
		}
	}

	return map[string]ToolConfig{
		"ctxpp": {
			Name:    "ctx++",
			Command: ctxppBin,
			Args: func() []string {
				if ctxppBin == "go" {
					return []string{"run", "./cmd/ctxpp", "mcp"}
				}
				return []string{"mcp"}
			}(),
			IndexTool: "ctxpp_index",
			IndexArgs: func(path string) any {
				return map[string]any{"path": path}
			},
			SearchTool: "ctxpp_search",
			SearchArgs: func(query string) any {
				return map[string]any{"query": query, "mode": "keyword", "limit": 10}
			},
			DBPath: func(repoRoot string) string {
				return filepath.Join(repoRoot, ".ctxpp", "index.db")
			},
		},
		"codemogger": {
			Name:    "codemogger",
			Command: "npx",
			BuildArgs: func(repoPath string) []string {
				dbPath := filepath.Join(repoPath, ".codemogger", "index.db")
				return []string{"-y", "codemogger", "--db", dbPath, "mcp"}
			},
			IndexTool: "codemogger_index",
			IndexArgs: func(path string) any {
				return map[string]any{"directory": path}
			},
			SearchTool: "codemogger_search",
			SearchArgs: func(query string) any {
				return map[string]any{"query": query, "mode": "keyword", "limit": 10, "includeSnippet": false}
			},
			DBPath: func(repoRoot string) string {
				return filepath.Join(repoRoot, ".codemogger", "index.db")
			},
		},
		"contextplus": {
			Name:    "contextplus",
			Command: "npx",
			BuildArgs: func(repoPath string) []string {
				return []string{"-y", "contextplus", repoPath}
			},
			// contextplus builds embeddings lazily on the first semantic search
			// call, so we use semantic_code_search as the "index" tool to capture
			// the full embedding build time in the index duration metric.
			IndexTool: "semantic_code_search",
			IndexArgs: func(path string) any {
				return map[string]any{"query": "init", "top_k": 1}
			},
			SearchTool: "semantic_code_search",
			SearchArgs: func(query string) any {
				return map[string]any{"query": query, "top_k": 10}
			},
			DBPath: func(repoRoot string) string {
				return filepath.Join(repoRoot, ".mcp_data", "embeddings-cache.json")
			},
		},
	}
}

func benchmarkTool(ctx context.Context, tc ToolConfig, repoPath string, queries []string, runsPerQuery, warmupRuns int, indexTimeout time.Duration) BenchResult {
	result := BenchResult{Tool: tc.Name}

	// Build environment with repo path.
	// Set CTXPP_PROJECT so ctxpp creates its DB inside the repo dir.
	env := append(os.Environ(), tc.Env...)
	env = append(env, "CTXPP_PROJECT="+repoPath)

	args := tc.resolveArgs(repoPath)
	log.Printf("  starting MCP server: %s %v", tc.Command, args)

	// Spawn MCP client with kill-on-timeout semantics.
	c, err := newMCPClient(tc.Command, env, args)
	if err != nil {
		result.Error = fmt.Sprintf("spawn: %v", err)
		log.Printf("  ERROR: %s", result.Error)
		return result
	}
	defer c.Close(10 * time.Second)

	// Initialize MCP session.
	initCtx, initCancel := context.WithTimeout(ctx, 60*time.Second)
	defer initCancel()

	initReq := mcp.InitializeRequest{}
	initReq.Params.ClientInfo = mcp.Implementation{Name: "ctxpp-bench", Version: "0.2.0"}

	_, err = c.Initialize(initCtx, initReq)
	if err != nil {
		result.Error = fmt.Sprintf("initialize: %v", err)
		log.Printf("  ERROR: %s", result.Error)
		return result
	}

	// --- Index phase ---
	log.Printf("  indexing %s...", repoPath)
	indexStart := time.Now()
	indexCtx, indexCancel := context.WithTimeout(ctx, indexTimeout)
	defer indexCancel()

	_, err = c.CallTool(indexCtx, mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      tc.IndexTool,
			Arguments: tc.IndexArgs(repoPath),
		},
	})
	result.IndexDuration = time.Since(indexStart)
	result.IndexDurationS = result.IndexDuration.String()
	if err != nil {
		result.Error = fmt.Sprintf("index: %v", err)
		log.Printf("  ERROR: %s", result.Error)
		return result
	}
	log.Printf("  indexed in %s", result.IndexDuration)

	// --- DB size measurement ---
	if tc.DBPath != nil {
		dbPath := tc.DBPath(repoPath)
		if info, err := os.Stat(dbPath); err == nil {
			result.DBSizeBytes = info.Size()
			result.DBSizeHuman = humanBytes(info.Size())
			log.Printf("  DB size: %s (%s)", result.DBSizeHuman, dbPath)
		} else {
			log.Printf("  DB size: could not stat %s: %v", dbPath, err)
		}
	}

	// --- Peak RSS measurement (subprocess, Linux only) ---
	result.PeakRSSBytes = c.readSubprocessPeakRSS()
	if result.PeakRSSBytes == 0 {
		// Fall back to harness process peak RSS.
		result.PeakRSSBytes = readPeakRSS()
	}
	if result.PeakRSSBytes > 0 {
		result.PeakRSSHuman = humanBytes(result.PeakRSSBytes)
		log.Printf("  peak RSS: %s", result.PeakRSSHuman)
	}

	// --- Warmup search runs (not recorded) ---
	if warmupRuns > 0 {
		log.Printf("  running %d warmup queries...", warmupRuns)
		for w := 0; w < warmupRuns; w++ {
			q := queries[w%len(queries)]
			warmCtx, warmCancel := context.WithTimeout(ctx, 120*time.Second)
			_, _ = c.CallTool(warmCtx, mcp.CallToolRequest{
				Params: mcp.CallToolParams{
					Name:      tc.SearchTool,
					Arguments: tc.SearchArgs(q),
				},
			})
			warmCancel()
		}
	}

	// --- Search phase ---
	log.Printf("  running %d queries x %d runs...", len(queries), runsPerQuery)
	var allLatencies []time.Duration

	for _, q := range queries {
		for run := 0; run < runsPerQuery; run++ {
			searchCtx, searchCancel := context.WithTimeout(ctx, 120*time.Second)

			start := time.Now()
			searchResult, err := c.CallTool(searchCtx, mcp.CallToolRequest{
				Params: mcp.CallToolParams{
					Name:      tc.SearchTool,
					Arguments: tc.SearchArgs(q),
				},
			})
			elapsed := time.Since(start)
			searchCancel()

			nResults := 0
			if err == nil && searchResult != nil {
				nResults = len(searchResult.Content)
			}

			sr := SearchResult{
				Query:    q,
				Duration: elapsed,
				Results:  nResults,
			}
			result.SearchLatencies = append(result.SearchLatencies, sr)
			allLatencies = append(allLatencies, elapsed)

			if err != nil {
				log.Printf("    query %q run %d: ERROR %v", q, run+1, err)
			}
		}
	}

	// Compute percentiles.
	if len(allLatencies) > 0 {
		sort.Slice(allLatencies, func(i, j int) bool { return allLatencies[i] < allLatencies[j] })
		result.SearchP50 = percentile(allLatencies, 50)
		result.SearchP50S = result.SearchP50.String()
		result.SearchP95 = percentile(allLatencies, 95)
		result.SearchP95S = result.SearchP95.String()
		result.SearchP99 = percentile(allLatencies, 99)
		result.SearchP99S = result.SearchP99.String()
	}

	log.Printf("  search p50=%s p95=%s p99=%s", result.SearchP50S, result.SearchP95S, result.SearchP99S)
	return result
}

func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(len(sorted))*p/100)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// ---- Git clone support -----------------------------------------------------

// cloneRepo clones a git repository to a temporary directory.
// If ref is non-empty, it checks out that ref after cloning.
func cloneRepo(ctx context.Context, url, ref string) (string, error) {
	root, err := os.MkdirTemp("", "ctxpp-bench-clone-*")
	if err != nil {
		return "", fmt.Errorf("create temp dir: %w", err)
	}

	cloneCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	args := []string{"clone", "--depth=1"}
	if ref != "" {
		args = append(args, "--branch", ref)
	}
	args = append(args, url, root)

	cmd := exec.CommandContext(cloneCtx, "git", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	log.Printf("  cloning: git %s", strings.Join(args, " "))
	if err := cmd.Run(); err != nil {
		os.RemoveAll(root)
		return "", fmt.Errorf("git clone %s: %w", url, err)
	}

	return root, nil
}

// ---- Synthetic repo generator ----------------------------------------------

func generateSyntheticRepo(numFiles int, seed int64) (string, error) {
	root, err := os.MkdirTemp("", "ctxpp-bench-*")
	if err != nil {
		return "", err
	}

	rng := rand.New(rand.NewSource(seed))

	for f := 0; f < numFiles; f++ {
		pkgName := fmt.Sprintf("pkg%d", f)
		dir := filepath.Join(root, pkgName)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}

		var code strings.Builder
		fmt.Fprintf(&code, "package %s\n\n", pkgName)
		code.WriteString("import (\n\t\"fmt\"\n\t\"strings\"\n)\n\n")

		// Vary counts slightly to create diversity.
		numTypes := 2 + rng.Intn(3)   // 2-4 types
		numFuncs := 3 + rng.Intn(5)   // 3-7 functions
		numMethods := 2 + rng.Intn(3) // 2-4 methods per type

		for t := 0; t < numTypes; t++ {
			typeName := fmt.Sprintf("Type%d", t)
			fmt.Fprintf(&code, "// %s is a generated type for benchmarking.\ntype %s struct {\n", typeName, typeName)
			code.WriteString("\tName string\n\tValue int\n\tData []byte\n}\n\n")

			for m := 0; m < numMethods; m++ {
				methName := fmt.Sprintf("Method%d", m)
				fmt.Fprintf(&code, "// %s performs operation %d on %s.\n", methName, m, typeName)
				fmt.Fprintf(&code, "func (t *%s) %s(input string) string {\n", typeName, methName)
				code.WriteString("\tresult := fmt.Sprintf(\"%s-%d\", input, t.Value)\n")
				code.WriteString("\treturn strings.TrimSpace(result)\n}\n\n")
			}
		}

		for fn := 0; fn < numFuncs; fn++ {
			funcName := fmt.Sprintf("Func%d", fn)
			fmt.Fprintf(&code, "// %s is a generated function for benchmarking.\n", funcName)
			fmt.Fprintf(&code, "func %s(a, b string) string {\n", funcName)
			code.WriteString("\treturn fmt.Sprintf(\"%s+%s\", a, b)\n}\n\n")
		}

		path := filepath.Join(dir, fmt.Sprintf("%s.go", pkgName))
		if err := os.WriteFile(path, []byte(code.String()), 0o644); err != nil {
			return "", err
		}
	}
	return root, nil
}

func countGoFiles(root string) int {
	count := 0
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".go") {
			count++
		}
		return nil
	})
	return count
}

// ---- Memory measurement ----------------------------------------------------

// readPeakRSS reads VmHWM (peak resident set size) from /proc/self/status.
// This measures the benchmark harness process itself, which includes memory
// used for JSON serialization and MCP protocol handling. The MCP server runs
// as a subprocess, so its memory is not directly captured here — a future
// version could use a custom subprocess manager to read /proc/<pid>/status.
// Only works on Linux; returns 0 otherwise.
func readPeakRSS() int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "VmHWM:") {
			// Format: "VmHWM:    12345 kB"
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, err := strconv.ParseInt(fields[1], 10, 64)
				if err == nil {
					return kb * 1024
				}
			}
		}
	}
	return 0
}

// ---- Subprocess-managed MCP client -----------------------------------------

// mcpClient wraps an mcp-go Client with access to the underlying subprocess,
// enabling graceful shutdown with a hard kill timeout. This prevents the
// harness from hanging indefinitely when the MCP server subprocess does not
// exit cleanly after stdin is closed.
type mcpClient struct {
	*client.Client
	mu     sync.Mutex
	cmd    *exec.Cmd // captured from CommandFunc
	closed bool
}

// newMCPClient spawns an MCP server subprocess and returns a client with
// kill-on-timeout Close semantics.
func newMCPClient(command string, env []string, args []string) (*mcpClient, error) {
	mc := &mcpClient{}

	// Use WithCommandFunc to capture the *exec.Cmd so we can kill it later.
	cmdFunc := func(ctx context.Context, cmd string, env []string, args []string) (*exec.Cmd, error) {
		c := exec.CommandContext(ctx, cmd, args...)
		c.Env = env
		// Set process group so we can kill the entire group.
		c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		mc.mu.Lock()
		mc.cmd = c
		mc.mu.Unlock()
		return c, nil
	}

	c, err := client.NewStdioMCPClientWithOptions(
		command, env, args,
		transport.WithCommandFunc(cmdFunc),
	)
	if err != nil {
		return nil, err
	}
	mc.Client = c
	return mc, nil
}

// Close attempts graceful shutdown of the MCP client and subprocess.
// If the subprocess does not exit within the timeout, it sends SIGKILL.
func (mc *mcpClient) Close(timeout time.Duration) {
	mc.mu.Lock()
	if mc.closed {
		mc.mu.Unlock()
		return
	}
	mc.closed = true
	cmd := mc.cmd
	mc.mu.Unlock()

	// Try graceful close in a goroutine.
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = mc.Client.Close()
	}()

	select {
	case <-done:
		// Graceful close succeeded.
	case <-time.After(timeout):
		log.Printf("  WARN: MCP client Close() timed out after %s, killing subprocess", timeout)
		if cmd != nil && cmd.Process != nil {
			// Kill the entire process group to catch any children.
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			// Also kill the process directly as a fallback.
			_ = cmd.Process.Kill()
		}
		// Wait for the goroutine to finish (cmd.Wait will now return).
		<-done
	}
}

// readSubprocessPeakRSS reads VmHWM from /proc/<pid>/status for the MCP
// server subprocess. Only works on Linux; returns 0 otherwise.
func (mc *mcpClient) readSubprocessPeakRSS() int64 {
	mc.mu.Lock()
	cmd := mc.cmd
	mc.mu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return 0
	}

	path := fmt.Sprintf("/proc/%d/status", cmd.Process.Pid)
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "VmHWM:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, err := strconv.ParseInt(fields[1], 10, 64)
				if err == nil {
					return kb * 1024
				}
			}
		}
	}
	return 0
}

// ---- Formatting helpers ----------------------------------------------------

func humanBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMG"[exp])
}

// ---- Markdown report -------------------------------------------------------

func printMarkdownSummary(w *os.File, r Report) {
	fmt.Fprintf(w, "\n## Comparison Benchmark Results\n\n")
	fmt.Fprintf(w, "- **Timestamp**: %s\n", r.Timestamp)
	if r.RepoURL != "" {
		fmt.Fprintf(w, "- **Repo**: %s (cloned", r.RepoURL)
		if r.RepoRef != "" {
			fmt.Fprintf(w, ", ref=%s", r.RepoRef)
		}
		fmt.Fprintf(w, ")\n")
	} else {
		fmt.Fprintf(w, "- **Repo**: %s (synthetic=%v)\n", r.RepoPath, r.Synthetic)
	}
	fmt.Fprintf(w, "- **Go files**: %d\n", r.RepoFiles)
	fmt.Fprintf(w, "- **Queries**: %d x %d runs (%d warmup)\n", r.QueryCount, r.RunsPerQuery, r.WarmupRuns)
	fmt.Fprintf(w, "- **System**: %s/%s, %d CPUs, %s\n\n", r.GOOS, r.GOARCH, r.NumCPU, r.GoVersion)

	fmt.Fprintf(w, "| Tool | Index Time | DB Size | Peak RSS | Search p50 | Search p95 | Search p99 | Error |\n")
	fmt.Fprintf(w, "|------|-----------|---------|----------|-----------|-----------|-----------|-------|\n")

	for _, res := range r.Results {
		errStr := ""
		if res.Error != "" {
			errStr = res.Error
			if len(errStr) > 40 {
				errStr = errStr[:40] + "..."
			}
		}
		dbSize := "-"
		if res.DBSizeBytes > 0 {
			dbSize = res.DBSizeHuman
		}
		peakRSS := "-"
		if res.PeakRSSBytes > 0 {
			peakRSS = res.PeakRSSHuman
		}
		fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s | %s | %s |\n",
			res.Tool,
			res.IndexDurationS,
			dbSize,
			peakRSS,
			res.SearchP50S,
			res.SearchP95S,
			res.SearchP99S,
			errStr,
		)
	}
	fmt.Fprintln(w)
}
