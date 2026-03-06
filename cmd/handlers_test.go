package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/parser"
	"github.com/cavenine/ctxpp/internal/store"
)

// testApp creates an app with a temporary store, indexes the given fixture
// directory, and returns the app ready for handler testing.
func testApp(t *testing.T, fixtureDir string) *app {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("store.Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	emb := embed.NewBundledEmbedder(768)
	parsers := []parser.Parser{parser.NewGoParser()}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	idx := indexer.New(indexer.Config{
		ProjectRoot: fixtureDir,
		Logger:      logger,
	}, st, parsers, emb)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	return &app{
		store:         st,
		indexer:       idx,
		indexEmbedder: emb,
		queryEmbedder: emb,
		root:          fixtureDir,
	}
}

// setupFixture creates a temporary directory with Go source files for testing.
func setupFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()

	// svc/svc.go — a service with functions and a type with methods.
	svcDir := filepath.Join(root, "svc")
	if err := os.MkdirAll(svcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	svcCode := `package svc

import "fmt"

// UserService manages users.
type UserService struct {
	Name string
}

// CreateUser creates a new user.
func (s *UserService) CreateUser(name string) string {
	return fmt.Sprintf("created:%s", name)
}

// DeleteUser deletes a user by name.
func (s *UserService) DeleteUser(name string) {
	fmt.Println("deleted", name)
}

// FetchAccount retrieves an account.
func FetchAccount(id int) string {
	return fmt.Sprintf("account:%d", id)
}
`
	if err := os.WriteFile(filepath.Join(svcDir, "svc.go"), []byte(svcCode), 0o644); err != nil {
		t.Fatal(err)
	}

	// handler/handler.go — a handler that calls svc functions.
	handlerDir := filepath.Join(root, "handler")
	if err := os.MkdirAll(handlerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	handlerCode := `package handler

import "fmt"

// HandleLogin processes a login request.
func HandleLogin(username string) string {
	return fmt.Sprintf("login:%s", username)
}

// HandleLogout processes a logout request.
func HandleLogout(username string) {
	fmt.Println("logout", username)
}
`
	if err := os.WriteFile(filepath.Join(handlerDir, "handler.go"), []byte(handlerCode), 0o644); err != nil {
		t.Fatal(err)
	}

	return root
}

// setupFixtureWithCallGraph creates a fixture with a clear call chain:
// Orchestrate -> DoStep -> Finalize
// This enables testing feature traverse (BFS) and blast radius (reverse edges).
func setupFixtureWithCallGraph(t *testing.T) string {
	t.Helper()
	root := t.TempDir()

	code := `package pipeline

import "fmt"

// Finalize completes the pipeline.
func Finalize() {
	fmt.Println("done")
}

// DoStep performs one step, then finalizes.
func DoStep() {
	fmt.Println("step")
	Finalize()
}

// Orchestrate runs the full pipeline.
func Orchestrate() {
	fmt.Println("start")
	DoStep()
}
`
	if err := os.WriteFile(filepath.Join(root, "pipeline.go"), []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// makeToolRequest builds a CallToolRequest with the given arguments.
func makeToolRequest(args map[string]any) mcp.CallToolRequest {
	return mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Arguments: args,
		},
	}
}

// getResultText extracts the text content from a CallToolResult.
func getResultText(t *testing.T, result *mcp.CallToolResult) string {
	t.Helper()
	if len(result.Content) == 0 {
		t.Fatal("result has no content")
	}
	tc, ok := result.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("result content is %T, want TextContent", result.Content[0])
	}
	return tc.Text
}

// ---- handleIndex tests -----------------------------------------------------

func TestHandleIndex_IndexesProject(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleIndex(context.Background(), makeToolRequest(nil))
	if err != nil {
		t.Fatalf("handleIndex() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "indexed project") {
		t.Errorf("handleIndex() result = %q, want 'indexed project' in output", text)
	}
	// Re-index should report skipped files.
	if !strings.Contains(text, "skipped:") {
		t.Errorf("handleIndex() result = %q, want 'skipped:' in output", text)
	}
}

func TestHandleIndex_WithCustomPath(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	// Index again with explicit path.
	result, err := a.handleIndex(context.Background(), makeToolRequest(map[string]any{
		"path": root,
	}))
	if err != nil {
		t.Fatalf("handleIndex() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, root) {
		t.Errorf("handleIndex() result = %q, want root path %q in output", text, root)
	}
}

// ---- handleSearch tests ----------------------------------------------------

func TestHandleSearch_KeywordFindsSymbol(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "FetchAccount",
		"mode":  "keyword",
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "FetchAccount") {
		t.Errorf("handleSearch() keyword result = %q, want 'FetchAccount'", text)
	}
}

func TestHandleSearch_HybridDefault(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	// No mode specified — defaults to hybrid.
	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "UserService",
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "UserService") {
		t.Errorf("handleSearch() hybrid result = %q, want 'UserService'", text)
	}
}

func TestHandleSearch_HybridFallsBackToKeywordOnEmbedError(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)
	a.queryEmbedder = &failingEmbedder{Embedder: a.queryEmbedder, err: errors.New("embed unavailable")}

	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "UserService",
		"mode":  "hybrid",
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "UserService") {
		t.Errorf("handleSearch() hybrid fallback result = %q, want 'UserService' from keyword fallback", text)
	}
}

func TestHandleSearch_SemanticMode(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	// Semantic mode with bundled (stub) embedder: no embeddings are stored
	// so the result is an empty list. Verify it returns valid empty JSON.
	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "user management",
		"mode":  "semantic",
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	// With the stub embedder, no embeddings exist so result is "[]".
	if text != "[]" && !strings.Contains(text, "\"name\"") {
		t.Errorf("handleSearch() semantic result = %q, want empty JSON array or symbol list", text)
	}
}

func TestHandleSearch_WithLimit(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "HandleLogin",
		"mode":  "keyword",
		"limit": float64(1), // JSON numbers are float64
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)

	var syms []symbolJSON
	if err := json.Unmarshal([]byte(text), &syms); err != nil {
		t.Fatalf("failed to parse result JSON: %v", err)
	}
	if len(syms) != 1 {
		t.Errorf("handleSearch() with limit=1 returned %d results, want 1", len(syms))
	}
}

// ---- handleFileSkeleton tests ----------------------------------------------

func TestHandleFileSkeleton_ReturnsSymbols(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleFileSkeleton(context.Background(), makeToolRequest(map[string]any{
		"path": "svc/svc.go",
	}))
	if err != nil {
		t.Fatalf("handleFileSkeleton() error = %v", err)
	}
	text := getResultText(t, result)

	// Should contain all symbols from svc.go.
	for _, want := range []string{"UserService", "CreateUser", "DeleteUser", "FetchAccount"} {
		if !strings.Contains(text, want) {
			t.Errorf("handleFileSkeleton() result missing %q", want)
		}
	}
}

func TestHandleFileSkeleton_MissingPath(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleFileSkeleton(context.Background(), makeToolRequest(map[string]any{}))
	if err != nil {
		t.Fatalf("handleFileSkeleton() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "path is required") {
		t.Errorf("handleFileSkeleton() missing path result = %q, want 'path is required'", text)
	}
}

func TestHandleFileSkeleton_NonexistentFile(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleFileSkeleton(context.Background(), makeToolRequest(map[string]any{
		"path": "nonexistent.go",
	}))
	if err != nil {
		t.Fatalf("handleFileSkeleton() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "no symbols found") {
		t.Errorf("handleFileSkeleton() nonexistent result = %q, want 'no symbols found'", text)
	}
}

// ---- handleFeatureTraverse tests -------------------------------------------

func TestHandleFeatureTraverse_FindsSymbol(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "HandleLogin",
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "HandleLogin") {
		t.Errorf("handleFeatureTraverse() result = %q, want 'HandleLogin'", text)
	}
}

func TestHandleFeatureTraverse_NoMatch(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "NonexistentSymbol",
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "no symbols found") && !strings.Contains(text, "no exact match") {
		t.Errorf("handleFeatureTraverse() no-match result = %q, want error message", text)
	}
}

func TestHandleFeatureTraverse_WalksCallGraph(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "Orchestrate",
		"depth": float64(2),
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)

	// Orchestrate calls DoStep which calls Finalize.
	if !strings.Contains(text, "Orchestrate") {
		t.Error("missing seed symbol Orchestrate")
	}
	if !strings.Contains(text, "DoStep") {
		t.Error("missing callee DoStep (depth 1)")
	}
	if !strings.Contains(text, "Finalize") {
		t.Error("missing callee Finalize (depth 2)")
	}
}

func TestHandleFeatureTraverse_CustomDepth(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	// Depth 1: should find Orchestrate and DoStep, but NOT Finalize.
	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "Orchestrate",
		"depth": float64(1),
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "Orchestrate") {
		t.Error("missing seed symbol Orchestrate")
	}
	if !strings.Contains(text, "DoStep") {
		t.Error("missing callee DoStep at depth 1")
	}
	// Finalize is at depth 2 and should NOT appear with depth=1.
	if strings.Contains(text, "Finalize") {
		t.Error("Finalize should not appear at depth=1")
	}
}

// ---- handleBlastRadius tests -----------------------------------------------

func TestHandleBlastRadius_ReturnsCallers(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	result, err := a.handleBlastRadius(context.Background(), makeToolRequest(map[string]any{
		"symbol": "DoStep",
	}))
	if err != nil {
		t.Fatalf("handleBlastRadius() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "DoStep") {
		t.Errorf("handleBlastRadius() result = %q, want 'DoStep'", text)
	}
	// Orchestrate calls DoStep, so it should appear as a caller.
	if !strings.Contains(text, "Orchestrate") {
		t.Error("handleBlastRadius() missing caller Orchestrate")
	}
}

func TestHandleBlastRadius_MissingSymbol(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleBlastRadius(context.Background(), makeToolRequest(map[string]any{}))
	if err != nil {
		t.Fatalf("handleBlastRadius() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "symbol is required") {
		t.Errorf("handleBlastRadius() missing symbol result = %q, want 'symbol is required'", text)
	}
}

func TestHandleBlastRadius_NoCallers(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	// Orchestrate is the top-level caller; nothing calls it.
	result, err := a.handleBlastRadius(context.Background(), makeToolRequest(map[string]any{
		"symbol": "Orchestrate",
	}))
	if err != nil {
		t.Fatalf("handleBlastRadius() error = %v", err)
	}
	text := getResultText(t, result)

	// Should have the symbol field but empty callers.
	var out struct {
		Symbol  string `json:"symbol"`
		Callers []any  `json:"callers"`
	}
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatalf("failed to parse blast radius JSON: %v", err)
	}
	if out.Symbol != "Orchestrate" {
		t.Errorf("symbol = %q, want %q", out.Symbol, "Orchestrate")
	}
	if len(out.Callers) != 0 {
		t.Errorf("expected 0 callers for top-level Orchestrate, got %d", len(out.Callers))
	}
}

// ---- handleSearch additional edge cases ------------------------------------

func TestHandleSearch_EmptyModeDefaultsToHybrid(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	// Explicitly pass empty mode string — should default to hybrid.
	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "HandleLogin",
		"mode":  "",
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "HandleLogin") {
		t.Errorf("handleSearch() empty mode result = %q, want 'HandleLogin'", text)
	}
}

func TestHandleSearch_NegativeLimitUsesDefault(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	// Negative limit should be treated as default (10).
	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "FetchAccount",
		"mode":  "keyword",
		"limit": float64(-5),
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "FetchAccount") {
		t.Errorf("handleSearch() negative limit result = %q, want 'FetchAccount'", text)
	}
}

func TestHandleSearch_ZeroLimitUsesDefault(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	result, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "DeleteUser",
		"mode":  "keyword",
		"limit": float64(0),
	}))
	if err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "DeleteUser") {
		t.Errorf("handleSearch() zero limit result = %q, want 'DeleteUser'", text)
	}
}

// ---- handleFeatureTraverse edge cases --------------------------------------

func TestHandleFeatureTraverse_DefaultDepth(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	// No depth specified — defaults to 3, which should traverse the whole chain.
	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "Orchestrate",
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)

	for _, want := range []string{"Orchestrate", "DoStep", "Finalize"} {
		if !strings.Contains(text, want) {
			t.Errorf("handleFeatureTraverse() default depth result missing %q", want)
		}
	}
}

func TestHandleFeatureTraverse_NegativeDepthUsesDefault(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	// Negative depth should be treated as default (3).
	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "Orchestrate",
		"depth": float64(-1),
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "Orchestrate") {
		t.Error("missing seed symbol Orchestrate with negative depth")
	}
}

func TestHandleFeatureTraverse_LeafNode(t *testing.T) {
	root := setupFixtureWithCallGraph(t)
	a := testApp(t, root)

	// Finalize is a leaf — no callees. Should return just itself.
	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "Finalize",
		"depth": float64(3),
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "Finalize") {
		t.Error("missing Finalize in leaf traversal")
	}

	var syms []symbolJSON
	if err := json.Unmarshal([]byte(text), &syms); err != nil {
		t.Fatalf("failed to parse result JSON: %v", err)
	}
	// Should only have Finalize itself (leaf node, no callees).
	if len(syms) != 1 {
		t.Errorf("leaf traversal returned %d symbols, want 1", len(syms))
	}
}

func TestHandleFeatureTraverse_PartialNameNoExactMatch(t *testing.T) {
	root := setupFixture(t)
	a := testApp(t, root)

	// "Handle" matches HandleLogin and HandleLogout via FTS, but there is no
	// symbol with the exact Name "Handle". Should return "no exact match".
	result, err := a.handleFeatureTraverse(context.Background(), makeToolRequest(map[string]any{
		"query": "Handle",
	}))
	if err != nil {
		t.Fatalf("handleFeatureTraverse() error = %v", err)
	}
	text := getResultText(t, result)
	if !strings.Contains(text, "no exact match") && !strings.Contains(text, "no symbols found") {
		t.Errorf("handleFeatureTraverse() partial match result = %q, want error message", text)
	}
}

// ---- embedder split tests --------------------------------------------------

// trackingEmbedder records every Embed call so we can assert which embedder
// is used by which code path.
type trackingEmbedder struct {
	embed.Embedder
	calls atomic.Int64
}

type failingEmbedder struct {
	embed.Embedder
	err error
}

func (f *failingEmbedder) Embed(context.Context, string) ([]float32, error) {
	return nil, f.err
}

func (te *trackingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	te.calls.Add(1)
	return te.Embedder.Embed(ctx, text)
}

func (te *trackingEmbedder) callCount() int {
	return int(te.calls.Load())
}

// TestHandleSearch_UsesQueryEmbedder verifies that handleSearch calls the
// queryEmbedder, not the indexEmbedder, so the cache is never polluted by
// large indexing batches.
func TestHandleSearch_UsesQueryEmbedder(t *testing.T) {
	root := setupFixture(t)

	base := embed.NewBundledEmbedder(768)
	indexEmb := &trackingEmbedder{Embedder: base}
	queryEmb := &trackingEmbedder{Embedder: base}

	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("store.Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	parsers := []parser.Parser{parser.NewGoParser()}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	idx := indexer.New(indexer.Config{ProjectRoot: root, Logger: logger}, st, parsers, indexEmb)
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	a := &app{
		store:         st,
		indexer:       idx,
		indexEmbedder: indexEmb,
		queryEmbedder: queryEmb,
		root:          root,
	}

	if _, err := a.handleSearch(context.Background(), makeToolRequest(map[string]any{
		"query": "UserService",
		"mode":  "semantic",
	})); err != nil {
		t.Fatalf("handleSearch() error = %v", err)
	}

	if queryEmb.callCount() == 0 {
		t.Error("queryEmbedder was not called during handleSearch")
	}
	if indexEmb.callCount() != 0 {
		t.Errorf("indexEmbedder was called %d times during handleSearch, want 0", indexEmb.callCount())
	}
}

// TestHandleIndex_UsesIndexEmbedder verifies that handleIndex calls the
// indexEmbedder, not the queryEmbedder, so query cache entries are not evicted
// by large indexing batches.
func TestHandleIndex_UsesIndexEmbedder(t *testing.T) {
	root := setupFixture(t)

	base := embed.NewBundledEmbedder(768)
	indexEmb := &trackingEmbedder{Embedder: base}
	queryEmb := &trackingEmbedder{Embedder: base}

	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("store.Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	parsers := []parser.Parser{parser.NewGoParser()}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	idx := indexer.New(indexer.Config{ProjectRoot: root, Logger: logger}, st, parsers, indexEmb)

	a := &app{
		store:         st,
		indexer:       idx,
		indexEmbedder: indexEmb,
		queryEmbedder: queryEmb,
		root:          root,
	}

	queryCallsBefore := queryEmb.callCount()
	if _, err := a.handleIndex(context.Background(), makeToolRequest(nil)); err != nil {
		t.Fatalf("handleIndex() error = %v", err)
	}

	if queryEmb.callCount() != queryCallsBefore {
		t.Errorf("queryEmbedder was called %d times during handleIndex, want 0", queryEmb.callCount()-queryCallsBefore)
	}
}
