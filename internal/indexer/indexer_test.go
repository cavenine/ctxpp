package indexer

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/parser"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/cavenine/ctxpp/internal/types"
)

// discardLogger returns a logger that discards all output.
// Used in tests and benchmarks to avoid polluting output.
func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func openTestStore(t *testing.T) *store.Store {
	t.Helper()
	st, err := store.Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("store.Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

func newTestIndexer(t *testing.T, root string, st *store.Store) *Indexer {
	t.Helper()
	return New(
		Config{ProjectRoot: root, Logger: discardLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(3),
	)
}

// fakeEmbedder is a test embedder that produces deterministic non-zero vectors.
// Unlike BundledEmbedder, its Model() is not "bundled-zero" so the indexer
// will not skip the embed pipeline.
type fakeEmbedder struct {
	dims int
}

func (e *fakeEmbedder) Model() string { return "fake-test" }
func (e *fakeEmbedder) Dims() int     { return e.dims }
func (e *fakeEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	v := make([]float32, e.dims)
	for i := range v {
		v[i] = 0.1
	}
	return v, nil
}

type flakyBatchEmbedder struct {
	dims int
	mu   sync.Mutex

	batchCalls int
	embedCalls int
}

func (e *flakyBatchEmbedder) Model() string { return "flaky-batch" }
func (e *flakyBatchEmbedder) Dims() int     { return e.dims }

func (e *flakyBatchEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	e.mu.Lock()
	e.embedCalls++
	e.mu.Unlock()
	v := make([]float32, e.dims)
	for i := range v {
		v[i] = 0.2
	}
	return v, nil
}

func (e *flakyBatchEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	e.mu.Lock()
	e.batchCalls++
	call := e.batchCalls
	e.mu.Unlock()

	// First batch call fails completely to exercise indexer self-heal path.
	if call == 1 {
		return nil, errors.New("simulated batch failure")
	}

	vecs := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, e.dims)
		for j := range v {
			v[j] = 0.2
		}
		vecs[i] = v
	}
	return vecs, nil
}

func (e *flakyBatchEmbedder) counts() (batchCalls, embedCalls int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.batchCalls, e.embedCalls
}

func newTestIndexerWithEmbed(t *testing.T, root string, st *store.Store) *Indexer {
	t.Helper()
	return New(
		Config{ProjectRoot: root, Logger: discardLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		&fakeEmbedder{dims: 3},
	)
}

func newTestIndexerWithParsers(t *testing.T, root string, st *store.Store, parsers ...parser.Parser) *Indexer {
	t.Helper()
	return New(
		Config{ProjectRoot: root, Logger: discardLogger()},
		st,
		parsers,
		embed.NewBundledEmbedder(3),
	)
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) error = %v", name, err)
	}
	return path
}

func findSymbolByName(syms []types.Symbol, name string) *types.Symbol {
	for i := range syms {
		if syms[i].Name == name {
			return &syms[i]
		}
	}
	return nil
}

func TestIndexer_SkipsHiddenDirectories(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	writeFile(t, filepath.Join(root, ".git"), "hook.go", `package git

func Hook() {}
`)
	writeFile(t, root, "main.go", `package main

func Main() {}
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	sha, err := st.GetFileSHA(".git/hook.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if sha != "" {
		t.Error(".git/hook.go was indexed but hidden directories must be skipped")
	}
}

func TestIndexer_RespectsGitignore(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, ".gitignore", "generated.go\n")
	writeFile(t, root, "generated.go", `package main

func Generated() {}
`)
	writeFile(t, root, "real.go", `package main

func Real() {}
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	ignored, err := st.GetFileSHA("generated.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if ignored != "" {
		t.Error("generated.go was indexed but should have been ignored via .gitignore")
	}

	indexed, err := st.GetFileSHA("real.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if indexed == "" {
		t.Error("real.go was not indexed but should have been")
	}
}

func TestIndexer_IndexesKotlinFiles(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexerWithParsers(t, root, st, parser.NewKotlinParser())

	writeFile(t, root, "widget.kt", `package demo

class Widget {
	fun render() {}
}
`)

	if _, err := idx.Index(t.Context()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	sha, err := st.GetFileSHA("widget.kt")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if sha == "" {
		t.Fatal("widget.kt was not indexed")
	}

	syms, err := st.GetSymbolsByFile("widget.kt")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if findSymbolByName(syms, "Widget") == nil {
		t.Fatal("indexed Kotlin symbols missing Widget")
	}
}

func TestIndexer_IndexesCSharpFiles(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexerWithParsers(t, root, st, parser.NewCSharpParser())

	writeFile(t, root, "widget.cs", `namespace Demo.App;

public class Widget {
	public void Render() {}
}
`)

	if _, err := idx.Index(t.Context()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	sha, err := st.GetFileSHA("widget.cs")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if sha == "" {
		t.Fatal("widget.cs was not indexed")
	}

	syms, err := st.GetSymbolsByFile("widget.cs")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if findSymbolByName(syms, "Widget") == nil {
		t.Fatal("indexed C# symbols missing Widget")
	}
}

func TestIndexer_WatchFileInfoRecognizesKotlinAndCSharp(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexerWithParsers(t, root, st, parser.NewKotlinParser(), parser.NewCSharpParser())

	tests := []struct {
		name string
		file string
	}{
		{name: "kotlin", file: "widget.kt"},
		{name: "csharp", file: "widget.cs"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, root, tc.file, "")
			rel, ext, ok, err := idx.watchFileInfo(path, nil)
			if err != nil {
				t.Fatalf("watchFileInfo() error = %v", err)
			}
			if !ok {
				t.Fatal("watchFileInfo() ok = false, want true")
			}
			if rel != tc.file {
				t.Errorf("rel = %q, want %q", rel, tc.file)
			}
			if ext != filepath.Ext(tc.file) {
				t.Errorf("ext = %q, want %q", ext, filepath.Ext(tc.file))
			}
		})
	}
}

func TestIndexer_IgnoresNonGoFiles(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "README.md", "# hello")
	writeFile(t, root, "data.json", `{"key":"value"}`)
	writeFile(t, root, "main.go", `package main

func Main() {}
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	// Only main.go should have been indexed.
	for _, nonGo := range []string{"README.md", "data.json"} {
		sha, err := st.GetFileSHA(nonGo)
		if err != nil {
			t.Fatalf("GetFileSHA(%q) error = %v", nonGo, err)
		}
		if sha != "" {
			t.Errorf("non-Go file %q was indexed, want ignored", nonGo)
		}
	}
}

func TestIndexer_ReindexesChangedFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	path := writeFile(t, root, "svc.go", `package svc

func Old() {}
`)
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("first Index() error = %v", err)
	}

	// Overwrite with new content — SHA will differ.
	if err := os.WriteFile(path, []byte(`package svc

func New() {}
`), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("second Index() error = %v", err)
	}

	syms, err := st.GetSymbolsByFile("svc.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}

	found := make(map[string]bool)
	for _, s := range syms {
		found[s.Name] = true
	}
	if found["Old"] {
		t.Error("stale symbol Old still present after reindex")
	}
	if !found["New"] {
		t.Error("symbol New not found after reindex")
	}
}

func TestIndexer_StoresSymbolsOnFirstIndex(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "lib.go", `package lib

func Add(a, b int) int { return a + b }
func Sub(a, b int) int { return a - b }
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	syms, err := st.GetSymbolsByFile("lib.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}

	found := make(map[string]bool)
	for _, s := range syms {
		found[s.Name] = true
	}

	for _, want := range []string{"Add", "Sub"} {
		if !found[want] {
			t.Errorf("symbol %q not found after index; got %v", want, syms)
		}
	}
}

func TestIndexer_IndexReturnsStats(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "a.go", `package a

func Foo() {}
func Bar() {}
`)
	writeFile(t, root, "b.go", `package b

func Baz() {}
`)

	stats, err := idx.Index(context.Background())
	if err != nil {
		t.Fatalf("Index() error = %v", err)
	}
	if stats.FilesIndexed != 2 {
		t.Errorf("FilesIndexed = %d, want 2", stats.FilesIndexed)
	}
	if stats.SymbolsIndexed != 3 {
		t.Errorf("SymbolsIndexed = %d, want 3", stats.SymbolsIndexed)
	}
	if stats.FilesSkipped != 0 {
		t.Errorf("FilesSkipped = %d, want 0", stats.FilesSkipped)
	}
	if stats.Duration <= 0 {
		t.Error("Duration must be > 0")
	}
}

func TestIndexer_IndexStatsSkipsCounted(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "main.go", `package main

func Main() {}
`)

	// First index.
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("first Index() error = %v", err)
	}

	// Second index — file unchanged, must be skipped.
	stats, err := idx.Index(context.Background())
	if err != nil {
		t.Fatalf("second Index() error = %v", err)
	}
	if stats.FilesSkipped != 1 {
		t.Errorf("FilesSkipped = %d, want 1", stats.FilesSkipped)
	}
	if stats.FilesIndexed != 0 {
		t.Errorf("FilesIndexed = %d, want 0 (skipped file must not be counted)", stats.FilesIndexed)
	}
}

func TestIndexer_SkipsUnchangedFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	src := `package main

func Hello() {}
`
	writeFile(t, root, "main.go", src)

	// First index — file should be parsed and stored.
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("first Index() error = %v", err)
	}

	sha1, err := st.GetFileSHA("main.go")
	if err != nil || sha1 == "" {
		t.Fatalf("expected SHA after first index, got %q err %v", sha1, err)
	}

	// Artificially corrupt the symbol to detect if re-parse happens.
	if err := st.UpsertFile(types.FileRecord{
		Path: "main.go", SHA256: sha1, ModTime: 0, Lang: "go",
	}); err != nil {
		t.Fatalf("UpsertFile() error = %v", err)
	}
	if err := st.DeleteSymbolsByFile("main.go"); err != nil {
		t.Fatalf("DeleteSymbolsByFile() error = %v", err)
	}

	// Second index — SHA unchanged, so file must be skipped (symbols stay empty).
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("second Index() error = %v", err)
	}

	syms, err := st.GetSymbolsByFile("main.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) != 0 {
		t.Errorf("expected 0 symbols (skip), got %d — file was re-indexed when it should not have been", len(syms))
	}
}

func TestIndexer_ForceReindexesUnchangedFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := New(
		Config{ProjectRoot: root, Logger: discardLogger(), Force: true},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(3),
	)

	src := `package main

func Hello() {}
`
	writeFile(t, root, "main.go", src)

	if _, err := idx.Index(t.Context()); err != nil {
		t.Fatalf("first Index() error = %v", err)
	}

	sha1, err := st.GetFileSHA("main.go")
	if err != nil || sha1 == "" {
		t.Fatalf("expected SHA after first index, got %q err %v", sha1, err)
	}
	if err := st.UpsertFile(types.FileRecord{Path: "main.go", SHA256: sha1, ModTime: 0, Lang: "go"}); err != nil {
		t.Fatalf("UpsertFile() error = %v", err)
	}
	if err := st.DeleteSymbolsByFile("main.go"); err != nil {
		t.Fatalf("DeleteSymbolsByFile() error = %v", err)
	}

	stats, err := idx.Index(t.Context())
	if err != nil {
		t.Fatalf("second Index() error = %v", err)
	}
	if stats.FilesSkipped != 0 {
		t.Errorf("FilesSkipped = %d, want 0 when force reindex is enabled", stats.FilesSkipped)
	}

	syms, err := st.GetSymbolsByFile("main.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) == 0 {
		t.Fatal("expected symbols after forced reindex, got none")
	}
}

// ---- hashBytes Benchmarks --------------------------------------------------

func BenchmarkHashBytes_1KB(b *testing.B) {
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = hashBytes(data)
	}
}

func BenchmarkHashBytes_64KB(b *testing.B) {
	data := make([]byte, 64*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = hashBytes(data)
	}
}

func BenchmarkHashBytes_1MB(b *testing.B) {
	data := make([]byte, 1024*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = hashBytes(data)
	}
}

// ---- buildEmbedText tests --------------------------------------------------

func TestBuildEmbedText(t *testing.T) {
	tests := []struct {
		name string
		sym  types.Symbol
		want string
	}{
		{
			name: "function with signature and doc",
			sym: types.Symbol{
				Kind:       types.KindFunction,
				Name:       "Foo",
				Signature:  "func Foo(a int) error",
				DocComment: "Foo does something.",
			},
			want: "function Foo: func Foo(a int) error\nFoo does something.",
		},
		{
			name: "method with receiver",
			sym: types.Symbol{
				Kind:       types.KindMethod,
				Name:       "Bar",
				Receiver:   "Server",
				Signature:  "func (s *Server) Bar()",
				DocComment: "",
			},
			want: "method Server.Bar: func (s *Server) Bar()",
		},
		{
			name: "type no signature no doc",
			sym: types.Symbol{
				Kind: types.KindType,
				Name: "Config",
			},
			want: "type Config",
		},
		{
			name: "function no signature with doc",
			sym: types.Symbol{
				Kind:       types.KindFunction,
				Name:       "Init",
				DocComment: "Init initializes.",
			},
			want: "function Init\nInit initializes.",
		},
		{
			name: "function with file path",
			sym: types.Symbol{
				File:      "internal/server/handler.go",
				Kind:      types.KindFunction,
				Name:      "Serve",
				Signature: "func Serve()",
			},
			want: "internal/server/handler.go function Serve: func Serve()",
		},
		{
			name: "function with snippet",
			sym: types.Symbol{
				Kind:      types.KindFunction,
				Name:      "Add",
				Signature: "func Add(a, b int) int",
				Snippet:   "func Add(a, b int) int {\n\treturn a + b\n}",
			},
			want: "function Add: func Add(a, b int) int\nfunc Add(a, b int) int {\n\treturn a + b\n}",
		},
		{
			name: "function with file path and snippet",
			sym: types.Symbol{
				File:      "pkg/math/add.go",
				Kind:      types.KindFunction,
				Name:      "Add",
				Signature: "func Add(a, b int) int",
				Snippet:   "func Add(a, b int) int { return a + b }",
			},
			want: "pkg/math/add.go function Add: func Add(a, b int) int\nfunc Add(a, b int) int { return a + b }",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := buildEmbedText(tc.sym)
			if got != tc.want {
				t.Errorf("buildEmbedText() = %q, want %q", got, tc.want)
			}
		})
	}
}

// ---- buildEnrichedEmbedText tests ------------------------------------------

func TestBuildEnrichedEmbedText(t *testing.T) {
	tests := []struct {
		name    string
		sym     types.Symbol
		calls   []string
		imports []string
		want    string
	}{
		{
			name: "no calls no imports",
			sym: types.Symbol{
				Kind:      types.KindFunction,
				Name:      "Foo",
				Signature: "func Foo(a int) error",
			},
			want: "function Foo: func Foo(a int) error",
		},
		{
			name: "with calls",
			sym: types.Symbol{
				Kind:      types.KindFunction,
				Name:      "Foo",
				Signature: "func Foo()",
			},
			calls: []string{"Bar", "Baz", "Qux"},
			want:  "function Foo: func Foo()\ncalls: Bar, Baz, Qux",
		},
		{
			name: "with calls and imports and file path",
			sym: types.Symbol{
				File:       "internal/server/server.go",
				Kind:       types.KindMethod,
				Name:       "Handle",
				Receiver:   "Server",
				Signature:  "func (s *Server) Handle()",
				DocComment: "Handle handles requests.",
			},
			calls:   []string{"ValidateSession", "GetUser"},
			imports: []string{"net/http", "github.com/foo/bar"},
			want:    "internal/server/server.go method Server.Handle: func (s *Server) Handle()\nHandle handles requests.\ncalls: ValidateSession, GetUser\nimports: net/http, github.com/foo/bar",
		},
		{
			name: "calls deduped",
			sym: types.Symbol{
				Kind: types.KindFunction,
				Name: "Run",
			},
			calls: []string{"A", "B", "A", "C", "B"},
			want:  "function Run\ncalls: A, B, C",
		},
		{
			name: "calls truncated to maxEnrichCalls",
			sym: types.Symbol{
				Kind: types.KindFunction,
				Name: "Big",
			},
			calls: []string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"},
			want:  "function Big\ncalls: A, B, C, D, E, F, G, H, I, J",
		},
		{
			name: "imports truncated to maxEnrichImports",
			sym: types.Symbol{
				Kind: types.KindFunction,
				Name: "Big",
			},
			imports: []string{"a", "b", "c", "d", "e", "f", "g", "h"},
			want:    "function Big\nimports: a, b, c, d, e, f",
		},
		{
			name: "only imports no calls",
			sym: types.Symbol{
				Kind: types.KindFunction,
				Name: "Init",
			},
			imports: []string{"os", "fmt"},
			want:    "function Init\nimports: os, fmt",
		},
		{
			name: "with snippet",
			sym: types.Symbol{
				Kind:      types.KindFunction,
				Name:      "Add",
				Signature: "func Add(a, b int) int",
				Snippet:   "func Add(a, b int) int {\n\treturn a + b\n}",
			},
			want: "function Add: func Add(a, b int) int\nfunc Add(a, b int) int {\n\treturn a + b\n}",
		},
		{
			name: "with file, snippet, calls, and imports",
			sym: types.Symbol{
				File:       "pkg/svc/svc.go",
				Kind:       types.KindFunction,
				Name:       "Run",
				Signature:  "func Run(ctx context.Context) error",
				DocComment: "Run starts the service.",
				Snippet:    "func Run(ctx context.Context) error {\n\treturn serve(ctx)\n}",
			},
			calls:   []string{"serve"},
			imports: []string{"context"},
			want:    "pkg/svc/svc.go function Run: func Run(ctx context.Context) error\nRun starts the service.\nfunc Run(ctx context.Context) error {\n\treturn serve(ctx)\n}\ncalls: serve\nimports: context",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := buildEnrichedEmbedText(tc.sym, tc.calls, tc.imports)
			if got != tc.want {
				t.Errorf("buildEnrichedEmbedText() = %q, want %q", got, tc.want)
			}
		})
	}
}

// ---- dedupStrings tests ----------------------------------------------------

func TestDedupStrings(t *testing.T) {
	tests := []struct {
		name  string
		input []string
		want  []string
	}{
		{name: "empty", input: nil, want: []string{}},
		{name: "no dups", input: []string{"a", "b", "c"}, want: []string{"a", "b", "c"}},
		{name: "with dups", input: []string{"a", "b", "a", "c", "b"}, want: []string{"a", "b", "c"}},
		{name: "all same", input: []string{"x", "x", "x"}, want: []string{"x"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := dedupStrings(tc.input)
			if len(got) != len(tc.want) {
				t.Fatalf("dedupStrings() len = %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("dedupStrings()[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// ---- buildCallMap / buildImportMap tests ------------------------------------

func TestBuildCallMap(t *testing.T) {
	edges := []types.CallEdge{
		{CallerFile: "a.go", CallerSymbol: "Foo", CalleeSymbol: "Bar"},
		{CallerFile: "a.go", CallerSymbol: "Foo", CalleeSymbol: "Baz"},
		{CallerFile: "b.go", CallerSymbol: "Run", CalleeSymbol: "Init"},
	}
	m := buildCallMap(edges)
	if got := m["a.go:Foo"]; len(got) != 2 || got[0] != "Bar" || got[1] != "Baz" {
		t.Errorf("buildCallMap[a.go:Foo] = %v, want [Bar Baz]", got)
	}
	if got := m["b.go:Run"]; len(got) != 1 || got[0] != "Init" {
		t.Errorf("buildCallMap[b.go:Run] = %v, want [Init]", got)
	}
}

func TestBuildImportMap(t *testing.T) {
	edges := []types.ImportEdge{
		{ImporterFile: "a.go", ImportedPath: "fmt"},
		{ImporterFile: "a.go", ImportedPath: "os"},
		{ImporterFile: "b.go", ImportedPath: "net/http"},
	}
	m := buildImportMap(edges)
	if got := m["a.go"]; len(got) != 2 || got[0] != "fmt" || got[1] != "os" {
		t.Errorf("buildImportMap[a.go] = %v, want [fmt os]", got)
	}
	if got := m["b.go"]; len(got) != 1 || got[0] != "net/http" {
		t.Errorf("buildImportMap[b.go] = %v, want [net/http]", got)
	}
}

// ---- Integration: enriched embed text reaches the embedder -----------------

// recordingEmbedder records the texts passed to Embed for inspection.
type recordingEmbedder struct {
	dims  int
	mu    sync.Mutex
	texts []string
}

func (e *recordingEmbedder) Model() string { return "recording-test" }
func (e *recordingEmbedder) Dims() int     { return e.dims }
func (e *recordingEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.mu.Lock()
	e.texts = append(e.texts, text)
	e.mu.Unlock()
	v := make([]float32, e.dims)
	for i := range v {
		v[i] = 0.1
	}
	return v, nil
}

func TestIndex_EnrichedEmbedTextReachesEmbedder(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)

	rec := &recordingEmbedder{dims: 3}
	idx := New(
		Config{ProjectRoot: root, Logger: discardLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		rec,
	)

	writeFile(t, root, "svc.go", `package svc

import "fmt"

// Greet produces a greeting.
func Greet(name string) string {
	msg := fmt.Sprintf("hello %s", name)
	return transform(msg)
}

func transform(s string) string { return s }
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	// The Greet function should have enriched embed text containing "calls:".
	rec.mu.Lock()
	defer rec.mu.Unlock()

	var greetText string
	for _, text := range rec.texts {
		if strings.Contains(text, "Greet") {
			greetText = text
			break
		}
	}
	if greetText == "" {
		t.Fatal("no embed text found containing 'Greet'")
	}
	if !strings.Contains(greetText, "calls:") {
		t.Errorf("Greet embed text missing 'calls:' enrichment: %q", greetText)
	}
	if !strings.Contains(greetText, "Sprintf") {
		t.Errorf("Greet embed text missing call target 'Sprintf': %q", greetText)
	}
	if !strings.Contains(greetText, "transform") {
		t.Errorf("Greet embed text missing call target 'transform': %q", greetText)
	}
	if !strings.Contains(greetText, "imports:") {
		t.Errorf("Greet embed text missing 'imports:' enrichment: %q", greetText)
	}
	if !strings.Contains(greetText, "fmt") {
		t.Errorf("Greet embed text missing import 'fmt': %q", greetText)
	}
	// Verify file path is included in embed text.
	if !strings.Contains(greetText, "svc.go") {
		t.Errorf("Greet embed text missing file path 'svc.go': %q", greetText)
	}
	// Verify snippet (body excerpt) is included in embed text.
	if !strings.Contains(greetText, "Sprintf") && !strings.Contains(greetText, "transform(msg)") {
		t.Errorf("Greet embed text missing snippet content: %q", greetText)
	}
}

// ---- indexFile tests (synchronous single-file path) ------------------------

func TestIndexFile_IndexesNewFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "single.go", `package single

// Greet says hello.
func Greet(name string) string { return "hi " + name }
`)

	count, skipped, err := idx.indexFile(context.Background(), filepath.Join(root, "single.go"), "single.go", ".go")
	if err != nil {
		t.Fatalf("indexFile() error = %v", err)
	}
	if skipped {
		t.Error("indexFile() skipped = true, want false for new file")
	}
	if count != 1 {
		t.Errorf("indexFile() count = %d, want 1", count)
	}

	syms, err := st.GetSymbolsByFile("single.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(syms) != 1 || syms[0].Name != "Greet" {
		t.Errorf("GetSymbolsByFile() = %v, want [Greet]", syms)
	}
}

func TestIndexFile_SkipsUnchangedFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "same.go", `package same

func Same() {}
`)

	// First index.
	_, _, err := idx.indexFile(context.Background(), filepath.Join(root, "same.go"), "same.go", ".go")
	if err != nil {
		t.Fatalf("first indexFile() error = %v", err)
	}

	// Second index — same content, should skip.
	_, skipped, err := idx.indexFile(context.Background(), filepath.Join(root, "same.go"), "same.go", ".go")
	if err != nil {
		t.Fatalf("second indexFile() error = %v", err)
	}
	if !skipped {
		t.Error("indexFile() skipped = false, want true for unchanged file")
	}
}

// ---- embedSymbols tests ----------------------------------------------------

func TestEmbedSymbols_StoresEmbeddings(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexerWithEmbed(t, root, st)

	// Seed a file and symbols so foreign keys are satisfied.
	if err := st.UpsertFile(types.FileRecord{Path: "emb.go", SHA256: "x", ModTime: 1, Lang: "go"}); err != nil {
		t.Fatal(err)
	}
	syms := []types.Symbol{
		{ID: "emb.go:A:func", File: "emb.go", Name: "A", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		{ID: "emb.go:B:func", File: "emb.go", Name: "B", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
	}
	if err := st.UpsertSymbolsBatch(syms); err != nil {
		t.Fatal(err)
	}

	idx.embedSymbols(context.Background(), syms, nil, nil)

	// Verify embeddings were stored by checking SymbolIDsWithoutEmbeddings.
	missing, err := st.SymbolIDsWithoutEmbeddings()
	if err != nil {
		t.Fatal(err)
	}
	if len(missing) != 0 {
		t.Errorf("SymbolIDsWithoutEmbeddings() = %v, want empty (all should be embedded)", missing)
	}
}

func TestEmbedSymbols_EmptySlice(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	// Should not panic or error with empty input.
	idx.embedSymbols(context.Background(), nil, nil, nil)
}

func TestIndexer_BatchEmbedFailureSelfHealsWithPerItemRetry(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	emb := &flakyBatchEmbedder{dims: 3}
	idx := New(
		Config{ProjectRoot: root, Logger: discardLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		emb,
	)

	writeFile(t, root, "retry.go", `package retry

func A() {}
func B() {}
func C() {}
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	missing, err := st.SymbolIDsWithoutEmbeddings()
	if err != nil {
		t.Fatalf("SymbolIDsWithoutEmbeddings() error = %v", err)
	}
	if len(missing) != 0 {
		t.Fatalf("SymbolIDsWithoutEmbeddings() = %v, want empty after retry self-heal", missing)
	}

	batchCalls, embedCalls := emb.counts()
	if batchCalls == 0 {
		t.Fatal("EmbedBatch() was not called, want batch path")
	}
	if embedCalls == 0 {
		t.Fatal("Embed() per-item retry was not called after batch failure")
	}
}

// ---- Watch tests -----------------------------------------------------------

func waitFor(t *testing.T, timeout time.Duration, fn func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fn() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("condition not met before timeout")
}

func TestWatch_DetectsNewFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	// Use a short debounce for fast tests.
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	// Give watcher time to start.
	time.Sleep(100 * time.Millisecond)

	// Create a new file.
	writeFile(t, root, "watched.go", `package watched

func WatchedFunc() {}
`)

	// Wait for debounce + processing.
	time.Sleep(500 * time.Millisecond)

	syms, err := st.GetSymbolsByFile("watched.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) == 0 {
		t.Error("Watch did not index the new file")
	}

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_DetectsFileInDirectoryCreatedAfterStartup(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	time.Sleep(100 * time.Millisecond)

	newDir := filepath.Join(root, "newdir")
	if err := os.MkdirAll(newDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	// Give the watcher a moment to attach to the new directory.
	time.Sleep(100 * time.Millisecond)

	writeFile(t, newDir, "new.go", `package newdir

func NewDirFunc() {}
`)

	waitFor(t, 2*time.Second, func() bool {
		syms, err := st.GetSymbolsByFile("newdir/new.go")
		if err != nil {
			t.Fatalf("GetSymbolsByFile() error = %v", err)
		}
		return len(syms) > 0
	})

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_IndexesExistingFilesInNewlyCreatedDirectoryTree(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	time.Sleep(100 * time.Millisecond)

	srcParent := t.TempDir()
	srcTree := filepath.Join(srcParent, "imported")
	if err := os.MkdirAll(filepath.Join(srcTree, "nested"), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	writeFile(t, filepath.Join(srcTree, "nested"), "existing.go", `package nested

func Existing() {}
`)

	dstTree := filepath.Join(root, "imported")
	if err := os.Rename(srcTree, dstTree); err != nil {
		t.Fatalf("Rename() error = %v", err)
	}

	waitFor(t, 2*time.Second, func() bool {
		syms, err := st.GetSymbolsByFile("imported/nested/existing.go")
		if err != nil {
			t.Fatalf("GetSymbolsByFile() error = %v", err)
		}
		return len(syms) > 0
	})

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_ReconcilesFileCreatedWhileOfflineOnStartup(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	writeFile(t, root, "offline.go", `package offline

func OfflineCreated() {}
`)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	waitFor(t, 2*time.Second, func() bool {
		syms, err := st.GetSymbolsByFile("offline.go")
		if err != nil {
			t.Fatalf("GetSymbolsByFile() error = %v", err)
		}
		return len(syms) > 0
	})

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_ReconcilesDeletedFileWhileOfflineOnStartup(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	path := writeFile(t, root, "stale.go", `package stale

func Stale() {}
`)
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	if err := os.Remove(path); err != nil {
		t.Fatalf("Remove() error = %v", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	waitFor(t, 2*time.Second, func() bool {
		sha, err := st.GetFileSHA("stale.go")
		if err != nil {
			t.Fatalf("GetFileSHA() error = %v", err)
		}
		return sha == ""
	})

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_IgnoresNewNodeModulesDirectoryAfterStartup(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	time.Sleep(100 * time.Millisecond)

	nmDir := filepath.Join(root, "node_modules", "pkg")
	if err := os.MkdirAll(nmDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	writeFile(t, nmDir, "ignored.go", `package pkg

func Ignored() {}
`)

	// Wait for any potential (incorrect) indexing and assert it does not happen.
	time.Sleep(400 * time.Millisecond)

	sha, err := st.GetFileSHA("node_modules/pkg/ignored.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if sha != "" {
		t.Error("node_modules/pkg/ignored.go was indexed but should be ignored")
	}

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_DeleteWatchedDirectoryDoesNotCrash(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()

	time.Sleep(100 * time.Millisecond)

	tmpDir := filepath.Join(root, "tempdir")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	time.Sleep(100 * time.Millisecond)

	if err := os.RemoveAll(tmpDir); err != nil {
		t.Fatalf("RemoveAll() error = %v", err)
	}

	writeFile(t, root, "after.go", `package after

func AfterDelete() {}
`)

	waitFor(t, 2*time.Second, func() bool {
		syms, err := st.GetSymbolsByFile("after.go")
		if err != nil {
			t.Fatalf("GetSymbolsByFile() error = %v", err)
		}
		return len(syms) > 0
	})

	cancel()
	if err := <-watchErr; err != nil {
		t.Errorf("Watch() error = %v", err)
	}
}

func TestWatch_DetectsModifiedFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	// Pre-create file and index it.
	path := writeFile(t, root, "mod.go", `package mod

func OldFunc() {}
`)
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()
	time.Sleep(100 * time.Millisecond)

	// Modify the file.
	if err := os.WriteFile(path, []byte(`package mod

func NewFunc() {}
`), 0o644); err != nil {
		t.Fatal(err)
	}

	time.Sleep(500 * time.Millisecond)

	syms, err := st.GetSymbolsByFile("mod.go")
	if err != nil {
		t.Fatal(err)
	}
	found := make(map[string]bool)
	for _, s := range syms {
		found[s.Name] = true
	}
	if found["OldFunc"] {
		t.Error("stale symbol OldFunc still present after watch reindex")
	}
	if !found["NewFunc"] {
		t.Error("new symbol NewFunc not found after watch reindex")
	}

	cancel()
	<-watchErr
}

func TestWatch_DetectsDeletedFile(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)
	idx.cfg.WatchDebounce = 50 * time.Millisecond

	path := writeFile(t, root, "del.go", `package del

func DelFunc() {}
`)
	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	watchErr := make(chan error, 1)
	go func() {
		watchErr <- idx.Watch(ctx)
	}()
	time.Sleep(100 * time.Millisecond)

	// Delete the file.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}

	time.Sleep(500 * time.Millisecond)

	sha, err := st.GetFileSHA("del.go")
	if err != nil {
		t.Fatal(err)
	}
	if sha != "" {
		t.Error("deleted file still has SHA in store after Watch detected removal")
	}

	cancel()
	<-watchErr
}

// ---- Config setDefaults tests ----------------------------------------------

func TestConfig_SetDefaults(t *testing.T) {
	cfg := Config{}
	cfg.setDefaults()

	if cfg.Workers <= 0 {
		t.Errorf("Workers = %d, want > 0", cfg.Workers)
	}
	if cfg.EmbedConcurrency <= 0 {
		t.Errorf("EmbedConcurrency = %d, want > 0", cfg.EmbedConcurrency)
	}
	if cfg.WatchDebounce <= 0 {
		t.Errorf("WatchDebounce = %v, want > 0", cfg.WatchDebounce)
	}
	if cfg.Logger == nil {
		t.Error("Logger = nil, want non-nil default")
	}
}

// ---- Vendor indexing tests -------------------------------------------------

func TestIndexer_IndexesVendorWithTierVendor(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	// Create vendor directory with a Go file.
	vendorDir := filepath.Join(root, "vendor", "github.com", "foo")
	if err := os.MkdirAll(vendorDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	writeFile(t, vendorDir, "bar.go", `package foo

func VendoredFunc() {}
`)
	// Also create a normal source file.
	writeFile(t, root, "main.go", `package main

func Main() {}
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	// Vendor file should be indexed (not skipped).
	sha, err := st.GetFileSHA("vendor/github.com/foo/bar.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if sha == "" {
		t.Fatal("vendor file was not indexed; vendor should be indexed with TierVendor")
	}

	// Verify the vendor symbol has TierVendor.
	syms, err := st.GetSymbolsByFile("vendor/github.com/foo/bar.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) != 1 {
		t.Fatalf("GetSymbolsByFile() = %d symbols, want 1", len(syms))
	}
	if syms[0].SourceTier != types.TierVendor {
		t.Errorf("vendor symbol SourceTier = %d, want %d (TierVendor)", syms[0].SourceTier, types.TierVendor)
	}

	// Verify the main.go symbol has TierCode.
	mainSyms, err := st.GetSymbolsByFile("main.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile(main.go) error = %v", err)
	}
	if len(mainSyms) != 1 {
		t.Fatalf("GetSymbolsByFile(main.go) = %d symbols, want 1", len(mainSyms))
	}
	if mainSyms[0].SourceTier != types.TierCode {
		t.Errorf("main.go symbol SourceTier = %d, want %d (TierCode)", mainSyms[0].SourceTier, types.TierCode)
	}
}

func TestIndexer_GeneratedFileGetsTierLowSignal(t *testing.T) {
	root := t.TempDir()
	st := openTestStore(t)
	idx := newTestIndexer(t, root, st)

	writeFile(t, root, "zz_generated.deepcopy.go", `package main

func DeepCopy() {}
`)

	if _, err := idx.Index(context.Background()); err != nil {
		t.Fatalf("Index() error = %v", err)
	}

	syms, err := st.GetSymbolsByFile("zz_generated.deepcopy.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) != 1 {
		t.Fatalf("GetSymbolsByFile() = %d symbols, want 1", len(syms))
	}
	if syms[0].SourceTier != types.TierLowSignal {
		t.Errorf("generated file SourceTier = %d, want %d (TierLowSignal)", syms[0].SourceTier, types.TierLowSignal)
	}
}
