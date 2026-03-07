package store

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
	"github.com/coder/hnsw"
)

type stubSemanticSearcher struct {
	candidates []semanticCandidate
	err        error
}

func (s stubSemanticSearcher) Search(_ []float32, _ int) ([]semanticCandidate, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.candidates, nil
}

func openTestStore(t *testing.T) *Store {
	t.Helper()
	st, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

func openTestStoreWithOptions(t *testing.T, opts OpenOptions) *Store {
	t.Helper()
	st, err := OpenWithOptions(filepath.Join(t.TempDir(), ".ctxpp", "index.db"), opts)
	if err != nil {
		t.Fatalf("OpenWithOptions() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

func openTestStoreAtPath(t *testing.T, dbPath string, opts OpenOptions) *Store {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	st, err := OpenWithOptions(dbPath, opts)
	if err != nil {
		t.Fatalf("OpenWithOptions() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

// seedFile is a test helper that inserts a file and a batch of symbols.
func seedFile(t *testing.T, st *Store, file types.FileRecord, syms []types.Symbol) {
	t.Helper()
	if err := st.UpsertFile(file); err != nil {
		t.Fatalf("UpsertFile() error = %v", err)
	}
	if err := st.UpsertSymbolsBatch(syms); err != nil {
		t.Fatalf("UpsertSymbolsBatch() error = %v", err)
	}
}

// ---- Benchmarks ------------------------------------------------------------

// seedBenchSymbols creates n symbols with embeddings for benchmarking.
func seedBenchSymbols(b *testing.B, st *Store, n, dims int) {
	b.Helper()
	f := types.FileRecord{Path: "bench.go", SHA256: "x", ModTime: 1, Lang: "go"}
	if err := st.UpsertFile(f); err != nil {
		b.Fatal(err)
	}
	syms := make([]types.Symbol, n)
	for i := range syms {
		name := "Symbol" + strconv.Itoa(i)
		syms[i] = types.Symbol{
			ID: "bench.go:" + name + ":function", File: "bench.go",
			Name: name, Kind: types.KindFunction,
			Signature:  "func " + name + "()",
			DocComment: "doc for " + name,
			StartLine:  i*3 + 1, EndLine: i*3 + 3,
		}
	}
	if err := st.UpsertSymbolsBatch(syms); err != nil {
		b.Fatal(err)
	}
	if dims > 0 {
		items := make([]EmbeddingItem, n)
		for i := range items {
			vec := make([]float32, dims)
			for j := range vec {
				vec[j] = float32(i+j) / 1000
			}
			items[i] = EmbeddingItem{SymbolID: syms[i].ID, Model: "bench", Vector: vec}
		}
		if err := st.UpsertEmbeddingsBatch(items); err != nil {
			b.Fatal(err)
		}
	}
}

func openBenchStoreWithOptions(b *testing.B, opts OpenOptions) *Store {
	b.Helper()
	st, err := OpenWithOptions(filepath.Join(b.TempDir(), ".ctxpp", "bench.db"), opts)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = st.Close() })
	return st
}

func BenchmarkSearchKeyword_200(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	seedBenchSymbols(b, st, 200, 0)
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.SearchKeyword("Symbol100", 10)
	}
}

func BenchmarkSearchKeyword_10k(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	seedBenchSymbols(b, st, 10_000, 0)
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.SearchKeyword("Symbol5000", 10)
	}
}

func BenchmarkSearchSemantic_500(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	const dims = 768
	seedBenchSymbols(b, st, 500, dims)
	query := make([]float32, dims)
	for i := range query {
		query[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.SearchSemantic(query, 10)
	}
}

func BenchmarkSearchSemantic_10k(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	const dims = 768
	seedBenchSymbols(b, st, 10_000, dims)
	query := make([]float32, dims)
	for i := range query {
		query[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.SearchSemantic(query, 10)
	}
}

func BenchmarkSearchSemanticANN_10k(b *testing.B) {
	dbPath := filepath.Join(b.TempDir(), ".ctxpp", "bench.db")
	st, err := OpenWithOptions(dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	const dims = 768
	seedBenchSymbols(b, st, 10_000, dims)
	if err := st.BuildANNArtifacts(); err != nil {
		b.Fatal(err)
	}

	annStore, err := OpenWithOptions(dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	if err != nil {
		b.Fatal(err)
	}
	defer annStore.Close()

	query := make([]float32, dims)
	for i := range query {
		query[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = annStore.SearchSemantic(query, 10)
	}
}

func BenchmarkBuildANNArtifacts_10k(b *testing.B) {
	st := openBenchStoreWithOptions(b, OpenOptions{SemanticMode: SemanticModeBruteForce})
	const dims = 768
	seedBenchSymbols(b, st, 10_000, dims)
	b.ResetTimer()
	for b.Loop() {
		if err := st.BuildANNArtifacts(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSearchSemantic_100k(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	const dims = 768
	seedBenchSymbols(b, st, 100_000, dims)
	query := make([]float32, dims)
	for i := range query {
		query[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.SearchSemantic(query, 10)
	}
}

func BenchmarkSearchSemanticANN_100k(b *testing.B) {
	dbPath := filepath.Join(b.TempDir(), ".ctxpp", "bench.db")
	st, err := OpenWithOptions(dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	const dims = 768
	seedBenchSymbols(b, st, 100_000, dims)
	if err := st.BuildANNArtifacts(); err != nil {
		b.Fatal(err)
	}

	annStore, err := OpenWithOptions(dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	if err != nil {
		b.Fatal(err)
	}
	defer annStore.Close()

	query := make([]float32, dims)
	for i := range query {
		query[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = annStore.SearchSemantic(query, 10)
	}
}

func BenchmarkBuildANNArtifacts_100k(b *testing.B) {
	st := openBenchStoreWithOptions(b, OpenOptions{SemanticMode: SemanticModeBruteForce})
	const dims = 768
	seedBenchSymbols(b, st, 100_000, dims)
	b.ResetTimer()
	for b.Loop() {
		if err := st.BuildANNArtifacts(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUpsertEmbeddingsBatchWithDeferredANNSync_10k(b *testing.B) {
	const (
		dims      = 768
		total     = 10_000
		batchSize = 64
	)

	makeItems := func() []EmbeddingItem {
		items := make([]EmbeddingItem, 0, total)
		for i := 0; i < total; i++ {
			vec := make([]float32, dims)
			for j := range vec {
				vec[j] = float32(i+j) / 1000
			}
			items = append(items, EmbeddingItem{
				SymbolID: "bench.go:Symbol" + strconv.Itoa(i) + ":function",
				Model:    "bench",
				Vector:   vec,
			})
		}
		return items
	}

	st := openBenchStoreWithOptions(b, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedBenchSymbols(b, st, total, 0)
	if err := st.UpsertEmbeddingsBatch(makeItems()); err != nil {
		b.Fatal(err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for b.Loop() {
		st.BeginDeferredANNSync()
		items := makeItems()
		for start := 0; start < len(items); start += batchSize {
			end := start + batchSize
			if end > len(items) {
				end = len(items)
			}
			if err := st.UpsertEmbeddingsBatch(items[start:end]); err != nil {
				b.Fatal(err)
			}
		}
		if err := st.EndDeferredANNSync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUpsertEmbeddingsBatchWithImmediateANNSync_10k(b *testing.B) {
	const (
		dims      = 768
		total     = 10_000
		batchSize = 64
	)

	makeItems := func() []EmbeddingItem {
		items := make([]EmbeddingItem, 0, total)
		for i := 0; i < total; i++ {
			vec := make([]float32, dims)
			for j := range vec {
				vec[j] = float32(i+j) / 1000
			}
			items = append(items, EmbeddingItem{
				SymbolID: "bench.go:Symbol" + strconv.Itoa(i) + ":function",
				Model:    "bench",
				Vector:   vec,
			})
		}
		return items
	}

	st := openBenchStoreWithOptions(b, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedBenchSymbols(b, st, total, 0)
	if err := st.UpsertEmbeddingsBatch(makeItems()); err != nil {
		b.Fatal(err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for b.Loop() {
		items := makeItems()
		for start := 0; start < len(items); start += batchSize {
			end := start + batchSize
			if end > len(items) {
				end = len(items)
			}
			if err := st.UpsertEmbeddingsBatch(items[start:end]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkSearchHybrid_10k(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	const dims = 768
	seedBenchSymbols(b, st, 10_000, dims)
	query := make([]float32, dims)
	for i := range query {
		query[i] = 0.5
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.SearchHybrid(query, "Symbol5000", 10)
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	const dims = 768
	a := make([]float32, dims)
	bv := make([]float32, dims)
	for i := range a {
		a[i] = float32(i) / float32(dims)
		bv[i] = float32(dims-i) / float32(dims)
	}
	b.ResetTimer()
	for b.Loop() {
		cosineSimilarity(a, bv)
	}
}

func BenchmarkUpsertSymbolsBatch_1k(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	f := types.FileRecord{Path: "bench.go", SHA256: "x", ModTime: 1, Lang: "go"}
	if err := st.UpsertFile(f); err != nil {
		b.Fatal(err)
	}

	syms := make([]types.Symbol, 1000)
	for i := range syms {
		name := "Sym" + strconv.Itoa(i)
		syms[i] = types.Symbol{
			ID: "bench.go:" + name + ":function", File: "bench.go",
			Name: name, Kind: types.KindFunction,
			Signature: "func " + name + "()", StartLine: i, EndLine: i + 1,
		}
	}
	b.ResetTimer()
	for b.Loop() {
		_ = st.DeleteSymbolsByFile("bench.go")
		_ = st.UpsertSymbolsBatch(syms)
	}
}

// seedMultiFileSymbols creates nFiles files, each with symsPerFile symbols.
func seedMultiFileSymbols(b *testing.B, st *Store, nFiles, symsPerFile int) {
	b.Helper()
	for f := 0; f < nFiles; f++ {
		fpath := "file" + strconv.Itoa(f) + ".go"
		if err := st.UpsertFile(types.FileRecord{Path: fpath, SHA256: "sha" + strconv.Itoa(f), ModTime: 1, Lang: "go"}); err != nil {
			b.Fatal(err)
		}
		syms := make([]types.Symbol, symsPerFile)
		for i := range syms {
			name := "Sym" + strconv.Itoa(f) + "_" + strconv.Itoa(i)
			syms[i] = types.Symbol{
				ID: fpath + ":" + name + ":func", File: fpath,
				Name: name, Kind: types.KindFunction,
				Signature: "func " + name + "()", StartLine: i*3 + 1, EndLine: i*3 + 3,
			}
		}
		if err := st.UpsertSymbolsBatch(syms); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGetSymbolsByFile_100files_50syms(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	seedMultiFileSymbols(b, st, 100, 50) // 5000 total symbols
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.GetSymbolsByFile("file50.go")
	}
}

func BenchmarkDeleteSymbolsByFile_100files_50syms(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	seedMultiFileSymbols(b, st, 100, 50)
	b.ResetTimer()
	for b.Loop() {
		// Delete and re-insert to keep benchmark repeatable.
		_ = st.DeleteSymbolsByFile("file50.go")
		syms := make([]types.Symbol, 50)
		for i := range syms {
			name := "Sym50_" + strconv.Itoa(i)
			syms[i] = types.Symbol{
				ID: "file50.go:" + name + ":func", File: "file50.go",
				Name: name, Kind: types.KindFunction, StartLine: i*3 + 1, EndLine: i*3 + 3,
			}
		}
		_ = st.UpsertSymbolsBatch(syms)
	}
}

func BenchmarkCalleeSymbols_10kEdges(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	// Create 10k call edges from 100 callers to 100 callees.
	for f := 0; f < 100; f++ {
		fpath := "file" + strconv.Itoa(f) + ".go"
		edges := make([]types.CallEdge, 100)
		for i := range edges {
			edges[i] = types.CallEdge{
				CallerFile:   fpath,
				CallerSymbol: fpath + ":Caller" + strconv.Itoa(f) + ":func",
				CalleeSymbol: "Callee" + strconv.Itoa(i),
				Line:         i + 1,
			}
		}
		if err := st.UpsertCallEdges(fpath, edges); err != nil {
			b.Fatal(err)
		}
	}
	target := "file50.go:Caller50:func"
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.CalleeSymbols(target)
	}
}

func BenchmarkCallerSymbols_10kEdges(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	for f := 0; f < 100; f++ {
		fpath := "file" + strconv.Itoa(f) + ".go"
		edges := make([]types.CallEdge, 100)
		for i := range edges {
			edges[i] = types.CallEdge{
				CallerFile:   fpath,
				CallerSymbol: fpath + ":Caller" + strconv.Itoa(f) + ":func",
				CalleeSymbol: "Callee" + strconv.Itoa(i),
				Line:         i + 1,
			}
		}
		if err := st.UpsertCallEdges(fpath, edges); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.CallerSymbols("Callee50")
	}
}

func BenchmarkGetSymbolsByNames_100names(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	seedBenchSymbols(b, st, 10_000, 0)
	names := make([]string, 100)
	for i := range names {
		names[i] = "Symbol" + strconv.Itoa(i*100)
	}
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.GetSymbolsByNames(names)
	}
}

// BenchmarkGetSymbolsByFile_CoveringIndex tests the same query with a covering
// index on (file, start_line) vs the default (file) index.
func BenchmarkGetSymbolsByFile_CoveringIdx(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	// Add covering index
	if _, err := st.DB().Exec(`CREATE INDEX IF NOT EXISTS idx_symbols_file_line ON symbols(file, start_line)`); err != nil {
		b.Fatal(err)
	}
	seedMultiFileSymbols(b, st, 100, 50)
	b.ResetTimer()
	for b.Loop() {
		_, _ = st.GetSymbolsByFile("file50.go")
	}
}

func TestUpsertEmbeddingsBatch_StoresMultipleEmbeddings(t *testing.T) {
	st := openTestStore(t)

	f := types.FileRecord{Path: "batch.go", SHA256: "abc", ModTime: 1, Lang: "go"}
	syms := []types.Symbol{
		{ID: "batch.go:Foo:func", File: "batch.go", Name: "Foo", Kind: types.KindFunction, StartLine: 1, EndLine: 3},
		{ID: "batch.go:Bar:func", File: "batch.go", Name: "Bar", Kind: types.KindFunction, StartLine: 5, EndLine: 7},
		{ID: "batch.go:Baz:func", File: "batch.go", Name: "Baz", Kind: types.KindFunction, StartLine: 9, EndLine: 11},
	}
	seedFile(t, st, f, syms)

	items := []EmbeddingItem{
		{SymbolID: "batch.go:Foo:func", Model: "test", Vector: []float32{0.1, 0.2, 0.3}},
		{SymbolID: "batch.go:Bar:func", Model: "test", Vector: []float32{0.4, 0.5, 0.6}},
		{SymbolID: "batch.go:Baz:func", Model: "test", Vector: []float32{0.7, 0.8, 0.9}},
	}
	if err := st.UpsertEmbeddingsBatch(items); err != nil {
		t.Fatalf("UpsertEmbeddingsBatch() error = %v", err)
	}

	// Verify by semantic search — all three should be present.
	query := []float32{0.7, 0.8, 0.9} // closest to Baz
	got, err := st.SearchSemantic(query, 3)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("SearchSemantic() returned %d symbols, want 3", len(got))
	}
	if got[0].Name != "Baz" {
		t.Errorf("top result = %q, want Baz (closest to query)", got[0].Name)
	}
}

func TestUpsertEmbeddingsBatch_EmptySlice(t *testing.T) {
	st := openTestStore(t)
	if err := st.UpsertEmbeddingsBatch(nil); err != nil {
		t.Fatalf("UpsertEmbeddingsBatch(nil) error = %v", err)
	}
}

func TestUpsertCallEdges_ReplacesStaleEdges(t *testing.T) {
	st := openTestStore(t)

	first := []types.CallEdge{{CallerFile: "a.go", CallerSymbol: "A", CalleeSymbol: "Old", Line: 1}}
	if err := st.UpsertCallEdges("a.go", first); err != nil {
		t.Fatalf("UpsertCallEdges() error = %v", err)
	}

	second := []types.CallEdge{{CallerFile: "a.go", CallerSymbol: "A", CalleeSymbol: "New", Line: 1}}
	if err := st.UpsertCallEdges("a.go", second); err != nil {
		t.Fatalf("UpsertCallEdges() error = %v", err)
	}

	got, err := st.CalleeSymbols("A")
	if err != nil {
		t.Fatalf("CalleeSymbols() error = %v", err)
	}
	if len(got) != 1 || got[0] != "New" {
		t.Errorf("CalleeSymbols() = %v, want [New]", got)
	}
}

func TestCallEdges_CalleesAndCallers(t *testing.T) {
	st := openTestStore(t)

	edges := []types.CallEdge{
		{CallerFile: "main.go", CallerSymbol: "main", CalleeSymbol: "Foo", Line: 5},
		{CallerFile: "main.go", CallerSymbol: "main", CalleeSymbol: "Bar", Line: 6},
		{CallerFile: "foo.go", CallerSymbol: "Foo", CalleeSymbol: "Bar", Line: 2},
	}
	if err := st.UpsertCallEdges("main.go", edges[:2]); err != nil {
		t.Fatalf("UpsertCallEdges(main.go) error = %v", err)
	}
	if err := st.UpsertCallEdges("foo.go", edges[2:]); err != nil {
		t.Fatalf("UpsertCallEdges(foo.go) error = %v", err)
	}

	tests := []struct {
		name        string
		symbol      string
		wantCallees []string
		wantCallers []string
	}{
		{
			name:        "main calls Foo and Bar",
			symbol:      "main",
			wantCallees: []string{"Foo", "Bar"},
			wantCallers: []string{},
		},
		{
			name:        "Bar is called by main and Foo",
			symbol:      "Bar",
			wantCallees: []string{},
			wantCallers: []string{"main", "Foo"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			callees, err := st.CalleeSymbols(tc.symbol)
			if err != nil {
				t.Fatalf("CalleeSymbols() error = %v", err)
			}
			if len(callees) != len(tc.wantCallees) {
				t.Errorf("CalleeSymbols() = %v, want %v", callees, tc.wantCallees)
			}

			callers, err := st.CallerSymbols(tc.symbol)
			if err != nil {
				t.Fatalf("CallerSymbols() error = %v", err)
			}
			if len(callers) != len(tc.wantCallers) {
				t.Errorf("CallerSymbols() = %v, want %v", callers, tc.wantCallers)
			}
		})
	}
}

func TestSearchSemantic_ReturnsClosestVector(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "d.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "d.go:Near:function", File: "d.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "d.go:Far:function", File: "d.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	// Near points in the same direction as the query; Far points away.
	query := []float32{1, 0, 0}
	if err := st.UpsertEmbedding("d.go:Near:function", "m", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding() error = %v", err)
	}
	if err := st.UpsertEmbedding("d.go:Far:function", "m", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding() error = %v", err)
	}

	got, err := st.SearchSemantic(query, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) == 0 {
		t.Fatal("SearchSemantic() returned no results")
	}
	if got[0].Name != "Near" {
		t.Errorf("top result = %q, want %q", got[0].Name, "Near")
	}
}

func TestSearchSemantic_ANNModeFallsBackToBruteForce(t *testing.T) {
	st := openTestStoreWithOptions(t, OpenOptions{SemanticMode: SemanticModeANN})
	seedFile(t, st,
		types.FileRecord{Path: "fallback.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "fallback.go:Near:function", File: "fallback.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "fallback.go:Far:function", File: "fallback.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	query := []float32{1, 0, 0}
	if err := st.UpsertEmbedding("fallback.go:Near:function", "m", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding() error = %v", err)
	}
	if err := st.UpsertEmbedding("fallback.go:Far:function", "m", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding() error = %v", err)
	}

	got, err := st.SearchSemantic(query, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) == 0 {
		t.Fatal("SearchSemantic() returned no results")
	}
	if got[0].Name != "Near" {
		t.Fatalf("top result = %q, want Near", got[0].Name)
	}
	if st.semanticMode != SemanticModeBruteForce {
		t.Fatalf("semanticMode = %q, want fallback to %q", st.semanticMode, SemanticModeBruteForce)
	}
}

func TestSearchSemantic_ANNModeUsesANNSearcherWhenAvailable(t *testing.T) {
	origFactory := newANNSemanticSearcher
	t.Cleanup(func() { newANNSemanticSearcher = origFactory })

	newANNSemanticSearcher = func(_ *Store) (semanticSearcher, error) {
		return stubSemanticSearcher{candidates: []semanticCandidate{{rowid: 0, score: 1}}}, nil
	}

	st := openTestStoreWithOptions(t, OpenOptions{SemanticMode: SemanticModeANN})
	if st.semanticMode != SemanticModeANN {
		t.Fatalf("semanticMode = %q, want %q", st.semanticMode, SemanticModeANN)
	}
	if _, ok := st.semanticSearcher.(stubSemanticSearcher); !ok {
		t.Fatalf("semanticSearcher type = %T, want stubSemanticSearcher", st.semanticSearcher)
	}
}

func TestANNArtifactPaths_UseCtxppSiblingFiles(t *testing.T) {
	paths := annArtifactPaths(filepath.Join("repo", ".ctxpp", "index.db"))
	if paths.Index != filepath.Join("repo", ".ctxpp", "ann-hnsw.bin") {
		t.Fatalf("Index path = %q, want %q", paths.Index, filepath.Join("repo", ".ctxpp", "ann-hnsw.bin"))
	}
	if paths.Metadata != filepath.Join("repo", ".ctxpp", "ann-hnsw.json") {
		t.Fatalf("Metadata path = %q, want %q", paths.Metadata, filepath.Join("repo", ".ctxpp", "ann-hnsw.json"))
	}
}

func TestSearchSemantic_ANNModeLoadsHNSWArtifactsWhenPresent(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	seedStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, seedStore,
		types.FileRecord{Path: "load.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "load.go:Only:function", File: "load.go", Name: "Only", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	if err := seedStore.UpsertEmbedding("load.go:Only:function", "test-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding() error = %v", err)
	}
	var rowID int64
	if err := seedStore.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "load.go:Only:function").Scan(&rowID); err != nil {
		t.Fatalf("rowid lookup error = %v", err)
	}
	paths := annArtifactPaths(dbPath)
	if err := os.MkdirAll(filepath.Dir(paths.Index), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	meta := annMetadata{FormatVersion: annFormatVersion, Engine: annEngineHNSW, Model: "test-model", Dims: 3, Count: 1}
	if err := writeANNMetadata(paths.Metadata, meta); err != nil {
		t.Fatalf("writeANNMetadata() error = %v", err)
	}
	g := hnsw.NewGraph[int64]()
	g.Add(hnsw.MakeNode(rowID, []float32{1, 0, 0}))
	f, err := os.Create(paths.Index)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := g.Export(f); err != nil {
		_ = f.Close()
		t.Fatalf("Export() error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	if st.semanticMode != SemanticModeANN {
		t.Fatalf("semanticMode = %q, want %q", st.semanticMode, SemanticModeANN)
	}
	if _, ok := st.semanticSearcher.(*hnswSemanticSearcher); !ok {
		t.Fatalf("semanticSearcher type = %T, want *hnswSemanticSearcher", st.semanticSearcher)
	}
}

func TestSearchSemantic_ANNModeSearchesUsingHNSWArtifact(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	paths := annArtifactPaths(dbPath)
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "ann.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "ann.go:Near:function", File: "ann.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "ann.go:Far:function", File: "ann.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)
	if err := st.UpsertEmbedding("ann.go:Near:function", "m", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(near) error = %v", err)
	}
	if err := st.UpsertEmbedding("ann.go:Far:function", "m", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(far) error = %v", err)
	}

	var nearRowID, farRowID int64
	if err := st.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "ann.go:Near:function").Scan(&nearRowID); err != nil {
		t.Fatalf("near rowid lookup error = %v", err)
	}
	if err := st.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "ann.go:Far:function").Scan(&farRowID); err != nil {
		t.Fatalf("far rowid lookup error = %v", err)
	}

	g := hnsw.NewGraph[int64]()
	g.Add(
		hnsw.MakeNode(nearRowID, []float32{1, 0, 0}),
		hnsw.MakeNode(farRowID, []float32{0, 0, 1}),
	)
	if err := os.MkdirAll(filepath.Dir(paths.Index), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	f, err := os.Create(paths.Index)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := g.Export(f); err != nil {
		_ = f.Close()
		t.Fatalf("Export() error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := writeANNMetadata(paths.Metadata, annMetadata{FormatVersion: annFormatVersion, Engine: annEngineHNSW}); err != nil {
		t.Fatalf("writeANNMetadata() error = %v", err)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Near" {
		t.Fatalf("top result = %q, want Near", got[0].Name)
	}
}

func TestBuildHNSWArtifactsFromEmbeddings_WritesSearchableIndexAndMetadata(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "build.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "build.go:Near:function", File: "build.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "build.go:Far:function", File: "build.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)
	if err := st.UpsertEmbedding("build.go:Near:function", "test-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(near) error = %v", err)
	}
	if err := st.UpsertEmbedding("build.go:Far:function", "test-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(far) error = %v", err)
	}

	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	paths := annArtifactPaths(dbPath)
	if _, err := os.Stat(paths.Index); err != nil {
		t.Fatalf("Stat(index) error = %v", err)
	}
	meta, err := readANNMetadata(paths.Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.FormatVersion != annFormatVersion {
		t.Fatalf("FormatVersion = %d, want %d", meta.FormatVersion, annFormatVersion)
	}
	if meta.Engine != annEngineHNSW {
		t.Fatalf("Engine = %q, want %q", meta.Engine, annEngineHNSW)
	}
	if meta.Model != "test-model" {
		t.Fatalf("Model = %q, want test-model", meta.Model)
	}
	if meta.Dims != 3 {
		t.Fatalf("Dims = %d, want 3", meta.Dims)
	}
	if meta.Count != 2 {
		t.Fatalf("Count = %d, want 2", meta.Count)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Near" {
		t.Fatalf("top result = %q, want Near", got[0].Name)
	}
}

func TestSearchSemantic_ANNModeRebuildsArtifactsWhenMetadataIsStale(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "rebuild.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "rebuild.go:Near:function", File: "rebuild.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "rebuild.go:Far:function", File: "rebuild.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)
	if err := st.UpsertEmbedding("rebuild.go:Near:function", "fresh-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(near) error = %v", err)
	}
	if err := st.UpsertEmbedding("rebuild.go:Far:function", "fresh-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(far) error = %v", err)
	}

	paths := annArtifactPaths(dbPath)
	if err := os.MkdirAll(filepath.Dir(paths.Index), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	g := hnsw.NewGraph[int64]()
	g.Add(hnsw.MakeNode(int64(1), []float32{1, 0, 0}))
	f, err := os.Create(paths.Index)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := g.Export(f); err != nil {
		_ = f.Close()
		t.Fatalf("Export() error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := writeANNMetadata(paths.Metadata, annMetadata{FormatVersion: annFormatVersion, Engine: annEngineHNSW, Model: "stale-model", Dims: 3, Count: 1}); err != nil {
		t.Fatalf("writeANNMetadata() error = %v", err)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	if annStore.semanticMode != SemanticModeANN {
		t.Fatalf("semanticMode = %q, want %q", annStore.semanticMode, SemanticModeANN)
	}
	meta, err := readANNMetadata(paths.Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.Model != "fresh-model" {
		t.Fatalf("Model = %q, want fresh-model", meta.Model)
	}
	if meta.Count != 2 {
		t.Fatalf("Count = %d, want 2", meta.Count)
	}
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Near" {
		t.Fatalf("top result = %q, want Near", got[0].Name)
	}
}

func TestUpsertEmbedding_RebuildsExistingANNArtifacts(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "sync.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "sync.go:Near:function", File: "sync.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "sync.go:Far:function", File: "sync.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			{ID: "sync.go:New:function", File: "sync.go", Name: "New", Kind: types.KindFunction, StartLine: 5, EndLine: 6},
		},
	)
	if err := st.UpsertEmbedding("sync.go:Near:function", "sync-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(near) error = %v", err)
	}
	if err := st.UpsertEmbedding("sync.go:Far:function", "sync-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(far) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	if err := st.UpsertEmbedding("sync.go:New:function", "sync-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(new) error = %v", err)
	}

	meta, err := readANNMetadata(annArtifactPaths(dbPath).Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.Count != 3 {
		t.Fatalf("Count = %d, want 3", meta.Count)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Near" && got[0].Name != "New" {
		t.Fatalf("top result = %q, want Near or New after ANN sync", got[0].Name)
	}
	if got[1].Name != "Near" && got[1].Name != "New" {
		t.Fatalf("second result = %q, want Near or New after ANN sync", got[1].Name)
	}
}

func TestDeleteFile_RebuildsExistingANNArtifacts(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "keep.go", SHA256: "a", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "keep.go:Keep:function", File: "keep.go", Name: "Keep", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	seedFile(t, st,
		types.FileRecord{Path: "drop.go", SHA256: "b", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "drop.go:Drop:function", File: "drop.go", Name: "Drop", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	if err := st.UpsertEmbedding("keep.go:Keep:function", "sync-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(keep) error = %v", err)
	}
	if err := st.UpsertEmbedding("drop.go:Drop:function", "sync-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(drop) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	if err := st.DeleteFile("drop.go"); err != nil {
		t.Fatalf("DeleteFile() error = %v", err)
	}

	meta, err := readANNMetadata(annArtifactPaths(dbPath).Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.Count != 1 {
		t.Fatalf("Count = %d, want 1", meta.Count)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SearchSemantic() returned %d results, want 1", len(got))
	}
	if got[0].Name != "Keep" {
		t.Fatalf("top result = %q, want Keep", got[0].Name)
	}
}

func TestDeleteSymbolsByFile_UpdatesANNMetadataAndResults(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "gone.go", SHA256: "a", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "gone.go:Gone:function", File: "gone.go", Name: "Gone", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	seedFile(t, st,
		types.FileRecord{Path: "stay.go", SHA256: "b", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "stay.go:Stay:function", File: "stay.go", Name: "Stay", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	if err := st.UpsertEmbedding("gone.go:Gone:function", "delete-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(gone) error = %v", err)
	}
	if err := st.UpsertEmbedding("stay.go:Stay:function", "delete-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(stay) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	if err := st.DeleteSymbolsByFile("gone.go"); err != nil {
		t.Fatalf("DeleteSymbolsByFile() error = %v", err)
	}

	meta, err := readANNMetadata(annArtifactPaths(dbPath).Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.Count != 1 {
		t.Fatalf("Count = %d, want 1", meta.Count)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SearchSemantic() returned %d results, want 1", len(got))
	}
	if got[0].Name != "Stay" {
		t.Fatalf("top result = %q, want Stay", got[0].Name)
	}
}

func TestSearchSemantic_AutoModeUsesANNWhenArtifactsAreValid(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "auto.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "auto.go:Near:function", File: "auto.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "auto.go:Far:function", File: "auto.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)
	if err := st.UpsertEmbedding("auto.go:Near:function", "auto-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(near) error = %v", err)
	}
	if err := st.UpsertEmbedding("auto.go:Far:function", "auto-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(far) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	autoStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeAuto})
	if autoStore.semanticMode != SemanticModeANN {
		t.Fatalf("semanticMode = %q, want %q", autoStore.semanticMode, SemanticModeANN)
	}
	if _, ok := autoStore.semanticSearcher.(*hnswSemanticSearcher); !ok {
		t.Fatalf("semanticSearcher type = %T, want *hnswSemanticSearcher", autoStore.semanticSearcher)
	}
	got, err := autoStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Near" {
		t.Fatalf("top result = %q, want Near", got[0].Name)
	}
}

func TestDeferredANNSync_BatchesArtifactRefreshUntilEnd(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "defer.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "defer.go:Old:function", File: "defer.go", Name: "Old", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "defer.go:New:function", File: "defer.go", Name: "New", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)
	if err := st.UpsertEmbedding("defer.go:Old:function", "defer-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(old) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	st.BeginDeferredANNSync()
	if err := st.UpsertEmbedding("defer.go:New:function", "defer-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(new) error = %v", err)
	}

	meta, err := readANNMetadata(annArtifactPaths(dbPath).Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.Count != 1 {
		t.Fatalf("Count = %d, want 1 before deferred sync flush", meta.Count)
	}
	if err := st.EndDeferredANNSync(); err != nil {
		t.Fatalf("EndDeferredANNSync() error = %v", err)
	}

	meta, err = readANNMetadata(annArtifactPaths(dbPath).Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() after flush error = %v", err)
	}
	if meta.Count != 2 {
		t.Fatalf("Count = %d, want 2 after deferred sync flush", meta.Count)
	}
}

func TestDeferredANNSync_FlushesIncrementalDeletesAndUpserts(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "flush.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "flush.go:Old:function", File: "flush.go", Name: "Old", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "flush.go:Keep:function", File: "flush.go", Name: "Keep", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			{ID: "flush.go:New:function", File: "flush.go", Name: "New", Kind: types.KindFunction, StartLine: 5, EndLine: 6},
		},
	)
	if err := st.UpsertEmbedding("flush.go:Old:function", "flush-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(old) error = %v", err)
	}
	if err := st.UpsertEmbedding("flush.go:Keep:function", "flush-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(keep) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	st.BeginDeferredANNSync()
	if err := st.DeleteSymbolsByFile("flush.go"); err != nil {
		t.Fatalf("DeleteSymbolsByFile() error = %v", err)
	}
	if err := st.UpsertSymbolsBatch([]types.Symbol{
		{ID: "flush.go:Keep:function", File: "flush.go", Name: "Keep", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		{ID: "flush.go:New:function", File: "flush.go", Name: "New", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
	}); err != nil {
		t.Fatalf("UpsertSymbolsBatch() error = %v", err)
	}
	if err := st.UpsertEmbedding("flush.go:Keep:function", "flush-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(keep) error = %v", err)
	}
	if err := st.UpsertEmbedding("flush.go:New:function", "flush-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(new) error = %v", err)
	}
	if err := st.EndDeferredANNSync(); err != nil {
		t.Fatalf("EndDeferredANNSync() error = %v", err)
	}

	meta, err := readANNMetadata(annArtifactPaths(dbPath).Metadata)
	if err != nil {
		t.Fatalf("readANNMetadata() error = %v", err)
	}
	if meta.Count != 2 {
		t.Fatalf("Count = %d, want 2", meta.Count)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 3)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Keep" && got[0].Name != "New" {
		t.Fatalf("top result = %q, want Keep or New", got[0].Name)
	}
	if got[1].Name != "Keep" && got[1].Name != "New" {
		t.Fatalf("second result = %q, want Keep or New", got[1].Name)
	}
}

func TestUpsertParsedFile_RebuildsExistingANNArtifacts(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "parsed.go", SHA256: "v1", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "parsed.go:Old:func", File: "parsed.go", Name: "Old", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	if err := st.UpsertEmbedding("parsed.go:Old:func", "parsed-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(old) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	pf := ParsedFileData{
		File:    types.FileRecord{Path: "parsed.go", SHA256: "v2", ModTime: 2, Lang: "go"},
		Symbols: []types.Symbol{{ID: "parsed.go:New:func", File: "parsed.go", Name: "New", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	}
	if err := st.UpsertParsedFile(pf); err != nil {
		t.Fatalf("UpsertParsedFile() error = %v", err)
	}

	paths := annArtifactPaths(dbPath)
	if _, err := os.Stat(paths.Metadata); !os.IsNotExist(err) {
		t.Fatalf("Stat(metadata) error = %v, want not exists", err)
	}
	if err := st.UpsertEmbedding("parsed.go:New:func", "parsed-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(new) error = %v", err)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SearchSemantic() returned %d results, want 1", len(got))
	}
	if got[0].Name != "New" {
		t.Fatalf("top result = %q, want New", got[0].Name)
	}
}

func TestUpsertParsedFileBatch_RebuildsExistingANNArtifacts(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), ".ctxpp", "index.db")
	st := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeBruteForce})
	seedFile(t, st,
		types.FileRecord{Path: "batch.go", SHA256: "v1", ModTime: 1, Lang: "go"},
		[]types.Symbol{{ID: "batch.go:Old:func", File: "batch.go", Name: "Old", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	)
	if err := st.UpsertEmbedding("batch.go:Old:func", "batch-model", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(old) error = %v", err)
	}
	if err := st.BuildANNArtifacts(); err != nil {
		t.Fatalf("BuildANNArtifacts() error = %v", err)
	}

	files := []ParsedFileData{{
		File:    types.FileRecord{Path: "batch.go", SHA256: "v2", ModTime: 2, Lang: "go"},
		Symbols: []types.Symbol{{ID: "batch.go:Fresh:func", File: "batch.go", Name: "Fresh", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	}}
	if err := st.UpsertParsedFileBatch(files); err != nil {
		t.Fatalf("UpsertParsedFileBatch() error = %v", err)
	}

	paths := annArtifactPaths(dbPath)
	if _, err := os.Stat(paths.Metadata); !os.IsNotExist(err) {
		t.Fatalf("Stat(metadata) error = %v, want not exists", err)
	}
	if err := st.UpsertEmbedding("batch.go:Fresh:func", "batch-model", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(fresh) error = %v", err)
	}

	annStore := openTestStoreAtPath(t, dbPath, OpenOptions{SemanticMode: SemanticModeANN})
	got, err := annStore.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SearchSemantic() returned %d results, want 1", len(got))
	}
	if got[0].Name != "Fresh" {
		t.Fatalf("top result = %q, want Fresh", got[0].Name)
	}
}

func TestSearchSemantic_UsesConfiguredSearcher(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "cfg.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "cfg.go:Alpha:function", File: "cfg.go", Name: "Alpha", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "cfg.go:Beta:function", File: "cfg.go", Name: "Beta", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	var alphaRowID, betaRowID int64
	if err := st.UpsertEmbedding("cfg.go:Alpha:function", "m", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(alpha) error = %v", err)
	}
	if err := st.UpsertEmbedding("cfg.go:Beta:function", "m", []float32{0, 1, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(beta) error = %v", err)
	}
	if err := st.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "cfg.go:Alpha:function").Scan(&alphaRowID); err != nil {
		t.Fatalf("rowid alpha lookup error = %v", err)
	}
	if err := st.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "cfg.go:Beta:function").Scan(&betaRowID); err != nil {
		t.Fatalf("rowid beta lookup error = %v", err)
	}

	st.semanticSearcher = stubSemanticSearcher{candidates: []semanticCandidate{
		{rowid: betaRowID, score: 0.95},
	}}

	got, err := st.SearchSemantic([]float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SearchSemantic() returned %d results, want 1", len(got))
	}
	if got[0].Name != "Beta" {
		t.Fatalf("top result = %q, want Beta", got[0].Name)
	}
}

func TestSearchSemantic_RecomputesExactScoresAfterCandidateGeneration(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "rerank.go", SHA256: "q", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "rerank.go:Near:function", File: "rerank.go", Name: "Near", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "rerank.go:Far:function", File: "rerank.go", Name: "Far", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	query := []float32{1, 0, 0}
	if err := st.UpsertEmbedding("rerank.go:Near:function", "m", []float32{1, 0, 0}); err != nil {
		t.Fatalf("UpsertEmbedding(near) error = %v", err)
	}
	if err := st.UpsertEmbedding("rerank.go:Far:function", "m", []float32{0, 0, 1}); err != nil {
		t.Fatalf("UpsertEmbedding(far) error = %v", err)
	}

	var nearRowID, farRowID int64
	if err := st.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "rerank.go:Near:function").Scan(&nearRowID); err != nil {
		t.Fatalf("rowid near lookup error = %v", err)
	}
	if err := st.rdb.QueryRow(`SELECT rowid FROM embeddings WHERE symbol_id=?`, "rerank.go:Far:function").Scan(&farRowID); err != nil {
		t.Fatalf("rowid far lookup error = %v", err)
	}

	st.semanticSearcher = stubSemanticSearcher{candidates: []semanticCandidate{
		{rowid: farRowID, score: 0.99},
		{rowid: nearRowID, score: 0.01},
	}}

	got, err := st.SearchSemantic(query, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "Near" {
		t.Fatalf("top result = %q, want Near after exact rerank", got[0].Name)
	}
	if got[1].Name != "Far" {
		t.Fatalf("second result = %q, want Far after exact rerank", got[1].Name)
	}
}

func TestSearchKeyword_FindsByName(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "c.go", SHA256: "z", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "c.go:ParseConfig:function", File: "c.go", Name: "ParseConfig",
				Kind: types.KindFunction, Signature: "func ParseConfig(path string) error",
				StartLine: 1, EndLine: 10},
			{ID: "c.go:WriteOutput:function", File: "c.go", Name: "WriteOutput",
				Kind: types.KindFunction, Signature: "func WriteOutput(w io.Writer) error",
				StartLine: 11, EndLine: 20},
		},
	)

	tests := []struct {
		name      string
		query     string
		wantCount int
		wantName  string
	}{
		{name: "exact name", query: "ParseConfig", wantCount: 1, wantName: "ParseConfig"},
		{name: "no match", query: "DefinitelyAbsent9999", wantCount: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := st.SearchKeyword(tc.query, 10)
			if err != nil {
				t.Fatalf("SearchKeyword() error = %v", err)
			}
			if len(got) != tc.wantCount {
				t.Fatalf("SearchKeyword() len = %d, want %d", len(got), tc.wantCount)
			}
			if tc.wantCount > 0 && got[0].Name != tc.wantName {
				t.Errorf("result[0].Name = %q, want %q", got[0].Name, tc.wantName)
			}
		})
	}
}

func TestDeleteFile_CascadesToSymbols(t *testing.T) {
	st := openTestStore(t)

	f := types.FileRecord{Path: "b.go", SHA256: "x", ModTime: 1, Lang: "go"}
	if err := st.UpsertFile(f); err != nil {
		t.Fatalf("UpsertFile() error = %v", err)
	}
	sym := types.Symbol{
		ID: "b.go:Bar:function", File: "b.go", Name: "Bar",
		Kind: types.KindFunction, StartLine: 1, EndLine: 5,
	}
	if err := st.UpsertSymbol(sym); err != nil {
		t.Fatalf("UpsertSymbol() error = %v", err)
	}

	if err := st.DeleteFile("b.go"); err != nil {
		t.Fatalf("DeleteFile() error = %v", err)
	}

	syms, err := st.GetSymbolsByFile("b.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) != 0 {
		t.Errorf("GetSymbolsByFile() after DeleteFile = %d symbols, want 0", len(syms))
	}
}

func TestUpsertFile_UpdatesExisting(t *testing.T) {
	st := openTestStore(t)

	base := types.FileRecord{Path: "a.go", SHA256: "old", ModTime: 1, Lang: "go"}
	if err := st.UpsertFile(base); err != nil {
		t.Fatalf("first UpsertFile() error = %v", err)
	}
	updated := types.FileRecord{Path: "a.go", SHA256: "new", ModTime: 2, Lang: "go"}
	if err := st.UpsertFile(updated); err != nil {
		t.Fatalf("second UpsertFile() error = %v", err)
	}

	got, err := st.GetFileSHA("a.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if got != "new" {
		t.Errorf("GetFileSHA() = %q after update, want %q", got, "new")
	}
}

func TestGetFileSHA_MissingReturnsEmpty(t *testing.T) {
	st := openTestStore(t)

	got, err := st.GetFileSHA("does/not/exist.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if got != "" {
		t.Errorf("GetFileSHA() = %q for missing file, want empty string", got)
	}
}

func TestGetFileSHA_ReturnsStoredHash(t *testing.T) {
	st := openTestStore(t)

	err := st.UpsertFile(types.FileRecord{
		Path: "pkg/foo/foo.go", SHA256: "abc123", ModTime: 1000, Lang: "go",
	})
	if err != nil {
		t.Fatalf("UpsertFile() error = %v", err)
	}

	got, err := st.GetFileSHA("pkg/foo/foo.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if got != "abc123" {
		t.Errorf("GetFileSHA() = %q, want %q", got, "abc123")
	}
}

func TestListFiles_Empty(t *testing.T) {
	st := openTestStore(t)

	got, err := st.ListFiles()
	if err != nil {
		t.Fatalf("ListFiles() error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("ListFiles() len = %d, want 0", len(got))
	}
}

func TestListFiles_ReturnsIndexedPaths(t *testing.T) {
	st := openTestStore(t)

	files := []types.FileRecord{
		{Path: "a.go", SHA256: "sha-a", ModTime: 1, Lang: "go"},
		{Path: "pkg/b.go", SHA256: "sha-b", ModTime: 2, Lang: "go"},
		{Path: "cmd/main.go", SHA256: "sha-c", ModTime: 3, Lang: "go"},
	}
	for _, f := range files {
		if err := st.UpsertFile(f); err != nil {
			t.Fatalf("UpsertFile(%q) error = %v", f.Path, err)
		}
	}

	got, err := st.ListFiles()
	if err != nil {
		t.Fatalf("ListFiles() error = %v", err)
	}
	if len(got) != len(files) {
		t.Fatalf("ListFiles() len = %d, want %d", len(got), len(files))
	}

	wantSet := make(map[string]struct{}, len(files))
	for _, f := range files {
		wantSet[f.Path] = struct{}{}
	}
	for _, p := range got {
		if _, ok := wantSet[p]; !ok {
			t.Fatalf("ListFiles() unexpected path %q", p)
		}
		delete(wantSet, p)
	}
	if len(wantSet) != 0 {
		t.Fatalf("ListFiles() missing paths: %v", wantSet)
	}
}

func TestGetSymbolsByNames_ReturnsMatchingSymbols(t *testing.T) {
	st := openTestStore(t)

	f := types.FileRecord{Path: "names.go", SHA256: "x", ModTime: 1, Lang: "go"}
	syms := []types.Symbol{
		{ID: "names.go:Foo:func", File: "names.go", Name: "Foo", Kind: types.KindFunction, StartLine: 1, EndLine: 3},
		{ID: "names.go:Bar:func", File: "names.go", Name: "Bar", Kind: types.KindFunction, StartLine: 5, EndLine: 7},
		{ID: "names.go:Baz:func", File: "names.go", Name: "Baz", Kind: types.KindFunction, StartLine: 9, EndLine: 11},
	}
	seedFile(t, st, f, syms)

	got, err := st.GetSymbolsByNames([]string{"Foo", "Baz"})
	if err != nil {
		t.Fatalf("GetSymbolsByNames() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("GetSymbolsByNames() returned %d, want 2", len(got))
	}
	names := make(map[string]bool)
	for _, s := range got {
		names[s.Name] = true
	}
	if !names["Foo"] || !names["Baz"] {
		t.Errorf("GetSymbolsByNames() = %v, want Foo and Baz", got)
	}
}

func TestGetSymbolsByNames_EmptySlice(t *testing.T) {
	st := openTestStore(t)
	got, err := st.GetSymbolsByNames(nil)
	if err != nil {
		t.Fatalf("GetSymbolsByNames(nil) error = %v", err)
	}
	if got != nil {
		t.Errorf("GetSymbolsByNames(nil) = %v, want nil", got)
	}
}

// ---- GetSymbol tests -------------------------------------------------------

func TestGetSymbol_Found(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "gs.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "gs.go:Foo:func", File: "gs.go", Name: "Foo", Kind: types.KindFunction,
				Signature: "func Foo()", DocComment: "Foo does things",
				StartLine: 1, EndLine: 3, Receiver: "", Package: "gs"},
		},
	)

	got, err := st.GetSymbol("gs.go:Foo:func")
	if err != nil {
		t.Fatalf("GetSymbol() error = %v", err)
	}
	if got == nil {
		t.Fatal("GetSymbol() = nil, want symbol")
	}
	if got.Name != "Foo" {
		t.Errorf("GetSymbol().Name = %q, want %q", got.Name, "Foo")
	}
	if got.Kind != types.KindFunction {
		t.Errorf("GetSymbol().Kind = %q, want %q", got.Kind, types.KindFunction)
	}
	if got.Package != "gs" {
		t.Errorf("GetSymbol().Package = %q, want %q", got.Package, "gs")
	}
}

func TestGetSymbol_NotFound(t *testing.T) {
	st := openTestStore(t)
	got, err := st.GetSymbol("nonexistent")
	if err != nil {
		t.Fatalf("GetSymbol() error = %v", err)
	}
	if got != nil {
		t.Errorf("GetSymbol(nonexistent) = %v, want nil", got)
	}
}

// ---- DeleteSymbolsByFile tests ---------------------------------------------

func TestDeleteSymbolsByFile_RemovesSymbols(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "del.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "del.go:A:func", File: "del.go", Name: "A", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "del.go:B:func", File: "del.go", Name: "B", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	if err := st.DeleteSymbolsByFile("del.go"); err != nil {
		t.Fatalf("DeleteSymbolsByFile() error = %v", err)
	}

	got, err := st.GetSymbolsByFile("del.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("GetSymbolsByFile() after delete = %d symbols, want 0", len(got))
	}
}

func TestDeleteSymbolsByFile_NoopForMissingFile(t *testing.T) {
	st := openTestStore(t)
	if err := st.DeleteSymbolsByFile("nonexistent.go"); err != nil {
		t.Fatalf("DeleteSymbolsByFile(missing) error = %v", err)
	}
}

// ---- SearchHybrid tests ----------------------------------------------------

func TestSearchHybrid_CombinesKeywordAndSemantic(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "hybrid.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "hybrid.go:Alpha:func", File: "hybrid.go", Name: "Alpha", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "hybrid.go:Beta:func", File: "hybrid.go", Name: "Beta", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			{ID: "hybrid.go:Gamma:func", File: "hybrid.go", Name: "Gamma", Kind: types.KindFunction, StartLine: 5, EndLine: 6},
		},
	)

	// Add embeddings: Alpha and Gamma have non-zero vectors, Beta is zero.
	if err := st.UpsertEmbedding("hybrid.go:Alpha:func", "m", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := st.UpsertEmbedding("hybrid.go:Beta:func", "m", []float32{0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := st.UpsertEmbedding("hybrid.go:Gamma:func", "m", []float32{0.9, 0.1, 0}); err != nil {
		t.Fatal(err)
	}

	// Keyword matches "Alpha", semantic should find Alpha+Gamma via [1,0,0] query.
	queryVec := []float32{1, 0, 0}
	got, err := st.SearchHybrid(queryVec, "Alpha", 10)
	if err != nil {
		t.Fatalf("SearchHybrid() error = %v", err)
	}
	if len(got) == 0 {
		t.Fatal("SearchHybrid() returned no results")
	}

	// Alpha should be present (from keyword hit).
	found := make(map[string]bool)
	for _, s := range got {
		found[s.Name] = true
	}
	if !found["Alpha"] {
		t.Error("SearchHybrid() missing Alpha (expected from keyword)")
	}
}

func TestSearchHybrid_DeduplicatesResults(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "dup.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "dup.go:Foo:func", File: "dup.go", Name: "Foo", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		},
	)
	if err := st.UpsertEmbedding("dup.go:Foo:func", "m", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}

	// Foo should appear in both keyword and semantic results but be deduplicated.
	got, err := st.SearchHybrid([]float32{1, 0, 0}, "Foo", 10)
	if err != nil {
		t.Fatalf("SearchHybrid() error = %v", err)
	}

	count := 0
	for _, s := range got {
		if s.Name == "Foo" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("Foo appears %d times, want 1 (deduplication)", count)
	}
}

func TestSearchHybrid_RespectsLimit(t *testing.T) {
	st := openTestStore(t)
	f := types.FileRecord{Path: "lim.go", SHA256: "x", ModTime: 1, Lang: "go"}
	var syms []types.Symbol
	for i := 0; i < 20; i++ {
		name := "Sym" + strconv.Itoa(i)
		syms = append(syms, types.Symbol{
			ID: "lim.go:" + name + ":func", File: "lim.go", Name: name,
			Kind: types.KindFunction, StartLine: i*2 + 1, EndLine: i*2 + 2,
		})
	}
	seedFile(t, st, f, syms)
	for _, sym := range syms {
		if err := st.UpsertEmbedding(sym.ID, "m", []float32{1, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}

	got, err := st.SearchHybrid([]float32{1, 0, 0}, "Sym0", 5)
	if err != nil {
		t.Fatalf("SearchHybrid() error = %v", err)
	}
	if len(got) > 5 {
		t.Errorf("SearchHybrid() returned %d results, want <= 5", len(got))
	}
}

// ---- GetSymbolsByIDs tests -------------------------------------------------

func TestGetSymbolsByIDs_ReturnsMatchingSymbols(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "ids.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "ids.go:A:func", File: "ids.go", Name: "A", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "ids.go:B:func", File: "ids.go", Name: "B", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			{ID: "ids.go:C:func", File: "ids.go", Name: "C", Kind: types.KindFunction, StartLine: 5, EndLine: 6},
		},
	)

	got, err := st.GetSymbolsByIDs([]string{"ids.go:A:func", "ids.go:C:func"})
	if err != nil {
		t.Fatalf("GetSymbolsByIDs() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("GetSymbolsByIDs() returned %d, want 2", len(got))
	}
	names := make(map[string]bool)
	for _, s := range got {
		names[s.Name] = true
	}
	if !names["A"] || !names["C"] {
		t.Errorf("GetSymbolsByIDs() = %v, want A and C", got)
	}
}

func TestGetSymbolsByIDs_EmptySlice(t *testing.T) {
	st := openTestStore(t)
	got, err := st.GetSymbolsByIDs(nil)
	if err != nil {
		t.Fatalf("GetSymbolsByIDs(nil) error = %v", err)
	}
	if got != nil {
		t.Errorf("GetSymbolsByIDs(nil) = %v, want nil", got)
	}
}

func TestGetSymbolsByIDs_PartialMatch(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "pm.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "pm.go:X:func", File: "pm.go", Name: "X", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		},
	)

	got, err := st.GetSymbolsByIDs([]string{"pm.go:X:func", "nonexistent:id"})
	if err != nil {
		t.Fatalf("GetSymbolsByIDs() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("GetSymbolsByIDs() returned %d, want 1 (partial match)", len(got))
	}
	if got[0].Name != "X" {
		t.Errorf("GetSymbolsByIDs()[0].Name = %q, want %q", got[0].Name, "X")
	}
}

// ---- UpsertImportEdges tests -----------------------------------------------

func TestUpsertImportEdges_StoresAndReplaces(t *testing.T) {
	st := openTestStore(t)

	first := []types.ImportEdge{
		{ImporterFile: "main.go", ImportedPath: "fmt"},
		{ImporterFile: "main.go", ImportedPath: "os"},
	}
	if err := st.UpsertImportEdges("main.go", first); err != nil {
		t.Fatalf("UpsertImportEdges() error = %v", err)
	}

	// Verify by querying directly.
	rows, err := st.DB().Query(`SELECT imported_path FROM import_edges WHERE importer_file='main.go' ORDER BY imported_path`)
	if err != nil {
		t.Fatal(err)
	}
	var paths []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, p)
	}
	rows.Close()
	if len(paths) != 2 || paths[0] != "fmt" || paths[1] != "os" {
		t.Errorf("import edges = %v, want [fmt os]", paths)
	}

	// Replace with different imports.
	second := []types.ImportEdge{
		{ImporterFile: "main.go", ImportedPath: "net/http"},
	}
	if err := st.UpsertImportEdges("main.go", second); err != nil {
		t.Fatalf("UpsertImportEdges() replace error = %v", err)
	}

	rows2, err := st.DB().Query(`SELECT imported_path FROM import_edges WHERE importer_file='main.go'`)
	if err != nil {
		t.Fatal(err)
	}
	var paths2 []string
	for rows2.Next() {
		var p string
		if err := rows2.Scan(&p); err != nil {
			t.Fatal(err)
		}
		paths2 = append(paths2, p)
	}
	rows2.Close()
	if len(paths2) != 1 || paths2[0] != "net/http" {
		t.Errorf("import edges after replace = %v, want [net/http]", paths2)
	}
}

func TestUpsertImportEdges_EmptyEdges(t *testing.T) {
	st := openTestStore(t)
	// Upserting empty edges should just clear existing.
	if err := st.UpsertImportEdges("x.go", nil); err != nil {
		t.Fatalf("UpsertImportEdges(nil) error = %v", err)
	}
}

// ---- SymbolIDsWithoutEmbeddings tests --------------------------------------

func TestSymbolIDsWithoutEmbeddings_FindsMissing(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "emb.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "emb.go:Has:func", File: "emb.go", Name: "Has", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "emb.go:Missing:func", File: "emb.go", Name: "Missing", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	// Only embed "Has".
	if err := st.UpsertEmbedding("emb.go:Has:func", "m", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}

	got, err := st.SymbolIDsWithoutEmbeddings()
	if err != nil {
		t.Fatalf("SymbolIDsWithoutEmbeddings() error = %v", err)
	}
	if len(got) != 1 || got[0] != "emb.go:Missing:func" {
		t.Errorf("SymbolIDsWithoutEmbeddings() = %v, want [emb.go:Missing:func]", got)
	}
}

func TestSymbolIDsWithoutEmbeddings_AllEmbedded(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "all.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "all.go:A:func", File: "all.go", Name: "A", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		},
	)
	if err := st.UpsertEmbedding("all.go:A:func", "m", []float32{1}); err != nil {
		t.Fatal(err)
	}

	got, err := st.SymbolIDsWithoutEmbeddings()
	if err != nil {
		t.Fatalf("SymbolIDsWithoutEmbeddings() error = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("SymbolIDsWithoutEmbeddings() = %v, want empty", got)
	}
}

// ---- cosineSimilarity tests ------------------------------------------------

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{name: "identical", a: []float32{1, 0, 0}, b: []float32{1, 0, 0}, want: 1.0},
		{name: "orthogonal", a: []float32{1, 0, 0}, b: []float32{0, 1, 0}, want: 0.0},
		{name: "opposite", a: []float32{1, 0, 0}, b: []float32{-1, 0, 0}, want: -1.0},
		{name: "empty", a: []float32{}, b: []float32{}, want: 0},
		{name: "mismatched len", a: []float32{1, 0}, b: []float32{1, 0, 0}, want: 0},
		{name: "zero vector", a: []float32{0, 0, 0}, b: []float32{1, 0, 0}, want: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := cosineSimilarity(tc.a, tc.b)
			if diff := got - tc.want; diff > 0.001 || diff < -0.001 {
				t.Errorf("cosineSimilarity() = %v, want %v", got, tc.want)
			}
		})
	}
}

// ---- decodeFloat32s tests --------------------------------------------------

func TestDecodeFloat32s_RoundTrips(t *testing.T) {
	original := []float32{1.5, -2.3, 0.0, 42.0, 0.001}
	blob := encodeFloat32s(original)
	decoded := decodeFloat32s(blob)

	if len(decoded) != len(original) {
		t.Fatalf("decodeFloat32s() len = %d, want %d", len(decoded), len(original))
	}
	for i := range original {
		if decoded[i] != original[i] {
			t.Errorf("decodeFloat32s()[%d] = %v, want %v", i, decoded[i], original[i])
		}
	}
}

func TestDecodeFloat32s_EmptyBlob(t *testing.T) {
	decoded := decodeFloat32s(nil)
	if len(decoded) != 0 {
		t.Errorf("decodeFloat32s(nil) len = %d, want 0", len(decoded))
	}
}

// ---- sortScored tests ------------------------------------------------------

func TestSortScored_SortsDescending(t *testing.T) {
	input := []scoredSymbol{
		{sym: types.Symbol{Name: "Low"}, score: 0.1},
		{sym: types.Symbol{Name: "High"}, score: 0.9},
		{sym: types.Symbol{Name: "Mid"}, score: 0.5},
	}
	sortScored(input)

	if input[0].sym.Name != "High" || input[1].sym.Name != "Mid" || input[2].sym.Name != "Low" {
		t.Errorf("sortScored() = [%s %s %s], want [High Mid Low]",
			input[0].sym.Name, input[1].sym.Name, input[2].sym.Name)
	}
}

func TestSortScored_EmptySlice(t *testing.T) {
	sortScored(nil)              // should not panic
	sortScored([]scoredSymbol{}) // should not panic
}

func TestSortScored_SingleElement(t *testing.T) {
	input := []scoredSymbol{{sym: types.Symbol{Name: "Only"}, score: 0.5}}
	sortScored(input)
	if input[0].sym.Name != "Only" {
		t.Error("sortScored() changed single element")
	}
}

// ---- DB() test -------------------------------------------------------------

func TestDB_ReturnsNonNil(t *testing.T) {
	st := openTestStore(t)
	if st.DB() == nil {
		t.Error("DB() = nil, want non-nil *sql.DB")
	}
}

// ---- SearchKeyword default limit -------------------------------------------

func TestSearchKeyword_DefaultLimit(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "kw.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "kw.go:Target:func", File: "kw.go", Name: "Target", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		},
	)

	// limit=0 should default to 10 internally and still return results.
	got, err := st.SearchKeyword("Target", 0)
	if err != nil {
		t.Fatalf("SearchKeyword() error = %v", err)
	}
	if len(got) != 1 {
		t.Errorf("SearchKeyword(limit=0) returned %d results, want 1", len(got))
	}
}

// ---- SearchSemantic default limit ------------------------------------------

func TestSearchSemantic_DefaultLimit(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "sem.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "sem.go:A:func", File: "sem.go", Name: "A", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
		},
	)
	if err := st.UpsertEmbedding("sem.go:A:func", "m", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}

	got, err := st.SearchSemantic([]float32{1, 0, 0}, 0) // limit=0 defaults
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) != 1 {
		t.Errorf("SearchSemantic(limit=0) returned %d, want 1", len(got))
	}
}

func TestSearchSemantic_NoEmbeddings(t *testing.T) {
	st := openTestStore(t)
	got, err := st.SearchSemantic([]float32{1, 0, 0}, 10)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if got != nil {
		t.Errorf("SearchSemantic(empty) = %v, want nil", got)
	}
}

func TestSearchSemantic_TierWeightingDemotesLowSignal(t *testing.T) {
	st := openTestStore(t)

	// Code symbol with slightly lower cosine similarity.
	// Changelog symbol with higher cosine similarity but TierLowSignal penalty (0.5x).
	seedFile(t, st,
		types.FileRecord{Path: "core.go", SHA256: "a", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "core.go:RBACHandler:func", File: "core.go", Name: "RBACHandler",
				Kind: types.KindFunction, StartLine: 1, EndLine: 10, SourceTier: types.TierCode},
		},
	)
	seedFile(t, st,
		types.FileRecord{Path: "CHANGELOG.md", SHA256: "b", ModTime: 1, Lang: "md"},
		[]types.Symbol{
			{ID: "CHANGELOG.md:RBACSection:section", File: "CHANGELOG.md", Name: "RBACSection",
				Kind: types.KindSection, StartLine: 1, EndLine: 20, SourceTier: types.TierLowSignal},
		},
	)

	// Changelog embedding: very close to query (cosine ~0.99).
	// Code embedding: slightly less close (cosine ~0.87).
	// Raw scores: Changelog > Code. But after tier penalty:
	//   Changelog: 0.99 * 0.5 = 0.495
	//   Code:      0.87 * 1.0 = 0.87
	// So Code should rank first.
	query := []float32{1, 0, 0}
	if err := st.UpsertEmbedding("CHANGELOG.md:RBACSection:section", "m", []float32{0.99, 0.1, 0.1}); err != nil {
		t.Fatal(err)
	}
	if err := st.UpsertEmbedding("core.go:RBACHandler:func", "m", []float32{0.9, 0.3, 0.3}); err != nil {
		t.Fatal(err)
	}

	got, err := st.SearchSemantic(query, 2)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("SearchSemantic() returned %d results, want 2", len(got))
	}
	if got[0].Name != "RBACHandler" {
		t.Errorf("top result = %q, want RBACHandler (code symbol should rank above changelog after tier penalty)", got[0].Name)
	}
	if got[1].Name != "RBACSection" {
		t.Errorf("second result = %q, want RBACSection", got[1].Name)
	}
}

// ---- UpsertParsedFile tests ------------------------------------------------

func TestUpsertParsedFile_StoresAllData(t *testing.T) {
	st := openTestStore(t)

	pf := ParsedFileData{
		File: types.FileRecord{Path: "pf.go", SHA256: "abc", ModTime: 100, Lang: "go"},
		Symbols: []types.Symbol{
			{ID: "pf.go:Alpha:func", File: "pf.go", Name: "Alpha", Kind: types.KindFunction,
				Signature: "func Alpha()", StartLine: 1, EndLine: 3, Package: "pf"},
			{ID: "pf.go:Beta:func", File: "pf.go", Name: "Beta", Kind: types.KindFunction,
				Signature: "func Beta()", StartLine: 5, EndLine: 7, Package: "pf"},
		},
		CallEdges: []types.CallEdge{
			{CallerFile: "pf.go", CallerSymbol: "pf.go:Alpha:func", CalleeSymbol: "Beta", Line: 2},
		},
		ImportEdges: []types.ImportEdge{
			{ImporterFile: "pf.go", ImportedPath: "fmt"},
		},
	}

	if err := st.UpsertParsedFile(pf); err != nil {
		t.Fatalf("UpsertParsedFile() error = %v", err)
	}

	// Verify file record.
	sha, err := st.GetFileSHA("pf.go")
	if err != nil {
		t.Fatalf("GetFileSHA() error = %v", err)
	}
	if sha != "abc" {
		t.Errorf("GetFileSHA() = %q, want %q", sha, "abc")
	}

	// Verify symbols.
	syms, err := st.GetSymbolsByFile("pf.go")
	if err != nil {
		t.Fatalf("GetSymbolsByFile() error = %v", err)
	}
	if len(syms) != 2 {
		t.Fatalf("GetSymbolsByFile() len = %d, want 2", len(syms))
	}

	// Verify call edges.
	callees, err := st.CalleeSymbols("pf.go:Alpha:func")
	if err != nil {
		t.Fatalf("CalleeSymbols() error = %v", err)
	}
	if len(callees) != 1 || callees[0] != "Beta" {
		t.Errorf("CalleeSymbols() = %v, want [Beta]", callees)
	}

	// Verify import edges.
	rows, err := st.DB().Query(`SELECT imported_path FROM import_edges WHERE importer_file='pf.go'`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var imports []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			t.Fatal(err)
		}
		imports = append(imports, p)
	}
	if len(imports) != 1 || imports[0] != "fmt" {
		t.Errorf("import edges = %v, want [fmt]", imports)
	}

	// Verify FTS works for the inserted symbols.
	kw, err := st.SearchKeyword("Alpha", 10)
	if err != nil {
		t.Fatalf("SearchKeyword() error = %v", err)
	}
	if len(kw) != 1 || kw[0].Name != "Alpha" {
		t.Errorf("SearchKeyword(Alpha) = %v, want [Alpha]", kw)
	}
}

func TestUpsertParsedFile_ReplacesStaleSymbols(t *testing.T) {
	st := openTestStore(t)

	// First version has OldFunc.
	first := ParsedFileData{
		File:    types.FileRecord{Path: "rep.go", SHA256: "v1", ModTime: 100, Lang: "go"},
		Symbols: []types.Symbol{{ID: "rep.go:OldFunc:func", File: "rep.go", Name: "OldFunc", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	}
	if err := st.UpsertParsedFile(first); err != nil {
		t.Fatal(err)
	}

	// Second version replaces with NewFunc.
	second := ParsedFileData{
		File:    types.FileRecord{Path: "rep.go", SHA256: "v2", ModTime: 200, Lang: "go"},
		Symbols: []types.Symbol{{ID: "rep.go:NewFunc:func", File: "rep.go", Name: "NewFunc", Kind: types.KindFunction, StartLine: 1, EndLine: 2}},
	}
	if err := st.UpsertParsedFile(second); err != nil {
		t.Fatal(err)
	}

	syms, err := st.GetSymbolsByFile("rep.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(syms) != 1 || syms[0].Name != "NewFunc" {
		t.Errorf("symbols after replace = %v, want [NewFunc]", syms)
	}

	sha, err := st.GetFileSHA("rep.go")
	if err != nil {
		t.Fatal(err)
	}
	if sha != "v2" {
		t.Errorf("SHA after replace = %q, want %q", sha, "v2")
	}
}

func TestUpsertParsedFile_EmptySymbols(t *testing.T) {
	st := openTestStore(t)
	pf := ParsedFileData{
		File: types.FileRecord{Path: "empty.go", SHA256: "x", ModTime: 1, Lang: "go"},
	}
	if err := st.UpsertParsedFile(pf); err != nil {
		t.Fatalf("UpsertParsedFile() error = %v", err)
	}
	sha, err := st.GetFileSHA("empty.go")
	if err != nil {
		t.Fatal(err)
	}
	if sha != "x" {
		t.Errorf("GetFileSHA() = %q, want %q", sha, "x")
	}
}

// ---- UpsertParsedFileBatch tests -------------------------------------------

func TestUpsertParsedFileBatch_StoresMultipleFiles(t *testing.T) {
	st := openTestStore(t)

	files := []ParsedFileData{
		{
			File: types.FileRecord{Path: "a.go", SHA256: "sha_a", ModTime: 1, Lang: "go"},
			Symbols: []types.Symbol{
				{ID: "a.go:Foo:func", File: "a.go", Name: "Foo", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			},
			CallEdges: []types.CallEdge{
				{CallerFile: "a.go", CallerSymbol: "a.go:Foo:func", CalleeSymbol: "Bar", Line: 2},
			},
		},
		{
			File: types.FileRecord{Path: "b.go", SHA256: "sha_b", ModTime: 2, Lang: "go"},
			Symbols: []types.Symbol{
				{ID: "b.go:Bar:func", File: "b.go", Name: "Bar", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
				{ID: "b.go:Baz:func", File: "b.go", Name: "Baz", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			},
			ImportEdges: []types.ImportEdge{
				{ImporterFile: "b.go", ImportedPath: "os"},
			},
		},
	}

	if err := st.UpsertParsedFileBatch(files); err != nil {
		t.Fatalf("UpsertParsedFileBatch() error = %v", err)
	}

	// Verify file a.go.
	shaA, err := st.GetFileSHA("a.go")
	if err != nil {
		t.Fatal(err)
	}
	if shaA != "sha_a" {
		t.Errorf("SHA a.go = %q, want %q", shaA, "sha_a")
	}
	symsA, err := st.GetSymbolsByFile("a.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(symsA) != 1 || symsA[0].Name != "Foo" {
		t.Errorf("symbols a.go = %v, want [Foo]", symsA)
	}

	// Verify file b.go.
	symsB, err := st.GetSymbolsByFile("b.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(symsB) != 2 {
		t.Errorf("symbols b.go len = %d, want 2", len(symsB))
	}

	// Verify call edges from a.go.
	callees, err := st.CalleeSymbols("a.go:Foo:func")
	if err != nil {
		t.Fatal(err)
	}
	if len(callees) != 1 || callees[0] != "Bar" {
		t.Errorf("callees = %v, want [Bar]", callees)
	}

	// Verify FTS works across the batch.
	kw, err := st.SearchKeyword("Baz", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(kw) != 1 || kw[0].Name != "Baz" {
		t.Errorf("SearchKeyword(Baz) = %v, want [Baz]", kw)
	}
}

func TestUpsertParsedFileBatch_EmptySlice(t *testing.T) {
	st := openTestStore(t)
	if err := st.UpsertParsedFileBatch(nil); err != nil {
		t.Fatalf("UpsertParsedFileBatch(nil) error = %v", err)
	}
}

// BenchmarkUpsertParsedFile_vs_Separate compares single-tx UpsertParsedFile
// against the old separate-call pattern.
func BenchmarkUpsertParsedFile_Single(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()

	pf := ParsedFileData{
		File: types.FileRecord{Path: "bench.go", SHA256: "x", ModTime: 1, Lang: "go"},
	}
	for i := 0; i < 50; i++ {
		name := "Sym" + strconv.Itoa(i)
		pf.Symbols = append(pf.Symbols, types.Symbol{
			ID: "bench.go:" + name + ":func", File: "bench.go",
			Name: name, Kind: types.KindFunction, StartLine: i*3 + 1, EndLine: i*3 + 3,
		})
		pf.CallEdges = append(pf.CallEdges, types.CallEdge{
			CallerFile: "bench.go", CallerSymbol: "bench.go:" + name + ":func",
			CalleeSymbol: "Target", Line: i + 1,
		})
	}
	pf.ImportEdges = []types.ImportEdge{
		{ImporterFile: "bench.go", ImportedPath: "fmt"},
		{ImporterFile: "bench.go", ImportedPath: "os"},
	}

	b.ResetTimer()
	for b.Loop() {
		_ = st.UpsertParsedFile(pf)
	}
}

func BenchmarkUpsertParsedFile_Separate(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()

	fr := types.FileRecord{Path: "bench.go", SHA256: "x", ModTime: 1, Lang: "go"}
	var syms []types.Symbol
	var edges []types.CallEdge
	for i := 0; i < 50; i++ {
		name := "Sym" + strconv.Itoa(i)
		syms = append(syms, types.Symbol{
			ID: "bench.go:" + name + ":func", File: "bench.go",
			Name: name, Kind: types.KindFunction, StartLine: i*3 + 1, EndLine: i*3 + 3,
		})
		edges = append(edges, types.CallEdge{
			CallerFile: "bench.go", CallerSymbol: "bench.go:" + name + ":func",
			CalleeSymbol: "Target", Line: i + 1,
		})
	}
	imports := []types.ImportEdge{
		{ImporterFile: "bench.go", ImportedPath: "fmt"},
		{ImporterFile: "bench.go", ImportedPath: "os"},
	}

	b.ResetTimer()
	for b.Loop() {
		_ = st.UpsertFile(fr)
		_ = st.DeleteSymbolsByFile("bench.go")
		_ = st.UpsertSymbolsBatch(syms)
		_ = st.UpsertCallEdges("bench.go", edges)
		_ = st.UpsertImportEdges("bench.go", imports)
	}
}

func BenchmarkUpsertParsedFileBatch_32files(b *testing.B) {
	st, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()

	var batch []ParsedFileData
	for f := 0; f < 32; f++ {
		fpath := "file" + strconv.Itoa(f) + ".go"
		pf := ParsedFileData{
			File: types.FileRecord{Path: fpath, SHA256: "sha" + strconv.Itoa(f), ModTime: 1, Lang: "go"},
		}
		for i := 0; i < 30; i++ {
			name := "Sym" + strconv.Itoa(f) + "_" + strconv.Itoa(i)
			pf.Symbols = append(pf.Symbols, types.Symbol{
				ID: fpath + ":" + name + ":func", File: fpath,
				Name: name, Kind: types.KindFunction, StartLine: i*3 + 1, EndLine: i*3 + 3,
			})
		}
		batch = append(batch, pf)
	}

	b.ResetTimer()
	for b.Loop() {
		_ = st.UpsertParsedFileBatch(batch)
	}
}

// ---- RRF SearchHybrid tests ------------------------------------------------

func TestSearchHybrid_RRFRanking(t *testing.T) {
	// A symbol that appears in both keyword AND semantic results should
	// rank higher than one that only appears in one of them.
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "rrf.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "rrf.go:Both:func", File: "rrf.go", Name: "Both", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "rrf.go:OnlySem:func", File: "rrf.go", Name: "OnlySem", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			{ID: "rrf.go:OnlyKW:func", File: "rrf.go", Name: "OnlyKW", Kind: types.KindFunction, StartLine: 5, EndLine: 6},
		},
	)
	// "Both" appears in keyword results (name matches "Both") and has high semantic score.
	if err := st.UpsertEmbedding("rrf.go:Both:func", "m", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	// "OnlySem" has high semantic score but won't match keyword "Both".
	if err := st.UpsertEmbedding("rrf.go:OnlySem:func", "m", []float32{0.95, 0.05, 0}); err != nil {
		t.Fatal(err)
	}
	// "OnlyKW" has low semantic score.
	if err := st.UpsertEmbedding("rrf.go:OnlyKW:func", "m", []float32{0, 0, 0.1}); err != nil {
		t.Fatal(err)
	}

	got, err := st.SearchHybrid([]float32{1, 0, 0}, "Both", 10)
	if err != nil {
		t.Fatalf("SearchHybrid() error = %v", err)
	}
	if len(got) == 0 {
		t.Fatal("SearchHybrid() returned no results")
	}
	// "Both" should be first — it has contributions from both FTS and semantic.
	if got[0].Name != "Both" {
		t.Errorf("expected Both to be ranked first via RRF, got %q", got[0].Name)
	}
}

// ---- RerankByCallGraph tests -----------------------------------------------

func TestRerankByCallGraph_BoostsConnectedSymbols(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "cg.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "cg.go:A:func", File: "cg.go", Name: "A", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "cg.go:B:func", File: "cg.go", Name: "B", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
			{ID: "cg.go:C:func", File: "cg.go", Name: "C", Kind: types.KindFunction, StartLine: 5, EndLine: 6},
			{ID: "cg.go:D:func", File: "cg.go", Name: "D", Kind: types.KindFunction, StartLine: 7, EndLine: 8},
		},
	)
	// A calls B, B calls C, D is isolated.
	if err := st.UpsertCallEdges("cg.go", []types.CallEdge{
		{CallerFile: "cg.go", CallerSymbol: "A", CalleeSymbol: "B", Line: 1},
		{CallerFile: "cg.go", CallerSymbol: "B", CalleeSymbol: "C", Line: 3},
	}); err != nil {
		t.Fatal(err)
	}

	// Input order: D, C, B, A (D first, but D is isolated).
	input := []types.Symbol{
		{ID: "cg.go:D:func", Name: "D"},
		{ID: "cg.go:C:func", Name: "C"},
		{ID: "cg.go:B:func", Name: "B"},
		{ID: "cg.go:A:func", Name: "A"},
	}

	got, err := st.RerankByCallGraph(input)
	if err != nil {
		t.Fatalf("RerankByCallGraph() error = %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("RerankByCallGraph() returned %d, want 4", len(got))
	}

	// B has the most connections (caller of C, callee of A) = 2 connections → boost 4.
	// C has 1 connection (callee of B) → boost 2.
	// A has 1 connection (caller of B) → boost 2.
	// D has 0 connections → no boost.
	// Expected priorities (lower = ranked earlier):
	//   D: pos 0 - 0 = 0
	//   C: pos 1 - 2 = -1
	//   B: pos 2 - 4 = -2
	//   A: pos 3 - 2 = 1
	// So final order should be: B, C, D, A.
	// B (most connected) should be first.
	if got[0].Name != "B" {
		t.Errorf("expected most-connected symbol B to be first, got %q; order: %v",
			got[0].Name, symbolNames(got))
	}
	// D (isolated) should rank after B and C (connected symbols).
	posB, posD := -1, -1
	for i, sym := range got {
		switch sym.Name {
		case "B":
			posB = i
		case "D":
			posD = i
		}
	}
	if posD < posB {
		t.Errorf("isolated symbol D (pos %d) ranked before connected B (pos %d); order: %v",
			posD, posB, symbolNames(got))
	}
}

func TestRerankByCallGraph_NoConnections(t *testing.T) {
	st := openTestStore(t)
	seedFile(t, st,
		types.FileRecord{Path: "nc.go", SHA256: "x", ModTime: 1, Lang: "go"},
		[]types.Symbol{
			{ID: "nc.go:X:func", File: "nc.go", Name: "X", Kind: types.KindFunction, StartLine: 1, EndLine: 2},
			{ID: "nc.go:Y:func", File: "nc.go", Name: "Y", Kind: types.KindFunction, StartLine: 3, EndLine: 4},
		},
	)

	input := []types.Symbol{
		{ID: "nc.go:X:func", Name: "X"},
		{ID: "nc.go:Y:func", Name: "Y"},
	}

	got, err := st.RerankByCallGraph(input)
	if err != nil {
		t.Fatalf("RerankByCallGraph() error = %v", err)
	}
	// With no connections, order should be preserved.
	if got[0].Name != "X" || got[1].Name != "Y" {
		t.Errorf("expected order preserved with no connections, got %v", symbolNames(got))
	}
}

func TestRerankByCallGraph_SingleSymbol(t *testing.T) {
	st := openTestStore(t)
	input := []types.Symbol{{ID: "x:A:func", Name: "A"}}
	got, err := st.RerankByCallGraph(input)
	if err != nil {
		t.Fatalf("RerankByCallGraph() error = %v", err)
	}
	if len(got) != 1 || got[0].Name != "A" {
		t.Errorf("unexpected result: %v", got)
	}
}

func symbolNames(syms []types.Symbol) []string {
	names := make([]string, len(syms))
	for i, s := range syms {
		names[i] = s.Name
	}
	return names
}

// seedDuplicateNameSymbols inserts n symbols that all share the same Name and
// Kind (simulating the same SQL table defined in multiple migration files), each
// with a distinct file and a distinct embedding vector. The vectors are set so
// that the target symbol (index 0) has the highest cosine similarity to the
// query vector (all-ones, unit-normalised).
func seedDuplicateNameSymbols(t *testing.T, st *Store, name string, kind types.SymbolKind, n, dims int) {
	t.Helper()
	syms := make([]types.Symbol, n)
	items := make([]EmbeddingItem, n)
	for i := range syms {
		file := "migration_" + strconv.Itoa(i) + ".sql"
		if err := st.UpsertFile(types.FileRecord{Path: file, SHA256: "sha" + strconv.Itoa(i), ModTime: int64(i + 1), Lang: "sql"}); err != nil {
			t.Fatalf("UpsertFile: %v", err)
		}
		id := file + ":" + name + ":" + string(kind)
		syms[i] = types.Symbol{
			ID: id, File: file, Name: name, Kind: kind,
			Signature: "CREATE TABLE " + name, StartLine: 1, EndLine: 5,
		}
		// Give index-0 the highest similarity: all dims set to 1/(sqrt(dims)).
		// All others get progressively smaller first component so they rank lower.
		vec := make([]float32, dims)
		for d := range vec {
			vec[d] = 1.0 / float32(dims)
		}
		if i > 0 {
			vec[0] = 0 // reduce similarity for non-primary duplicates
		}
		items[i] = EmbeddingItem{SymbolID: id, Model: "test", Vector: vec}
	}
	if err := st.UpsertSymbolsBatch(syms); err != nil {
		t.Fatalf("UpsertSymbolsBatch: %v", err)
	}
	if err := st.UpsertEmbeddingsBatch(items); err != nil {
		t.Fatalf("UpsertEmbeddingsBatch: %v", err)
	}
}

func TestSearchSemantic_DeduplicatesByNameAndKind(t *testing.T) {
	const dims = 8
	const dupName = "users"
	const dupKind = types.KindTable
	const nDups = 5 // 5 migration files all defining "users" table

	st := openTestStore(t)
	seedDuplicateNameSymbols(t, st, dupName, dupKind, nDups, dims)

	query := make([]float32, dims)
	for i := range query {
		query[i] = 1.0 / float32(dims)
	}

	results, err := st.SearchSemantic(query, 10)
	if err != nil {
		t.Fatalf("SearchSemantic() error = %v", err)
	}

	// Count how many results have the duplicate name+kind.
	count := 0
	for _, s := range results {
		if s.Name == dupName && s.Kind == dupKind {
			count++
		}
	}
	if count != 1 {
		t.Errorf("SearchSemantic returned %d symbols with name=%q kind=%q, want exactly 1", count, dupName, dupKind)
	}
}

func TestSearchHybrid_DeduplicatesByNameAndKind(t *testing.T) {
	const dims = 8
	const dupName = "orders"
	const dupKind = types.KindTable
	const nDups = 4

	st := openTestStore(t)
	seedDuplicateNameSymbols(t, st, dupName, dupKind, nDups, dims)

	query := make([]float32, dims)
	for i := range query {
		query[i] = 1.0 / float32(dims)
	}

	results, err := st.SearchHybrid(query, dupName, 10)
	if err != nil {
		t.Fatalf("SearchHybrid() error = %v", err)
	}

	count := 0
	for _, s := range results {
		if s.Name == dupName && s.Kind == dupKind {
			count++
		}
	}
	if count != 1 {
		t.Errorf("SearchHybrid returned %d symbols with name=%q kind=%q, want exactly 1", count, dupName, dupKind)
	}
}
