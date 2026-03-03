// Package bench provides macro-benchmarks for the ctxpp indexing and search pipeline.
//
// Run:
//
//	go test -bench=. -benchmem -timeout=10m ./bench/
//
// These benchmarks exercise the full pipeline: walk → parse → store → embed → search.
// They use synthetic Go codebases of configurable size to produce reproducible results.
package bench

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/parser"
	"github.com/cavenine/ctxpp/internal/store"
)

// benchLogger returns a logger that discards all output so benchmark
// measurements are not polluted by slog writes to stderr.
func benchLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// generateSyntheticRepo creates a temporary directory with numFiles Go source
// files, each containing typesPerFile types, funcsPerFile functions, and
// methodsPerFile methods. Returns the root directory path.
func generateSyntheticRepo(b *testing.B, numFiles, typesPerFile, funcsPerFile, methodsPerFile int) string {
	b.Helper()
	root := b.TempDir()

	for f := 0; f < numFiles; f++ {
		pkgName := fmt.Sprintf("pkg%d", f)
		dir := filepath.Join(root, pkgName)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			b.Fatal(err)
		}

		var code string
		code += fmt.Sprintf("package %s\n\n", pkgName)
		code += "import (\n\t\"fmt\"\n\t\"strings\"\n)\n\n"

		for t := 0; t < typesPerFile; t++ {
			typeName := fmt.Sprintf("Type%d", t)
			code += fmt.Sprintf("// %s is a generated type for benchmarking.\ntype %s struct {\n", typeName, typeName)
			code += fmt.Sprintf("\tName string\n\tValue int\n\tData []byte\n}\n\n")

			for m := 0; m < methodsPerFile; m++ {
				methName := fmt.Sprintf("Method%d", m)
				code += fmt.Sprintf("// %s performs operation %d on %s.\n", methName, m, typeName)
				code += fmt.Sprintf("func (t *%s) %s(input string) string {\n", typeName, methName)
				code += fmt.Sprintf("\tresult := fmt.Sprintf(\"%%s-%%d\", input, t.Value)\n")
				code += fmt.Sprintf("\treturn strings.TrimSpace(result)\n}\n\n")
			}
		}

		for fn := 0; fn < funcsPerFile; fn++ {
			funcName := fmt.Sprintf("Func%d", fn)
			code += fmt.Sprintf("// %s is a generated function for benchmarking.\n", funcName)
			code += fmt.Sprintf("func %s(a, b string) string {\n", funcName)
			code += fmt.Sprintf("\treturn fmt.Sprintf(\"%%s+%%s\", a, b)\n}\n\n")
		}

		path := filepath.Join(dir, fmt.Sprintf("%s.go", pkgName))
		if err := os.WriteFile(path, []byte(code), 0o644); err != nil {
			b.Fatal(err)
		}
	}
	return root
}

func openBenchStore(b *testing.B) *store.Store {
	b.Helper()
	st, err := store.Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = st.Close() })
	return st
}

// ---- Index Benchmarks ------------------------------------------------------

func BenchmarkIndex_50Files(b *testing.B) {
	root := generateSyntheticRepo(b, 50, 3, 5, 3)
	benchmarkIndex(b, root)
}

func BenchmarkIndex_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	benchmarkIndex(b, root)
}

func BenchmarkIndex_500Files(b *testing.B) {
	root := generateSyntheticRepo(b, 500, 3, 5, 3)
	benchmarkIndex(b, root)
}

func benchmarkIndex(b *testing.B, root string) {
	b.Helper()
	b.ReportAllocs()
	ctx := context.Background()

	for i := 0; i < b.N; i++ {
		st := openBenchStore(b)
		idx := indexer.New(
			indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
			st,
			[]parser.Parser{parser.NewGoParser()},
			embed.NewBundledEmbedder(768),
		)
		stats, err := idx.Index(ctx)
		if err != nil {
			b.Fatal(err)
		}
		if stats.FilesIndexed == 0 {
			b.Fatal("no files indexed")
		}
		b.ReportMetric(float64(stats.FilesIndexed), "files")
		b.ReportMetric(float64(stats.SymbolsIndexed), "symbols")
		_ = st.Close()
	}
}

// ---- Incremental Re-Index (no-op) -----------------------------------------

func BenchmarkReindex_NoChanges_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	st := openBenchStore(b)
	idx := indexer.New(
		indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(768),
	)
	ctx := context.Background()

	// Initial full index.
	if _, err := idx.Index(ctx); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stats, err := idx.Index(ctx)
		if err != nil {
			b.Fatal(err)
		}
		if stats.FilesIndexed != 0 {
			b.Errorf("expected 0 files re-indexed, got %d", stats.FilesIndexed)
		}
		b.ReportMetric(float64(stats.FilesSkipped), "skipped")
	}
}

// ---- Search Benchmarks (after full index) ----------------------------------

func BenchmarkSearch_Keyword_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	st := openBenchStore(b)
	idx := indexer.New(
		indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(768),
	)
	ctx := context.Background()
	if _, err := idx.Index(ctx); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		results, err := st.SearchKeyword("Method0", 20)
		if err != nil {
			b.Fatal(err)
		}
		if len(results) == 0 {
			b.Fatal("no search results")
		}
	}
}

func BenchmarkSearch_Semantic_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	st := openBenchStore(b)
	idx := indexer.New(
		indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(768),
	)
	ctx := context.Background()
	if _, err := idx.Index(ctx); err != nil {
		b.Fatal(err)
	}

	// Create a query vector (zero — bundled embedder uses zeros, so this tests scan speed).
	queryVec := make([]float32, 768)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		results, err := st.SearchSemantic(queryVec, 20)
		if err != nil {
			b.Fatal(err)
		}
		_ = results
	}
}

func BenchmarkSearch_Hybrid_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	st := openBenchStore(b)
	idx := indexer.New(
		indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(768),
	)
	ctx := context.Background()
	if _, err := idx.Index(ctx); err != nil {
		b.Fatal(err)
	}

	queryVec := make([]float32, 768)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		results, err := st.SearchHybrid(queryVec, "Func0", 20)
		if err != nil {
			b.Fatal(err)
		}
		if len(results) == 0 {
			b.Fatal("no search results")
		}
	}
}

// ---- Search Latency Distribution -------------------------------------------

// BenchmarkSearch_LatencyProfile runs many queries and reports p50/p95/p99.
// This is not a standard Go benchmark; it runs inside a b.Run for reporting.
func BenchmarkSearch_LatencyProfile_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	st := openBenchStore(b)
	idx := indexer.New(
		indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(768),
	)
	ctx := context.Background()
	if _, err := idx.Index(ctx); err != nil {
		b.Fatal(err)
	}

	queries := []string{"Method0", "Type1", "Func3", "performs operation", "generated function"}

	b.Run("keyword", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, err := st.SearchKeyword(q, 20)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("semantic", func(b *testing.B) {
		b.ReportAllocs()
		queryVec := make([]float32, 768)
		for i := 0; i < b.N; i++ {
			_, err := st.SearchSemantic(queryVec, 20)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("hybrid", func(b *testing.B) {
		b.ReportAllocs()
		queryVec := make([]float32, 768)
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			_, err := st.SearchHybrid(queryVec, q, 20)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// ---- DB Size Reporting (runs once, not iterated) ---------------------------

func BenchmarkDBSize_200Files(b *testing.B) {
	root := generateSyntheticRepo(b, 200, 3, 5, 3)
	dbPath := filepath.Join(b.TempDir(), "size_test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		b.Fatal(err)
	}

	idx := indexer.New(
		indexer.Config{ProjectRoot: root, Workers: 8, EmbedConcurrency: 4, Logger: benchLogger()},
		st,
		[]parser.Parser{parser.NewGoParser()},
		embed.NewBundledEmbedder(768),
	)
	ctx := context.Background()
	stats, err := idx.Index(ctx)
	if err != nil {
		b.Fatal(err)
	}
	_ = st.Close()

	info, err := os.Stat(dbPath)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportMetric(float64(info.Size()), "db_bytes")
	b.ReportMetric(float64(stats.SymbolsIndexed), "symbols")
	b.ReportMetric(float64(stats.FilesIndexed), "files")

	// Also report bytes per symbol for efficiency tracking.
	if stats.SymbolsIndexed > 0 {
		b.ReportMetric(float64(info.Size())/float64(stats.SymbolsIndexed), "bytes/symbol")
	}
}

// percentile returns the p-th percentile of a sorted slice (0-100).
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p / 100)
	return sorted[idx]
}

// sortFloat64s sorts a slice in-place.
func sortFloat64s(s []float64) {
	sort.Float64s(s)
}
