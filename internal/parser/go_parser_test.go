package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewGoParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func findSymbol(syms []types.Symbol, name string) *types.Symbol {
	for i := range syms {
		if syms[i].Name == name {
			return &syms[i]
		}
	}
	return nil
}

func TestGoParser_ExtractsCallEdges(t *testing.T) {
	src := []byte(`package main

import "fmt"

func Greet(name string) {
	msg := fmt.Sprintf("hello %s", name)
	Print(msg)
}

func Print(s string) {}
`)
	p := NewGoParser()
	result, err := p.Parse("main.go", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	callees := make(map[string]bool)
	for _, e := range result.CallEdges {
		if e.CallerSymbol == "Greet" {
			callees[e.CalleeSymbol] = true
		}
	}

	tests := []struct {
		name   string
		callee string
		want   bool
	}{
		{name: "calls Sprintf", callee: "Sprintf", want: true},
		{name: "calls Print", callee: "Print", want: true},
		{name: "does not call Greet", callee: "Greet", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if callees[tc.callee] != tc.want {
				t.Errorf("callees[%q] = %v, want %v", tc.callee, callees[tc.callee], tc.want)
			}
		})
	}
}

func TestGoParser_RecordsLineNumbers(t *testing.T) {
	result := parseFixture(t, "widget.go")

	sym := findSymbol(result.Symbols, "New")
	if sym == nil {
		t.Fatal("symbol New not found")
	}
	if sym.StartLine <= 0 {
		t.Errorf("StartLine = %d, want > 0", sym.StartLine)
	}
	if sym.EndLine < sym.StartLine {
		t.Errorf("EndLine %d < StartLine %d", sym.EndLine, sym.StartLine)
	}
}

func TestGoParser_ExtractsImports(t *testing.T) {
	result := parseFixture(t, "widget.go")

	imported := make(map[string]bool)
	for _, e := range result.ImportEdges {
		imported[e.ImportedPath] = true
	}

	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "fmt is imported", path: "fmt", want: true},
		{name: "io is imported", path: "io", want: true},
		{name: "os is not imported", path: "os", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if imported[tc.path] != tc.want {
				t.Errorf("imported[%q] = %v, want %v", tc.path, imported[tc.path], tc.want)
			}
		})
	}
}

func TestGoParser_ExtractsTypes(t *testing.T) {
	result := parseFixture(t, "widget.go")

	tests := []struct {
		name     string
		wantKind types.SymbolKind
	}{
		{name: "Widget", wantKind: types.KindStruct},
		{name: "Renderer", wantKind: types.KindInterface},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.name)
			if sym == nil {
				t.Fatalf("symbol %q not found", tc.name)
			}
			if sym.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", sym.Kind, tc.wantKind)
			}
		})
	}
}

func TestGoParser_ExtractsMethodWithReceiver(t *testing.T) {
	result := parseFixture(t, "widget.go")

	sym := findSymbol(result.Symbols, "Render")
	if sym == nil {
		t.Fatal("symbol Render not found")
	}
	if sym.Kind != types.KindMethod {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindMethod)
	}
	if sym.Receiver != "Widget" {
		t.Errorf("Receiver = %q, want %q", sym.Receiver, "Widget")
	}
}

func TestGoParser_ExtractsFunctions(t *testing.T) {
	result := parseFixture(t, "widget.go")

	tests := []struct {
		name     string
		wantKind types.SymbolKind
	}{
		{name: "New", wantKind: types.KindFunction},
		{name: "unexported", wantKind: types.KindFunction},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.name)
			if sym == nil {
				t.Fatalf("symbol %q not found in results", tc.name)
			}
			if sym.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", sym.Kind, tc.wantKind)
			}
		})
	}
}

func TestGoParser_BlankIdentifierVarNotIndexed(t *testing.T) {
	src := []byte(`package mypackage

// Reader is a named interface.
type Reader interface{ Read() }

// Ensure *MyType implements Reader — blank identifier, should NOT be indexed.
var _ Reader = (*MyType)(nil)

// Named var — should still be indexed.
var Version = "1.0"
`)
	p := NewGoParser()
	result, err := p.Parse("blanks.go", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	tests := []struct {
		name      string
		symName   string
		wantFound bool
	}{
		{name: "blank _ var is not indexed", symName: "_", wantFound: false},
		{name: "named var Version is indexed", symName: "Version", wantFound: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.symName)
			if tc.wantFound && sym == nil {
				t.Errorf("symbol %q not found, want it present", tc.symName)
			}
			if !tc.wantFound && sym != nil {
				t.Errorf("symbol %q found at %s:%d, want it absent", tc.symName, sym.File, sym.StartLine)
			}
		})
	}
}

// ---- Benchmarks ------------------------------------------------------------

func BenchmarkGoParser_SmallFile(b *testing.B) {
	src, err := os.ReadFile("testdata/widget.go")
	if err != nil {
		b.Fatal(err)
	}
	p := NewGoParser()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := p.Parse("widget.go", src)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGoParser_LargeFile(b *testing.B) {
	src, err := os.ReadFile("testdata/large.go")
	if err != nil {
		b.Fatal(err)
	}
	p := NewGoParser()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := p.Parse("large.go", src)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGoParser_LargeFile_Parallel(b *testing.B) {
	src, err := os.ReadFile("testdata/large.go")
	if err != nil {
		b.Fatal(err)
	}
	p := NewGoParser()
	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := p.Parse("large.go", src)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// ---- Snippet tests ---------------------------------------------------------

func TestGoParser_PopulatesSnippet(t *testing.T) {
	result := parseFixture(t, "widget.go")

	tests := []struct {
		name       string
		symbolName string
		wantPrefix string // snippet should start with this
	}{
		{
			name:       "function New has snippet",
			symbolName: "New",
			wantPrefix: "func New(name string) *Widget",
		},
		{
			name:       "method Render has snippet",
			symbolName: "Render",
			wantPrefix: "func (wg *Widget) Render(w io.Writer) error",
		},
		{
			name:       "struct Widget has snippet",
			symbolName: "Widget",
			wantPrefix: "Widget struct",
		},
		{
			name:       "interface Renderer has snippet",
			symbolName: "Renderer",
			wantPrefix: "Renderer interface",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.symbolName)
			if sym == nil {
				t.Fatalf("symbol %q not found", tc.symbolName)
			}
			if sym.Snippet == "" {
				t.Fatalf("Snippet is empty for %q", tc.symbolName)
			}
			if len(sym.Snippet) > maxSnippetBytes {
				t.Errorf("Snippet length %d exceeds maxSnippetBytes %d", len(sym.Snippet), maxSnippetBytes)
			}
			if !containsPrefix(sym.Snippet, tc.wantPrefix) {
				t.Errorf("Snippet = %q, want prefix %q", sym.Snippet, tc.wantPrefix)
			}
		})
	}
}

func containsPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func TestTruncateSnippet(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int // expected max length
	}{
		{name: "short string", input: "hello", want: 5},
		{name: "at limit", input: string(make([]byte, maxSnippetBytes)), want: maxSnippetBytes},
		{name: "over limit", input: string(make([]byte, maxSnippetBytes+100)), want: maxSnippetBytes},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := truncateSnippet(tc.input)
			if len(got) > tc.want {
				t.Errorf("truncateSnippet() len = %d, want <= %d", len(got), tc.want)
			}
		})
	}
}

func TestTruncateSnippet_UTF8Safe(t *testing.T) {
	// Build a string that has a multi-byte rune right at the cut boundary.
	// '€' is U+20AC, encoded as 3 bytes: 0xE2 0x82 0xAC
	prefix := string(make([]byte, maxSnippetBytes-1)) // 499 bytes of NUL
	input := prefix + "€"                             // 499 + 3 = 502 bytes
	got := truncateSnippet(input)
	if len(got) > maxSnippetBytes {
		t.Errorf("truncateSnippet() len = %d, want <= %d", len(got), maxSnippetBytes)
	}
	// Must not end with a partial rune.
	for i := len(got) - 1; i >= len(got)-3 && i >= 0; i-- {
		b := got[i]
		if b >= 0x80 && b < 0xC0 {
			// continuation byte at end — check there's a valid start before it
			continue
		}
		break
	}
}
