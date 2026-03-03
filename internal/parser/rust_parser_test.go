package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseRustFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewRustParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestRustParser_Language(t *testing.T) {
	p := NewRustParser()
	if got := p.Language(); got != "rust" {
		t.Errorf("Language() = %q, want %q", got, "rust")
	}
}

func TestRustParser_Extensions(t *testing.T) {
	p := NewRustParser()
	exts := p.Extensions()
	want := map[string]bool{".rs": true}
	for _, ext := range exts {
		if !want[ext] {
			t.Errorf("unexpected extension %q", ext)
		}
		delete(want, ext)
	}
	for ext := range want {
		t.Errorf("missing extension %q", ext)
	}
}

func TestRustParser_ExtractsStruct(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Kind != types.KindStruct {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindStruct)
	}
}

func TestRustParser_ExtractsTrait(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	sym := findSymbol(result.Symbols, "Renderer")
	if sym == nil {
		t.Fatal("symbol Renderer not found")
	}
	if sym.Kind != types.KindInterface {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindInterface)
	}
}

func TestRustParser_ExtractsEnum(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	sym := findSymbol(result.Symbols, "Status")
	if sym == nil {
		t.Fatal("symbol Status not found")
	}
	if sym.Kind != types.KindType {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindType)
	}
}

func TestRustParser_ExtractsFunction(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	sym := findSymbol(result.Symbols, "greet")
	if sym == nil {
		t.Fatal("symbol greet not found")
	}
	if sym.Kind != types.KindFunction {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
	}
}

func TestRustParser_ExtractsImplMethods(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	tests := []struct {
		name     string
		wantKind types.SymbolKind
	}{
		{name: "new", wantKind: types.KindMethod},
		{name: "increment", wantKind: types.KindMethod},
		{name: "render", wantKind: types.KindMethod},
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

func TestRustParser_ExtractsCallEdges(t *testing.T) {
	src := []byte(`
pub fn foo() {
    bar();
    baz(1);
}
pub fn bar() {}
pub fn baz(n: u32) {}
`)
	p := NewRustParser()
	result, err := p.Parse("test.rs", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	callees := make(map[string]bool)
	for _, e := range result.CallEdges {
		if e.CallerSymbol == "foo" {
			callees[e.CalleeSymbol] = true
		}
	}
	tests := []struct {
		name   string
		callee string
		want   bool
	}{
		{name: "calls bar", callee: "bar", want: true},
		{name: "calls baz", callee: "baz", want: true},
		{name: "does not call foo", callee: "foo", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if callees[tc.callee] != tc.want {
				t.Errorf("callees[%q] = %v, want %v", tc.callee, callees[tc.callee], tc.want)
			}
		})
	}
}

func TestRustParser_RecordsLineNumbers(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.StartLine <= 0 {
		t.Errorf("StartLine = %d, want > 0", sym.StartLine)
	}
	if sym.EndLine < sym.StartLine {
		t.Errorf("EndLine %d < StartLine %d", sym.EndLine, sym.StartLine)
	}
}

func TestRustParser_PopulatesSnippet(t *testing.T) {
	result := parseRustFixture(t, "widget.rs")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Snippet == "" {
		t.Error("Snippet is empty for Widget")
	}
	if len(sym.Snippet) > maxSnippetBytes {
		t.Errorf("Snippet length %d exceeds maxSnippetBytes %d", len(sym.Snippet), maxSnippetBytes)
	}
}
