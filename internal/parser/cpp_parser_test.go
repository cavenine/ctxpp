package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseCppFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewCppParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestCppParser_Language(t *testing.T) {
	p := NewCppParser()
	if got := p.Language(); got != "cpp" {
		t.Errorf("Language() = %q, want %q", got, "cpp")
	}
}

func TestCppParser_Extensions(t *testing.T) {
	p := NewCppParser()
	exts := p.Extensions()
	want := map[string]bool{
		".cpp": true, ".cc": true, ".cxx": true, ".c++": true,
		".hpp": true, ".hh": true, ".hxx": true, ".h++": true,
	}
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

func TestCppParser_ExtractsClass(t *testing.T) {
	result := parseCppFixture(t, "widget.hpp")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Kind != types.KindStruct {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindStruct)
	}
}

func TestCppParser_ExtractsEnumClass(t *testing.T) {
	result := parseCppFixture(t, "widget.hpp")

	sym := findSymbol(result.Symbols, "Color")
	if sym == nil {
		t.Fatal("symbol Color not found")
	}
	if sym.Kind != types.KindType {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindType)
	}
}

func TestCppParser_ExtractsMemberDeclarations(t *testing.T) {
	result := parseCppFixture(t, "widget.hpp")

	tests := []struct {
		name     string
		wantKind types.SymbolKind
	}{
		{name: "render", wantKind: types.KindMethod},
		{name: "increment", wantKind: types.KindMethod},
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

func TestCppParser_ExtractsOutOfLineDefinitions(t *testing.T) {
	result := parseCppFixture(t, "widget.cpp")

	tests := []struct {
		name     string
		wantKind types.SymbolKind
		receiver string
	}{
		{name: "render", wantKind: types.KindMethod, receiver: "Widget"},
		{name: "increment", wantKind: types.KindMethod, receiver: "Widget"},
		{name: "log", wantKind: types.KindMethod, receiver: "Widget"},
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
			if sym.Receiver != tc.receiver {
				t.Errorf("Receiver = %q, want %q", sym.Receiver, tc.receiver)
			}
		})
	}
}

func TestCppParser_ExtractsInclude(t *testing.T) {
	result := parseCppFixture(t, "widget.cpp")

	imported := make(map[string]bool)
	for _, e := range result.ImportEdges {
		imported[e.ImportedPath] = true
	}
	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "widget.hpp included", path: "widget.hpp", want: true},
		{name: "sstream included", path: "sstream", want: true},
		{name: "math not included", path: "math", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if imported[tc.path] != tc.want {
				t.Errorf("imported[%q] = %v, want %v", tc.path, imported[tc.path], tc.want)
			}
		})
	}
}

func TestCppParser_ExtractsCallEdges(t *testing.T) {
	src := []byte(`
void foo() {
    bar();
    baz(1);
}
void bar() {}
void baz(int n) {}
`)
	p := NewCppParser()
	result, err := p.Parse("test.cpp", src)
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

func TestCppParser_RecordsLineNumbers(t *testing.T) {
	result := parseCppFixture(t, "widget.cpp")

	sym := findSymbol(result.Symbols, "render")
	if sym == nil {
		t.Fatal("symbol render not found")
	}
	if sym.StartLine <= 0 {
		t.Errorf("StartLine = %d, want > 0", sym.StartLine)
	}
	if sym.EndLine < sym.StartLine {
		t.Errorf("EndLine %d < StartLine %d", sym.EndLine, sym.StartLine)
	}
}

func TestCppParser_PopulatesSnippet(t *testing.T) {
	result := parseCppFixture(t, "widget.cpp")

	sym := findSymbol(result.Symbols, "render")
	if sym == nil {
		t.Fatal("symbol render not found")
	}
	if sym.Snippet == "" {
		t.Error("Snippet is empty for render")
	}
	if len(sym.Snippet) > maxSnippetBytes {
		t.Errorf("Snippet length %d exceeds maxSnippetBytes %d", len(sym.Snippet), maxSnippetBytes)
	}
}
