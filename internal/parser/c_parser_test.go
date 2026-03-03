package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseCFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewCParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestCParser_Language(t *testing.T) {
	p := NewCParser()
	if got := p.Language(); got != "c" {
		t.Errorf("Language() = %q, want %q", got, "c")
	}
}

func TestCParser_Extensions(t *testing.T) {
	p := NewCParser()
	exts := p.Extensions()
	want := map[string]bool{".c": true, ".h": true}
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

func TestCParser_ExtractsTypedefStruct(t *testing.T) {
	result := parseCFixture(t, "widget.h")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Kind != types.KindStruct {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindStruct)
	}
}

func TestCParser_ExtractsTypedefEnum(t *testing.T) {
	result := parseCFixture(t, "widget.h")

	sym := findSymbol(result.Symbols, "Color")
	if sym == nil {
		t.Fatal("symbol Color not found")
	}
	if sym.Kind != types.KindType {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindType)
	}
}

func TestCParser_ExtractsFunctionDeclarations(t *testing.T) {
	result := parseCFixture(t, "widget.h")

	tests := []struct {
		name string
	}{
		{name: "widget_new"},
		{name: "widget_render"},
		{name: "widget_increment"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.name)
			if sym == nil {
				t.Fatalf("symbol %q not found", tc.name)
			}
			if sym.Kind != types.KindFunction {
				t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
			}
		})
	}
}

func TestCParser_ExtractsFunctionDefinitions(t *testing.T) {
	result := parseCFixture(t, "widget.c")

	tests := []struct {
		name string
	}{
		{name: "widget_new"},
		{name: "widget_render"},
		{name: "widget_increment"},
		{name: "widget_log"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.name)
			if sym == nil {
				t.Fatalf("symbol %q not found", tc.name)
			}
			if sym.Kind != types.KindFunction {
				t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
			}
		})
	}
}

func TestCParser_ExtractsMacro(t *testing.T) {
	result := parseCFixture(t, "widget.h")

	sym := findSymbol(result.Symbols, "MAX")
	if sym == nil {
		t.Fatal("symbol MAX not found")
	}
	if sym.Kind != types.KindFunction {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
	}
}

func TestCParser_ExtractsInclude(t *testing.T) {
	result := parseCFixture(t, "widget.c")

	imported := make(map[string]bool)
	for _, e := range result.ImportEdges {
		imported[e.ImportedPath] = true
	}
	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "widget.h included", path: "widget.h", want: true},
		{name: "stdio.h included", path: "stdio.h", want: true},
		{name: "math.h not included", path: "math.h", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if imported[tc.path] != tc.want {
				t.Errorf("imported[%q] = %v, want %v", tc.path, imported[tc.path], tc.want)
			}
		})
	}
}

func TestCParser_ExtractsCallEdges(t *testing.T) {
	src := []byte(`
void foo(void) {
    bar();
    baz(1, 2);
}
void bar(void) {}
void baz(int a, int b) {}
`)
	p := NewCParser()
	result, err := p.Parse("test.c", src)
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

func TestCParser_RecordsLineNumbers(t *testing.T) {
	result := parseCFixture(t, "widget.c")

	sym := findSymbol(result.Symbols, "widget_new")
	if sym == nil {
		t.Fatal("symbol widget_new not found")
	}
	if sym.StartLine <= 0 {
		t.Errorf("StartLine = %d, want > 0", sym.StartLine)
	}
	if sym.EndLine < sym.StartLine {
		t.Errorf("EndLine %d < StartLine %d", sym.EndLine, sym.StartLine)
	}
}

func TestCParser_PopulatesSnippet(t *testing.T) {
	result := parseCFixture(t, "widget.c")

	sym := findSymbol(result.Symbols, "widget_new")
	if sym == nil {
		t.Fatal("symbol widget_new not found")
	}
	if sym.Snippet == "" {
		t.Error("Snippet is empty for widget_new")
	}
	if len(sym.Snippet) > maxSnippetBytes {
		t.Errorf("Snippet length %d exceeds maxSnippetBytes %d", len(sym.Snippet), maxSnippetBytes)
	}
}
