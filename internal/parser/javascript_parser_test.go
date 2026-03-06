package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseJSFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewJavaScriptParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestJavaScriptParser_Language(t *testing.T) {
	p := NewJavaScriptParser()
	if got := p.Language(); got != "javascript" {
		t.Errorf("Language() = %q, want %q", got, "javascript")
	}
}

func TestJavaScriptParser_Extensions(t *testing.T) {
	p := NewJavaScriptParser()
	exts := p.Extensions()
	want := map[string]bool{".js": true, ".mjs": true, ".cjs": true, ".jsx": true}
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

func TestJavaScriptParser_ExtractsClass(t *testing.T) {
	result := parseJSFixture(t, "widget.js")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Kind != types.KindStruct {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindStruct)
	}
}

func TestJavaScriptParser_ExtractsFunction(t *testing.T) {
	result := parseJSFixture(t, "widget.js")

	sym := findSymbol(result.Symbols, "newWidget")
	if sym == nil {
		t.Fatal("symbol newWidget not found")
	}
	if sym.Kind != types.KindFunction {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
	}
}

func TestJavaScriptParser_ExtractsArrowFunction(t *testing.T) {
	result := parseJSFixture(t, "widget.js")

	sym := findSymbol(result.Symbols, "greet")
	if sym == nil {
		t.Fatal("symbol greet not found")
	}
	if sym.Kind != types.KindFunction {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
	}
}

func TestJavaScriptParser_ExtractsMethod(t *testing.T) {
	result := parseJSFixture(t, "widget.js")

	sym := findSymbol(result.Symbols, "render")
	if sym == nil {
		t.Fatal("symbol render not found")
	}
	if sym.Kind != types.KindMethod {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindMethod)
	}
	if sym.Receiver != "Widget" {
		t.Errorf("Receiver = %q, want %q", sym.Receiver, "Widget")
	}
}

func TestJavaScriptParser_ExtractsImport(t *testing.T) {
	result := parseJSFixture(t, "widget.js")

	imported := make(map[string]bool)
	for _, e := range result.ImportEdges {
		imported[e.ImportedPath] = true
	}
	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "fs/promises imported", path: "fs/promises", want: true},
		{name: "path imported", path: "path", want: true},
		{name: "os not imported", path: "os", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if imported[tc.path] != tc.want {
				t.Errorf("imported[%q] = %v, want %v", tc.path, imported[tc.path], tc.want)
			}
		})
	}
}

func TestJavaScriptParser_ExtractsCallEdges(t *testing.T) {
	src := []byte(`
function foo() {
    bar();
    baz(1);
}
function bar() {}
function baz(n) {}
`)
	p := NewJavaScriptParser()
	result, err := p.Parse("test.js", src)
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

func TestJavaScriptParser_RecordsLineNumbers(t *testing.T) {
	result := parseJSFixture(t, "widget.js")

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

func TestJavaScriptParser_UsesReceiverQualifiedIDsForClassMethods(t *testing.T) {
	src := []byte(`
class First {
  render() {}
}

class Second {
  render() {}
}
`)

	p := NewJavaScriptParser()
	result, err := p.Parse("widget.js", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	wantIDs := map[string]bool{
		"widget.js:First.render:method":  true,
		"widget.js:Second.render:method": true,
	}
	for _, sym := range result.Symbols {
		delete(wantIDs, sym.ID)
	}
	for id := range wantIDs {
		t.Errorf("missing qualified symbol ID %q", id)
	}
}
