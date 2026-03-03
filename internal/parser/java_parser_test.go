package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseJavaFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewJavaParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestJavaParser_Language(t *testing.T) {
	p := NewJavaParser()
	if got := p.Language(); got != "java" {
		t.Errorf("Language() = %q, want %q", got, "java")
	}
}

func TestJavaParser_Extensions(t *testing.T) {
	p := NewJavaParser()
	exts := p.Extensions()
	want := map[string]bool{".java": true}
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

func TestJavaParser_ExtractsClass(t *testing.T) {
	result := parseJavaFixture(t, "widget.java")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Kind != types.KindStruct {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindStruct)
	}
}

func TestJavaParser_ExtractsInterface(t *testing.T) {
	result := parseJavaFixture(t, "widget.java")

	sym := findSymbol(result.Symbols, "Renderer")
	if sym == nil {
		t.Fatal("symbol Renderer not found")
	}
	if sym.Kind != types.KindInterface {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindInterface)
	}
}

func TestJavaParser_ExtractsMethods(t *testing.T) {
	result := parseJavaFixture(t, "widget.java")

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

func TestJavaParser_ExtractsCallEdges(t *testing.T) {
	src := []byte(`
public class Foo {
    public void bar() {
        baz();
        qux(1, 2);
    }
    public void baz() {}
    public void qux(int a, int b) {}
}
`)
	p := NewJavaParser()
	result, err := p.Parse("Foo.java", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	callees := make(map[string]bool)
	for _, e := range result.CallEdges {
		if e.CallerSymbol == "bar" {
			callees[e.CalleeSymbol] = true
		}
	}
	tests := []struct {
		name   string
		callee string
		want   bool
	}{
		{name: "calls baz", callee: "baz", want: true},
		{name: "calls qux", callee: "qux", want: true},
		{name: "does not call bar", callee: "bar", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if callees[tc.callee] != tc.want {
				t.Errorf("callees[%q] = %v, want %v", tc.callee, callees[tc.callee], tc.want)
			}
		})
	}
}

func TestJavaParser_RecordsLineNumbers(t *testing.T) {
	result := parseJavaFixture(t, "widget.java")

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
