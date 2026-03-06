package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseCSharpFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewCSharpParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestCSharpParser_Language(t *testing.T) {
	p := NewCSharpParser()
	if got := p.Language(); got != "csharp" {
		t.Errorf("Language() = %q, want %q", got, "csharp")
	}
}

func TestCSharpParser_Extensions(t *testing.T) {
	p := NewCSharpParser()
	got := p.Extensions()
	if len(got) != 1 || got[0] != ".cs" {
		t.Errorf("Extensions() = %v, want [.cs]", got)
	}
}

func TestCSharpParser_ExtractsSymbolsAndImports(t *testing.T) {
	result := parseCSharpFixture(t, "widget.cs")

	tests := []struct {
		name         string
		symbol       string
		wantKind     types.SymbolKind
		wantReceiver string
		wantPackage  string
	}{
		{name: "class", symbol: "Widget", wantKind: types.KindStruct, wantPackage: "Demo.App"},
		{name: "interface", symbol: "IRenderer", wantKind: types.KindInterface, wantPackage: "Demo.App"},
		{name: "field", symbol: "count", wantKind: types.KindField, wantReceiver: "Widget", wantPackage: "Demo.App"},
		{name: "method", symbol: "Render", wantKind: types.KindMethod, wantReceiver: "Widget", wantPackage: "Demo.App"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym := findSymbol(result.Symbols, tc.symbol)
			if sym == nil {
				t.Fatalf("symbol %q not found", tc.symbol)
			}
			if sym.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", sym.Kind, tc.wantKind)
			}
			if sym.Receiver != tc.wantReceiver {
				t.Errorf("Receiver = %q, want %q", sym.Receiver, tc.wantReceiver)
			}
			if sym.Package != tc.wantPackage {
				t.Errorf("Package = %q, want %q", sym.Package, tc.wantPackage)
			}
		})
	}

	imports := map[string]bool{}
	for _, edge := range result.ImportEdges {
		imports[edge.ImportedPath] = true
	}
	for _, want := range []string{"System", "Demo.Rendering"} {
		if !imports[want] {
			t.Errorf("missing import edge %q", want)
		}
	}
}

func TestCSharpParser_ExtractsCallEdges(t *testing.T) {
	result := parseCSharpFixture(t, "widget.cs")

	callees := map[string]bool{}
	for _, edge := range result.CallEdges {
		if edge.CallerSymbol == "Render" {
			callees[edge.CalleeSymbol] = true
		}
	}

	for _, want := range []string{"WriteLine", "Helper"} {
		if !callees[want] {
			t.Errorf("missing callee %q for Render", want)
		}
	}
}
