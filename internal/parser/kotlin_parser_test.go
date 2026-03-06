package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseKotlinFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewKotlinParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestKotlinParser_Language(t *testing.T) {
	p := NewKotlinParser()
	if got := p.Language(); got != "kotlin" {
		t.Errorf("Language() = %q, want %q", got, "kotlin")
	}
}

func TestKotlinParser_Extensions(t *testing.T) {
	p := NewKotlinParser()
	got := p.Extensions()
	want := map[string]bool{".kt": true, ".kts": true}
	for _, ext := range got {
		if !want[ext] {
			t.Errorf("unexpected extension %q", ext)
		}
		delete(want, ext)
	}
	for ext := range want {
		t.Errorf("missing extension %q", ext)
	}
	if len(got) != 2 {
		t.Errorf("len(Extensions()) = %d, want 2", len(got))
	}
}

func TestKotlinParser_ExtractsSymbolsAndImports(t *testing.T) {
	result := parseKotlinFixture(t, "widget.kt")

	tests := []struct {
		name         string
		symbol       string
		wantKind     types.SymbolKind
		wantReceiver string
		wantPackage  string
	}{
		{name: "class", symbol: "Widget", wantKind: types.KindStruct, wantPackage: "demo.ui"},
		{name: "interface", symbol: "Painter", wantKind: types.KindInterface, wantPackage: "demo.ui"},
		{name: "method", symbol: "render", wantKind: types.KindMethod, wantReceiver: "Widget", wantPackage: "demo.ui"},
		{name: "function", symbol: "buildWidget", wantKind: types.KindFunction, wantPackage: "demo.ui"},
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
			if sym.StartLine <= 0 {
				t.Errorf("StartLine = %d, want > 0", sym.StartLine)
			}
			if sym.EndLine < sym.StartLine {
				t.Errorf("EndLine = %d, want >= StartLine %d", sym.EndLine, sym.StartLine)
			}
		})
	}

	imports := map[string]bool{}
	for _, edge := range result.ImportEdges {
		imports[edge.ImportedPath] = true
	}
	for _, want := range []string{"com.example.Renderer", "kotlin.io.println"} {
		if !imports[want] {
			t.Errorf("missing import edge %q", want)
		}
	}
}

func TestKotlinParser_ExtractsCallEdges(t *testing.T) {
	result := parseKotlinFixture(t, "widget.kt")

	callees := map[string]bool{}
	for _, edge := range result.CallEdges {
		if edge.CallerSymbol == "render" {
			callees[edge.CalleeSymbol] = true
		}
	}

	for _, want := range []string{"println", "helper"} {
		if !callees[want] {
			t.Errorf("missing callee %q for render", want)
		}
	}
}

func TestKotlinParser_UsesReceiverQualifiedIDsForMembers(t *testing.T) {
	src := []byte(`package demo

class First {
	val count = 1
	fun render() {}
}

class Second {
	val count = 2
	fun render() {}
}
`)

	p := NewKotlinParser()
	result, err := p.Parse("widget.kt", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	wantIDs := map[string]bool{
		"widget.kt:First.count:field":    true,
		"widget.kt:Second.count:field":   true,
		"widget.kt:First.render:method":  true,
		"widget.kt:Second.render:method": true,
	}
	for _, sym := range result.Symbols {
		delete(wantIDs, sym.ID)
	}
	for id := range wantIDs {
		t.Errorf("missing qualified symbol ID %q", id)
	}
}
