package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func parseTSFixture(t *testing.T, filename string) Result {
	t.Helper()
	src, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", filename, err)
	}
	p := NewTypeScriptParser()
	result, err := p.Parse(filename, src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	return result
}

func TestTypeScriptParser_Language(t *testing.T) {
	p := NewTypeScriptParser()
	if got := p.Language(); got != "typescript" {
		t.Errorf("Language() = %q, want %q", got, "typescript")
	}
}

func TestTypeScriptParser_Extensions(t *testing.T) {
	p := NewTypeScriptParser()
	exts := p.Extensions()
	want := map[string]bool{".ts": true, ".tsx": true, ".mts": true, ".cts": true}
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

func TestTypeScriptParser_ExtractsClass(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

	sym := findSymbol(result.Symbols, "Widget")
	if sym == nil {
		t.Fatal("symbol Widget not found")
	}
	if sym.Kind != types.KindStruct {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindStruct)
	}
}

func TestTypeScriptParser_ExtractsInterface(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

	sym := findSymbol(result.Symbols, "Renderable")
	if sym == nil {
		t.Fatal("symbol Renderable not found")
	}
	if sym.Kind != types.KindInterface {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindInterface)
	}
}

func TestTypeScriptParser_ExtractsTypeAlias(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

	sym := findSymbol(result.Symbols, "WidgetConfig")
	if sym == nil {
		t.Fatal("symbol WidgetConfig not found")
	}
	if sym.Kind != types.KindType {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindType)
	}
}

func TestTypeScriptParser_ExtractsEnum(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

	sym := findSymbol(result.Symbols, "Status")
	if sym == nil {
		t.Fatal("symbol Status not found")
	}
	if sym.Kind != types.KindType {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindType)
	}
}

func TestTypeScriptParser_ExtractsFunction(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

	sym := findSymbol(result.Symbols, "newWidget")
	if sym == nil {
		t.Fatal("symbol newWidget not found")
	}
	if sym.Kind != types.KindFunction {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
	}
}

func TestTypeScriptParser_ExtractsImport(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

	imported := make(map[string]bool)
	for _, e := range result.ImportEdges {
		imported[e.ImportedPath] = true
	}
	if !imported["fs/promises"] {
		t.Errorf("expected import of fs/promises, got %v", imported)
	}
}

func TestTypeScriptParser_RecordsLineNumbers(t *testing.T) {
	result := parseTSFixture(t, "widget.ts")

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

func TestTypeScriptParser_TSXExtension(t *testing.T) {
	// Verify that a .tsx file parses without error using the TSX grammar.
	src := []byte(`
import React from 'react';

interface Props {
    name: string;
}

function Greeting({ name }: Props): JSX.Element {
    return <div>Hello {name}</div>;
}

export default Greeting;
`)
	p := NewTypeScriptParser()
	result, err := p.Parse("greeting.tsx", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	sym := findSymbol(result.Symbols, "Greeting")
	if sym == nil {
		t.Fatal("symbol Greeting not found in TSX file")
	}
	if sym.Kind != types.KindFunction {
		t.Errorf("Kind = %q, want %q", sym.Kind, types.KindFunction)
	}
}

func TestTypeScriptParser_UsesReceiverQualifiedIDsForClassMethods(t *testing.T) {
	src := []byte(`
class First {
  render(): void {}
}

class Second {
  render(): void {}
}
`)

	p := NewTypeScriptParser()
	result, err := p.Parse("widget.ts", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	wantIDs := map[string]bool{
		"widget.ts:First.render:method":  true,
		"widget.ts:Second.render:method": true,
	}
	for _, sym := range result.Symbols {
		delete(wantIDs, sym.ID)
	}
	for id := range wantIDs {
		t.Errorf("missing qualified symbol ID %q", id)
	}
}
