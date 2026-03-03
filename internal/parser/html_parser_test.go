package parser

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestHTMLParser_Language(t *testing.T) {
	p := NewHTMLParser()
	if got := p.Language(); got != "html" {
		t.Errorf("Language() = %q, want %q", got, "html")
	}
}

func TestHTMLParser_Extensions(t *testing.T) {
	p := NewHTMLParser()
	exts := p.Extensions()
	want := map[string]bool{".html": true, ".htm": true}
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

func TestHTMLParser_Parse(t *testing.T) {
	p := NewHTMLParser()

	tests := []struct {
		name     string
		input    string
		wantSyms []struct {
			name string
			kind types.SymbolKind
		}
	}{
		{
			name: "title element",
			input: `<html>
<head><title>My Page</title></head>
<body></body>
</html>`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "My Page", kind: types.KindElement},
			},
		},
		{
			name: "headings h1 through h3",
			input: `<h1>Welcome</h1>
<p>Some text.</p>
<h2>Features</h2>
<p>More text.</p>
<h3>Feature One</h3>`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Welcome", kind: types.KindSection},
				{name: "Features", kind: types.KindSection},
				{name: "Feature One", kind: types.KindSection},
			},
		},
		{
			name: "headings with attributes",
			input: `<h1 class="title" id="main-title">Main Title</h1>
<h2 id="sub">Subtitle</h2>`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Main Title", kind: types.KindSection},
				{name: "Subtitle", kind: types.KindSection},
			},
		},
		{
			name: "title and headings together",
			input: `<html>
<head><title>Docs</title></head>
<body>
<h1>Getting Started</h1>
<h2>Installation</h2>
</body></html>`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Docs", kind: types.KindElement},
				{name: "Getting Started", kind: types.KindSection},
				{name: "Installation", kind: types.KindSection},
			},
		},
		{
			name:  "empty file",
			input: "",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name:  "no headings or title",
			input: "<html><body><p>Just text.</p></body></html>",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name: "multiline heading",
			input: `<h1>
  Long Title
</h1>`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Long Title", kind: types.KindSection},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := p.Parse("page.html", []byte(tc.input))
			if err != nil {
				t.Fatalf("Parse() error: %v", err)
			}
			if len(result.Symbols) != len(tc.wantSyms) {
				t.Fatalf("got %d symbols, want %d:\n  got:  %v",
					len(result.Symbols), len(tc.wantSyms), symbolNames(result.Symbols))
			}
			for i, want := range tc.wantSyms {
				got := result.Symbols[i]
				if got.Name != want.name {
					t.Errorf("symbol[%d].Name = %q, want %q", i, got.Name, want.name)
				}
				if got.Kind != want.kind {
					t.Errorf("symbol[%d].Kind = %q, want %q", i, got.Kind, want.kind)
				}
				if got.File != "page.html" {
					t.Errorf("symbol[%d].File = %q, want %q", i, got.File, "page.html")
				}
				if got.StartLine == 0 {
					t.Errorf("symbol[%d].StartLine = 0, want nonzero", i)
				}
			}
		})
	}
}
