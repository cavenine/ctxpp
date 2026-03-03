package parser

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestMarkdownParser_Language(t *testing.T) {
	p := NewMarkdownParser()
	if got := p.Language(); got != "markdown" {
		t.Errorf("Language() = %q, want %q", got, "markdown")
	}
}

func TestMarkdownParser_Extensions(t *testing.T) {
	p := NewMarkdownParser()
	exts := p.Extensions()
	want := map[string]bool{".md": true, ".mdx": true}
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

func TestMarkdownParser_Parse(t *testing.T) {
	p := NewMarkdownParser()

	tests := []struct {
		name     string
		input    string
		wantSyms []struct {
			name string
			kind types.SymbolKind
		}
	}{
		{
			name: "single heading",
			input: `# Getting Started

This is the introduction.`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Getting Started", kind: types.KindSection},
			},
		},
		{
			name: "multiple headings",
			input: `# API Reference

Overview of the API.

## Authentication

Use Bearer tokens.

## Endpoints

### GET /users

Returns all users.

### POST /users

Creates a user.`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "API Reference", kind: types.KindSection},
				{name: "Authentication", kind: types.KindSection},
				{name: "Endpoints", kind: types.KindSection},
				{name: "GET /users", kind: types.KindSection},
				{name: "POST /users", kind: types.KindSection},
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
			name:  "no headings",
			input: "Just some text\nwith no headings.\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name:  "heading in code block ignored",
			input: "# Real Heading\n\n```\n# Not a heading\n```\n\n## Another Real Heading\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Real Heading", kind: types.KindSection},
				{name: "Another Real Heading", kind: types.KindSection},
			},
		},
		{
			name:  "heading levels preserved in signature",
			input: "# Top\n\n## Mid\n\n### Deep\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Top", kind: types.KindSection},
				{name: "Mid", kind: types.KindSection},
				{name: "Deep", kind: types.KindSection},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := p.Parse("doc.md", []byte(tc.input))
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
				if got.File != "doc.md" {
					t.Errorf("symbol[%d].File = %q, want %q", i, got.File, "doc.md")
				}
				if got.StartLine == 0 {
					t.Errorf("symbol[%d].StartLine = 0, want nonzero", i)
				}
			}
		})
	}
}

func TestMarkdownParser_SectionBody(t *testing.T) {
	p := NewMarkdownParser()
	input := `# Overview

This document describes the API.
It has multiple lines of content.

## Next Section

More content here.`

	result, err := p.Parse("doc.md", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 2 {
		t.Fatalf("got %d symbols, want 2", len(result.Symbols))
	}
	// The "Overview" section should capture body text as DocComment.
	if result.Symbols[0].DocComment == "" {
		t.Error("expected non-empty DocComment for section with body text")
	}
	// EndLine of "Overview" should be before "Next Section" starts.
	if result.Symbols[0].EndLine >= result.Symbols[1].StartLine {
		t.Errorf("Overview.EndLine=%d >= NextSection.StartLine=%d",
			result.Symbols[0].EndLine, result.Symbols[1].StartLine)
	}
}

func TestMarkdownParser_Signature(t *testing.T) {
	p := NewMarkdownParser()
	input := "## Configuration Options\n\nSome text.\n"

	result, err := p.Parse("doc.md", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	// Signature should include the heading markers.
	sig := result.Symbols[0].Signature
	if sig != "## Configuration Options" {
		t.Errorf("Signature = %q, want %q", sig, "## Configuration Options")
	}
}
