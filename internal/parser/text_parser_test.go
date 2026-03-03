package parser

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestTextParser_Language(t *testing.T) {
	p := NewTextParser()
	if got := p.Language(); got != "text" {
		t.Errorf("Language() = %q, want %q", got, "text")
	}
}

func TestTextParser_Extensions(t *testing.T) {
	p := NewTextParser()
	exts := p.Extensions()
	want := map[string]bool{
		".txt":        true,
		".env":        true,
		".cfg":        true,
		".conf":       true,
		".ini":        true,
		".properties": true,
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

func TestTextParser_Parse(t *testing.T) {
	p := NewTextParser()

	tests := []struct {
		name     string
		file     string
		input    string
		wantName string
		wantKind types.SymbolKind
	}{
		{
			name:     "simple text file",
			file:     "notes.txt",
			input:    "Line one\nLine two\nLine three\n",
			wantName: "notes.txt",
			wantKind: types.KindDocument,
		},
		{
			name:     "env file",
			file:     ".env",
			input:    "DATABASE_URL=postgres://localhost/db\nPORT=8080\n",
			wantName: ".env",
			wantKind: types.KindDocument,
		},
		{
			name:     "Makefile",
			file:     "Makefile",
			input:    ".PHONY: build test\n\nbuild:\n\tgo build ./...\n\ntest:\n\tgo test ./...\n",
			wantName: "Makefile",
			wantKind: types.KindDocument,
		},
		{
			name:     "Dockerfile",
			file:     "Dockerfile",
			input:    "FROM golang:1.24\nWORKDIR /app\nCOPY . .\nRUN go build\n",
			wantName: "Dockerfile",
			wantKind: types.KindDocument,
		},
		{
			name:     "LICENSE",
			file:     "LICENSE",
			input:    "MIT License\n\nCopyright (c) 2026\n",
			wantName: "LICENSE",
			wantKind: types.KindDocument,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := p.Parse(tc.file, []byte(tc.input))
			if err != nil {
				t.Fatalf("Parse() error: %v", err)
			}
			if len(result.Symbols) != 1 {
				t.Fatalf("got %d symbols, want 1", len(result.Symbols))
			}
			sym := result.Symbols[0]
			if sym.Name != tc.wantName {
				t.Errorf("Name = %q, want %q", sym.Name, tc.wantName)
			}
			if sym.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", sym.Kind, tc.wantKind)
			}
			if sym.File != tc.file {
				t.Errorf("File = %q, want %q", sym.File, tc.file)
			}
			if sym.StartLine != 1 {
				t.Errorf("StartLine = %d, want 1", sym.StartLine)
			}
			if sym.Signature == "" {
				t.Error("expected non-empty Signature")
			}
		})
	}
}

func TestTextParser_EmptyFile(t *testing.T) {
	p := NewTextParser()
	result, err := p.Parse("empty.txt", []byte(""))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 0 {
		t.Errorf("got %d symbols for empty file, want 0", len(result.Symbols))
	}
}

func TestTextParser_WhitespaceOnly(t *testing.T) {
	p := NewTextParser()
	result, err := p.Parse("blank.txt", []byte("   \n\n  \n"))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 0 {
		t.Errorf("got %d symbols for whitespace-only file, want 0", len(result.Symbols))
	}
}

func TestTextParser_SignatureFirstLines(t *testing.T) {
	p := NewTextParser()
	input := "First line\nSecond line\nThird line\nFourth line\nFifth line\nSixth line\n"
	result, err := p.Parse("doc.txt", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	sig := result.Symbols[0].Signature
	// Signature should contain the first few lines, not the entire file.
	if len(sig) > 500 {
		t.Errorf("Signature too long (%d chars), expected truncation", len(sig))
	}
}

func TestTextParser_DocComment(t *testing.T) {
	p := NewTextParser()
	input := "Line 1\nLine 2\nLine 3\n"
	result, err := p.Parse("doc.txt", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	// DocComment should contain the full content for embedding.
	if result.Symbols[0].DocComment == "" {
		t.Error("expected non-empty DocComment")
	}
}

func TestTextParser_MatchesSpecialFilenames(t *testing.T) {
	p := NewTextParser()
	names := p.Filenames()
	want := map[string]bool{
		"Makefile":    true,
		"Dockerfile":  true,
		"LICENSE":     true,
		"Rakefile":    true,
		"Vagrantfile": true,
		"Procfile":    true,
	}
	for _, name := range names {
		if !want[name] {
			t.Errorf("unexpected filename %q", name)
		}
		delete(want, name)
	}
	for name := range want {
		t.Errorf("missing filename %q", name)
	}
}
