package parser

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestShellParser_Language(t *testing.T) {
	p := NewShellParser()
	if got := p.Language(); got != "shell" {
		t.Errorf("Language() = %q, want %q", got, "shell")
	}
}

func TestShellParser_Extensions(t *testing.T) {
	p := NewShellParser()
	exts := p.Extensions()
	want := map[string]bool{".sh": true, ".bash": true, ".dash": true, ".zsh": true}
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

func TestShellParser_Parse(t *testing.T) {
	p := NewShellParser()

	tests := []struct {
		name     string
		input    string
		wantSyms []struct {
			name string
			kind types.SymbolKind
		}
	}{
		{
			name: "simple function",
			input: `#!/bin/bash

build() {
    go build ./...
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "build", kind: types.KindFunction},
			},
		},
		{
			name: "function keyword",
			input: `function deploy {
    echo "deploying"
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "deploy", kind: types.KindFunction},
			},
		},
		{
			name: "function keyword with parens",
			input: `function cleanup() {
    rm -rf /tmp/build
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "cleanup", kind: types.KindFunction},
			},
		},
		{
			name: "multiple functions",
			input: `#!/bin/bash

# Build the project
build() {
    go build ./...
}

# Run all tests
test() {
    go test ./...
}

# Deploy to production
function deploy {
    echo "deploying"
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "build", kind: types.KindFunction},
				{name: "test", kind: types.KindFunction},
				{name: "deploy", kind: types.KindFunction},
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
			name:  "no functions",
			input: "#!/bin/bash\necho hello\nls -la\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name: "leading comment captured",
			input: `# Build the project from source.
# Requires Go 1.24+.
build() {
    go build ./...
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "build", kind: types.KindFunction},
			},
		},
		{
			name: "nested braces",
			input: `run() {
    if [ -f config ]; then
        source config
    fi
}

other() {
    echo done
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "run", kind: types.KindFunction},
				{name: "other", kind: types.KindFunction},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := p.Parse("script.sh", []byte(tc.input))
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
				if got.File != "script.sh" {
					t.Errorf("symbol[%d].File = %q, want %q", i, got.File, "script.sh")
				}
				if got.StartLine == 0 {
					t.Errorf("symbol[%d].StartLine = 0, want nonzero", i)
				}
			}
		})
	}
}

func TestShellParser_LeadingComment(t *testing.T) {
	p := NewShellParser()
	input := `# Build the project from source.
# Requires Go 1.24+.
build() {
    go build ./...
}`
	result, err := p.Parse("script.sh", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	if result.Symbols[0].DocComment == "" {
		t.Error("expected non-empty DocComment")
	}
}

func TestShellParser_EndLine(t *testing.T) {
	p := NewShellParser()
	input := `first() {
    echo one
    echo two
}

second() {
    echo three
}`
	result, err := p.Parse("script.sh", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 2 {
		t.Fatalf("got %d symbols, want 2", len(result.Symbols))
	}
	if result.Symbols[0].EndLine <= result.Symbols[0].StartLine {
		t.Errorf("first: EndLine=%d should be > StartLine=%d",
			result.Symbols[0].EndLine, result.Symbols[0].StartLine)
	}
	if result.Symbols[1].StartLine <= result.Symbols[0].EndLine {
		t.Errorf("second.StartLine=%d should be > first.EndLine=%d",
			result.Symbols[1].StartLine, result.Symbols[0].EndLine)
	}
}
