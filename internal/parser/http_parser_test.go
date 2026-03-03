package parser

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestHTTPParser_Language(t *testing.T) {
	p := NewHTTPParser()
	if got := p.Language(); got != "http" {
		t.Errorf("Language() = %q, want %q", got, "http")
	}
}

func TestHTTPParser_Extensions(t *testing.T) {
	p := NewHTTPParser()
	exts := p.Extensions()
	want := map[string]bool{".http": true, ".rest": true}
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

func TestHTTPParser_Parse(t *testing.T) {
	p := NewHTTPParser()

	tests := []struct {
		name     string
		input    string
		wantSyms []struct {
			name string
			kind types.SymbolKind
		}
	}{
		{
			name:  "single GET request",
			input: "GET https://api.example.com/users\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "GET /users", kind: types.KindFunction},
			},
		},
		{
			name: "POST with headers and body",
			input: `POST https://api.example.com/users
Content-Type: application/json

{
    "name": "alice"
}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "POST /users", kind: types.KindFunction},
			},
		},
		{
			name: "multiple requests separated by ###",
			input: `### List users
GET https://api.example.com/users

###

### Create user
POST https://api.example.com/users
Content-Type: application/json

{
    "name": "bob"
}

###

DELETE https://api.example.com/users/123`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "GET /users", kind: types.KindFunction},
				{name: "POST /users", kind: types.KindFunction},
				{name: "DELETE /users/123", kind: types.KindFunction},
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
			name:  "comments only",
			input: "# Just a comment\n# Another comment\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name: "request with comment name",
			input: `# @name getUserById
GET https://api.example.com/users/42
Authorization: Bearer token123`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "GET /users/42", kind: types.KindFunction},
			},
		},
		{
			name:  "request with query params",
			input: `GET https://api.example.com/users?page=1&limit=10`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "GET /users", kind: types.KindFunction},
			},
		},
		{
			name: "PUT and PATCH",
			input: `PUT https://api.example.com/users/1
Content-Type: application/json

{"name": "updated"}

###

PATCH https://api.example.com/users/1
Content-Type: application/json

{"active": true}`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "PUT /users/1", kind: types.KindFunction},
				{name: "PATCH /users/1", kind: types.KindFunction},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := p.Parse("api.http", []byte(tc.input))
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
				if got.File != "api.http" {
					t.Errorf("symbol[%d].File = %q, want %q", i, got.File, "api.http")
				}
				if got.StartLine == 0 {
					t.Errorf("symbol[%d].StartLine = 0, want nonzero", i)
				}
			}
		})
	}
}

func TestHTTPParser_LeadingComment(t *testing.T) {
	p := NewHTTPParser()
	input := `### List all active users
# Returns paginated results
GET https://api.example.com/users`
	result, err := p.Parse("api.http", []byte(input))
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

func TestHTTPParser_SignatureIncludesFullURL(t *testing.T) {
	p := NewHTTPParser()
	input := "GET https://api.example.com/users/42\nAuthorization: Bearer token\n"
	result, err := p.Parse("api.http", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	sig := result.Symbols[0].Signature
	if sig != "GET https://api.example.com/users/42" {
		t.Errorf("Signature = %q, want full request line", sig)
	}
}
