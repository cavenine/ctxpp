package parser

import (
	"os"
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestProtoParser_LanguageAndExtensions(t *testing.T) {
	p := NewProtoParser()
	if got := p.Language(); got != "protobuf" {
		t.Errorf("Language() = %q, want %q", got, "protobuf")
	}
	exts := p.Extensions()
	if len(exts) != 1 || exts[0] != ".proto" {
		t.Errorf("Extensions() = %v, want [\".proto\"]", exts)
	}
}

func TestProtoParser_Parse(t *testing.T) {
	p := NewProtoParser()
	src, err := os.ReadFile("testdata/example.proto")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	res, err := p.Parse("api/v1/example.proto", src)
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	// We expect:
	//   1 service (GreeterService)
	//   2 rpc methods (SayHello, SayGoodbye)
	//   4 messages (HelloRequest, HelloResponse, GoodbyeRequest, GoodbyeResponse)
	//   1 enum (Status)
	//   = 8 symbols total

	// Build a lookup by name for easier assertions.
	byName := make(map[string]types.Symbol, len(res.Symbols))
	for _, sym := range res.Symbols {
		byName[sym.Name] = sym
	}

	tests := []struct {
		name     string
		wantKind types.SymbolKind
		wantDoc  string // substring match
		wantSig  string // substring match
	}{
		{
			name:     "GreeterService",
			wantKind: types.KindInterface,
			wantDoc:  "GreeterService provides greeting RPCs",
			wantSig:  "service GreeterService",
		},
		{
			name:     "GreeterService.SayHello",
			wantKind: types.KindMethod,
			wantDoc:  "SayHello sends a greeting",
			wantSig:  "rpc SayHello(HelloRequest) returns (HelloResponse)",
		},
		{
			name:     "GreeterService.SayGoodbye",
			wantKind: types.KindMethod,
			wantDoc:  "SayGoodbye sends a farewell",
			wantSig:  "rpc SayGoodbye(GoodbyeRequest) returns (GoodbyeResponse)",
		},
		{
			name:     "HelloRequest",
			wantKind: types.KindType,
			wantDoc:  "HelloRequest is the request for SayHello",
			wantSig:  "message HelloRequest",
		},
		{
			name:     "HelloResponse",
			wantKind: types.KindType,
			wantDoc:  "HelloResponse is the response for SayHello",
			wantSig:  "message HelloResponse",
		},
		{
			name:     "GoodbyeRequest",
			wantKind: types.KindType,
			wantDoc:  "",
			wantSig:  "message GoodbyeRequest",
		},
		{
			name:     "GoodbyeResponse",
			wantKind: types.KindType,
			wantDoc:  "",
			wantSig:  "message GoodbyeResponse",
		},
		{
			name:     "Status",
			wantKind: types.KindType,
			wantDoc:  "Status enumerates request statuses",
			wantSig:  "enum Status",
		},
	}

	if len(res.Symbols) != len(tests) {
		t.Errorf("Parse() returned %d symbols, want %d", len(res.Symbols), len(tests))
		for _, sym := range res.Symbols {
			t.Logf("  %s (%s)", sym.Name, sym.Kind)
		}
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sym, ok := byName[tc.name]
			if !ok {
				t.Fatalf("symbol %q not found in results", tc.name)
			}
			if sym.Kind != tc.wantKind {
				t.Errorf("Kind = %q, want %q", sym.Kind, tc.wantKind)
			}
			if sym.File != "api/v1/example.proto" {
				t.Errorf("File = %q, want %q", sym.File, "api/v1/example.proto")
			}
			if tc.wantDoc != "" && !containsStr(sym.DocComment, tc.wantDoc) {
				t.Errorf("DocComment = %q, want substring %q", sym.DocComment, tc.wantDoc)
			}
			if tc.wantSig != "" && !containsStr(sym.Signature, tc.wantSig) {
				t.Errorf("Signature = %q, want substring %q", sym.Signature, tc.wantSig)
			}
			if sym.StartLine == 0 {
				t.Error("StartLine = 0, want > 0")
			}
		})
	}
}

func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && stringContains(s, substr)))
}

func stringContains(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
