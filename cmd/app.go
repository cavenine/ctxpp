package cmd

import (
	"encoding/json"
	"strconv"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/indexer"
	"github.com/cavenine/ctxpp/internal/parser"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/cavenine/ctxpp/internal/types"
)

// app holds shared dependencies for MCP handlers.
type app struct {
	store         *store.Store
	indexer       *indexer.Indexer
	indexEmbedder embed.Embedder // used by indexing paths (uncached, preserves BatchEmbedder)
	queryEmbedder embed.Embedder // used by search handlers (CachingEmbedder-wrapped)
	root          string
}

// allParsers returns the full set of language parsers ctx++ supports.
func allParsers() []parser.Parser {
	return []parser.Parser{
		parser.NewGoParser(),
		parser.NewJavaParser(),
		parser.NewKotlinParser(),
		parser.NewJavaScriptParser(),
		parser.NewTypeScriptParser(),
		parser.NewRustParser(),
		parser.NewCSharpParser(),
		parser.NewCParser(),
		parser.NewCppParser(),
		parser.NewSQLParser(),
		parser.NewMarkdownParser(),
		parser.NewHTMLParser(),
		parser.NewTextParser(),
		parser.NewShellParser(),
		parser.NewHTTPParser(),
		parser.NewProtoParser(),
	}
}

// symbolJSON is the JSON-serializable view of a symbol.
type symbolJSON struct {
	ID        string `json:"id"`
	File      string `json:"file"`
	Name      string `json:"name"`
	Kind      string `json:"kind"`
	Signature string `json:"signature"`
	Doc       string `json:"doc,omitempty"`
	Lines     string `json:"lines"`
	Receiver  string `json:"receiver,omitempty"`
	Package   string `json:"package,omitempty"`
}

func marshalSymbols(syms []types.Symbol) string {
	out := make([]symbolJSON, len(syms))
	for i, s := range syms {
		out[i] = symbolJSON{
			ID:        s.ID,
			File:      s.File,
			Name:      s.Name,
			Kind:      string(s.Kind),
			Signature: s.Signature,
			Doc:       s.DocComment,
			Lines:     strconv.Itoa(s.StartLine) + "-" + strconv.Itoa(s.EndLine),
			Receiver:  s.Receiver,
			Package:   s.Package,
		}
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return string(b)
}
