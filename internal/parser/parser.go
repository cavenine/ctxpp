// Package parser defines the Parser interface and per-language implementations.
package parser

import (
	"github.com/cavenine/ctxpp/internal/types"
)

// Result holds everything extracted from a single source file.
type Result struct {
	Symbols     []types.Symbol
	CallEdges   []types.CallEdge
	ImportEdges []types.ImportEdge
}

// Parser extracts symbols and graph edges from source files.
// Implementations are provided per language; the interface is intentionally
// small so new languages can be added without touching the indexer.
type Parser interface {
	// Language returns the canonical language tag (e.g. "go", "typescript").
	Language() string

	// Extensions returns the file extensions this parser handles (e.g. ".go").
	Extensions() []string

	// Parse extracts symbols and edges from the given source bytes.
	// filePath is the repo-relative path used to populate Symbol.File.
	Parse(filePath string, src []byte) (Result, error)
}

// FilenameParser is an optional interface that parsers can implement to match
// files by exact filename (e.g. "Makefile", "Dockerfile") in addition to
// extension matching. The indexer checks for this interface when a file has
// no matching extension.
type FilenameParser interface {
	Parser
	// Filenames returns exact basenames this parser handles (e.g. "Makefile").
	Filenames() []string
}
