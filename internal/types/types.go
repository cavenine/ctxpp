// Package types defines shared data structures for ctx++.
package types

// SymbolKind classifies a code symbol.
type SymbolKind string

const (
	KindFunction  SymbolKind = "function"
	KindMethod    SymbolKind = "method"
	KindStruct    SymbolKind = "struct"
	KindInterface SymbolKind = "interface"
	KindType      SymbolKind = "type"
	KindConst     SymbolKind = "const"
	KindVar       SymbolKind = "var"
	KindField     SymbolKind = "field"

	// SQL kinds.
	KindTable     SymbolKind = "table"
	KindView      SymbolKind = "view"
	KindIndex     SymbolKind = "index"
	KindTrigger   SymbolKind = "trigger"
	KindProcedure SymbolKind = "procedure"

	// Document kinds (markdown, HTML, text).
	KindSection  SymbolKind = "section"
	KindDocument SymbolKind = "document"
	KindElement  SymbolKind = "element"
)

// Symbol is an extracted code symbol from a source file.
type Symbol struct {
	// ID is a stable key: "<file>:<name>:<kind>".
	ID string

	// File is the repo-relative file path.
	File string

	// Name is the unqualified symbol identifier.
	Name string

	// Kind classifies the symbol.
	Kind SymbolKind

	// Signature is the full declaration signature (first line / header).
	Signature string

	// DocComment is the leading doc comment, if any.
	DocComment string

	// StartLine / EndLine are 1-based line numbers for the definition body.
	StartLine int
	EndLine   int

	// Receiver is set for methods (the receiver type name).
	Receiver string

	// Package is the package/module/namespace the symbol belongs to.
	Package string

	// SourceTier classifies the symbol's provenance for ranking.
	// Defaults to TierCode (1) if not set.
	SourceTier SourceTier

	// Snippet is a truncated excerpt of the symbol's source body (first
	// ~maxSnippetBytes bytes). Used only for embedding enrichment; not
	// persisted to the database.
	Snippet string
}

// CallEdge represents a call from one symbol to another.
type CallEdge struct {
	CallerFile   string
	CallerSymbol string
	CalleeFile   string
	CalleeSymbol string
	Line         int
}

// ImportEdge represents a file importing another file or package.
type ImportEdge struct {
	ImporterFile string
	ImportedPath string // module path or relative path
}

// SourceTier classifies symbols by provenance for ranking purposes.
// Lower tier numbers indicate higher-signal sources.
type SourceTier int

const (
	// TierCode is project source code (default). Ranking weight: 1.0.
	TierCode SourceTier = 1

	// TierDocs is project documentation, tests, and configs. Ranking weight: 0.85.
	TierDocs SourceTier = 2

	// TierVendor is vendored/third-party dependencies. Ranking weight: 0.7.
	TierVendor SourceTier = 3

	// TierLowSignal is changelogs, generated code, and test fixtures. Ranking weight: 0.5.
	TierLowSignal SourceTier = 4
)

// TierWeight returns the ranking multiplier for a given tier.
func (t SourceTier) TierWeight() float32 {
	switch t {
	case TierCode:
		return 1.0
	case TierDocs:
		return 0.85
	case TierVendor:
		return 0.7
	case TierLowSignal:
		return 0.5
	default:
		return 1.0
	}
}

// FileRecord tracks the indexed state of a source file.
type FileRecord struct {
	Path    string // repo-relative path
	SHA256  string // content hash for incremental skip
	ModTime int64  // unix nano
	Lang    string // language tag, e.g. "go"
}
