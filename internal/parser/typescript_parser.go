package parser

import (
	"context"
	"fmt"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/typescript/tsx"
	"github.com/smacker/go-tree-sitter/typescript/typescript"

	"github.com/cavenine/ctxpp/internal/types"
)

// TypeScriptParser implements Parser for TypeScript and TSX source files using tree-sitter.
// It reuses the JavaScript extraction helpers (jsWalkTopLevel, jsHandleDecl, etc.) since
// the TypeScript grammar is a superset of the JavaScript grammar and emits the same
// top-level node types for functions, classes, imports, and variable declarations.
// TypeScript-specific constructs (interfaces, type aliases, enums) are handled by
// tsHandleDecl which augments the base JavaScript dispatch.
type TypeScriptParser struct {
	tsLang  *sitter.Language
	tsxLang *sitter.Language
	tsPool  sync.Pool
	tsxPool sync.Pool
}

// NewTypeScriptParser constructs a TypeScriptParser with pooled tree-sitter parsers
// for both the TypeScript and TSX grammars.
func NewTypeScriptParser() *TypeScriptParser {
	tsLang := typescript.GetLanguage()
	tsxLang := tsx.GetLanguage()
	p := &TypeScriptParser{
		tsLang:  tsLang,
		tsxLang: tsxLang,
	}
	p.tsPool = sync.Pool{
		New: func() any {
			tp := sitter.NewParser()
			tp.SetLanguage(tsLang)
			return tp
		},
	}
	p.tsxPool = sync.Pool{
		New: func() any {
			tp := sitter.NewParser()
			tp.SetLanguage(tsxLang)
			return tp
		},
	}
	return p
}

func (p *TypeScriptParser) Language() string { return "typescript" }
func (p *TypeScriptParser) Extensions() []string {
	return []string{".ts", ".tsx", ".mts", ".cts"}
}

// Parse extracts symbols and edges from a TypeScript or TSX source file.
func (p *TypeScriptParser) Parse(filePath string, src []byte) (Result, error) {
	// Choose the right grammar based on the extension.
	isTSX := hasSuffix(filePath, ".tsx")
	var tsParser *sitter.Parser
	if isTSX {
		tsParser = p.tsxPool.Get().(*sitter.Parser)
		defer p.tsxPool.Put(tsParser)
	} else {
		tsParser = p.tsPool.Get().(*sitter.Parser)
		defer p.tsPool.Put(tsParser)
	}

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("typescript parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	tsWalkTopLevel(tree.RootNode(), src, filePath, &res)
	return res, nil
}

// hasSuffix is a small helper to avoid importing path/filepath.
func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// tsWalkTopLevel walks top-level declarations in a TS/TSX file. It delegates to
// tsHandleDecl which extends jsHandleDecl with TypeScript-specific node types.
func tsWalkTopLevel(n *sitter.Node, src []byte, filePath string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		tsHandleDecl(n.Child(i), src, filePath, res)
	}
}

// tsHandleDecl handles a single top-level declaration, covering both the shared
// JS constructs (via jsHandleDecl) and TypeScript-specific ones.
func tsHandleDecl(n *sitter.Node, src []byte, filePath string, res *Result) {
	switch n.Type() {
	case "export_statement":
		// Unwrap export wrappers and recurse.
		for i := 0; i < int(n.ChildCount()); i++ {
			tsHandleDecl(n.Child(i), src, filePath, res)
		}

	case "function_declaration", "generator_function_declaration":
		sym := jsFunctionSymbol(n, src, filePath, "")
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
			res.CallEdges = append(res.CallEdges, jsExtractCalls(n, src, filePath, sym.Name)...)
		}

	case "class_declaration":
		sym := jsClassSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
			res.Symbols = append(res.Symbols, jsClassMembers(n, src, filePath, sym.Name)...)
		}

	case "lexical_declaration", "variable_declaration":
		syms := jsVarDeclSymbols(n, src, filePath)
		res.Symbols = append(res.Symbols, syms...)
		for _, s := range syms {
			res.CallEdges = append(res.CallEdges, jsExtractCalls(n, src, filePath, s.Name)...)
		}

	case "import_statement":
		edge := jsExtractImport(n, src, filePath)
		if edge != nil {
			res.ImportEdges = append(res.ImportEdges, *edge)
		}

	// TypeScript-specific constructs
	case "interface_declaration":
		sym := tsInterfaceSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "type_alias_declaration":
		sym := tsTypeAliasSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "enum_declaration":
		sym := tsEnumSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}
	}
}

func tsInterfaceSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "type_identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	sig := firstLine(nodeText(n, src))
	if idx := indexRune(sig, '{'); idx >= 0 {
		sig = trimRight(sig[:idx])
	}
	return &types.Symbol{
		ID:         symbolID(filePath, name, types.KindInterface),
		File:       filePath,
		Name:       name,
		Kind:       types.KindInterface,
		Signature:  sig,
		DocComment: jsLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func tsTypeAliasSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "type_identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	return &types.Symbol{
		ID:        symbolID(filePath, name, types.KindType),
		File:      filePath,
		Name:      name,
		Kind:      types.KindType,
		Signature: firstLine(nodeText(n, src)),
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
		Snippet:   truncateSnippet(nodeText(n, src)),
	}
}

func tsEnumSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	return &types.Symbol{
		ID:        symbolID(filePath, name, types.KindType),
		File:      filePath,
		Name:      name,
		Kind:      types.KindType,
		Signature: "enum " + name,
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
	}
}

// indexRune finds the first occurrence of r in s and returns the index.
func indexRune(s string, r rune) int {
	for i, c := range s {
		if c == r {
			return i
		}
	}
	return -1
}

// trimRight removes trailing whitespace from s.
func trimRight(s string) string {
	end := len(s)
	for end > 0 && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	return s[:end]
}
