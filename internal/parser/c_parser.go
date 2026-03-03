package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/c"

	"github.com/cavenine/ctxpp/internal/types"
)

// CParser implements Parser for C source files using tree-sitter.
// It extracts functions, structs, enums, typedefs, and macro definitions.
type CParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewCParser constructs a CParser with a pooled tree-sitter parser.
func NewCParser() *CParser {
	lang := c.GetLanguage()
	return &CParser{
		lang: lang,
		pool: sync.Pool{
			New: func() any {
				p := sitter.NewParser()
				p.SetLanguage(lang)
				return p
			},
		},
	}
}

func (p *CParser) Language() string     { return "c" }
func (p *CParser) Extensions() []string { return []string{".c", ".h"} }

// Parse extracts symbols and edges from a C source file.
func (p *CParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("c parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	cWalkDecls(tree.RootNode(), src, filePath, &res)
	return res, nil
}

// cWalkDecls iterates the top-level children of a translation_unit extracting
// declarations of interest. It is also reused by the C++ parser for the shared
// C-compatible subset of nodes.
func cWalkDecls(n *sitter.Node, src []byte, filePath string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		cHandleDecl(child, src, filePath, res)
	}
}

// cHandleDecl dispatches a single declaration node for C. It recurses into
// preprocessor conditional blocks (preproc_ifdef, preproc_if, preproc_else,
// preproc_elif) so that declarations guarded by include-guards or feature
// macros are not missed.
func cHandleDecl(n *sitter.Node, src []byte, filePath string, res *Result) {
	switch n.Type() {
	// Preprocessor containers — recurse into children so we visit everything
	// nested inside #ifndef GUARD / #ifdef / #if / #else / #elif blocks.
	case "preproc_ifdef", "preproc_if", "preproc_else", "preproc_elif",
		"preproc_defined":
		cWalkDecls(n, src, filePath, res)

	case "function_definition":
		sym := cFunctionSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
			res.CallEdges = append(res.CallEdges, cExtractCalls(n, src, filePath, sym.Name)...)
		}

	case "declaration":
		// Covers: forward declarations of functions, extern declarations, and
		// global variable declarations. We only surface function declarations
		// (those with a function_declarator child).
		syms := cDeclSymbols(n, src, filePath)
		res.Symbols = append(res.Symbols, syms...)

	case "type_definition":
		// typedef struct { ... } Name; or typedef enum { ... } Name;
		sym := cTypedefSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "struct_specifier":
		// Named struct at top level (not inside typedef).
		sym := cStructSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "enum_specifier":
		sym := cEnumSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "preproc_function_def":
		// #define MACRO(args) body
		sym := cMacroSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "preproc_include":
		edge := cExtractInclude(n, src, filePath)
		if edge != nil {
			res.ImportEdges = append(res.ImportEdges, *edge)
		}
	}
}

// cFunctionSymbol extracts a symbol from a function_definition node.
func cFunctionSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	// function_definition fields: type, declarator, body
	declNode := n.ChildByFieldName("declarator")
	if declNode == nil {
		return nil
	}
	name := cDeclaratorName(declNode, src)
	if name == "" {
		return nil
	}
	// Signature: everything before the body block.
	sig := firstLine(nodeText(n, src))
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}
	return &types.Symbol{
		ID:         symbolID(filePath, name, types.KindFunction),
		File:       filePath,
		Name:       name,
		Kind:       types.KindFunction,
		Signature:  sig,
		DocComment: cLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

// cDeclSymbols extracts function forward declarations from a declaration node.
func cDeclSymbols(n *sitter.Node, src []byte, filePath string) []types.Symbol {
	var syms []types.Symbol
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		// A declarator that wraps a function_declarator indicates a function decl.
		if child.Type() == "function_declarator" || containsFuncDeclarator(child) {
			name := cDeclaratorName(child, src)
			if name == "" {
				continue
			}
			sig := strings.TrimRight(firstLine(nodeText(n, src)), " \t;")
			syms = append(syms, types.Symbol{
				ID:         symbolID(filePath, name, types.KindFunction),
				File:       filePath,
				Name:       name,
				Kind:       types.KindFunction,
				Signature:  sig,
				DocComment: cLeadingComment(n, src),
				StartLine:  int(n.StartPoint().Row) + 1,
				EndLine:    int(n.EndPoint().Row) + 1,
			})
		}
	}
	return syms
}

// containsFuncDeclarator reports whether n is or contains a function_declarator.
func containsFuncDeclarator(n *sitter.Node) bool {
	if n.Type() == "function_declarator" {
		return true
	}
	// pointer_declarator and other wrappers may nest around function_declarator.
	for i := 0; i < int(n.ChildCount()); i++ {
		if containsFuncDeclarator(n.Child(i)) {
			return true
		}
	}
	return false
}

// cTypedefSymbol extracts a symbol from a type_definition node.
// Examples:
//
//	typedef struct { int x; int y; } Point;
//	typedef enum   { RED, GREEN }    Color;
//	typedef int (*FnPtr)(int, int);
func cTypedefSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	// The last named child of a type_definition is typically the alias identifier.
	// tree-sitter-c names it via the "declarator" field.
	declNode := n.ChildByFieldName("declarator")
	name := ""
	if declNode != nil {
		name = nodeText(declNode, src)
	}
	if name == "" {
		// Fallback: last identifier child.
		for i := int(n.ChildCount()) - 1; i >= 0; i-- {
			c := n.Child(i)
			if c.Type() == "identifier" || c.Type() == "type_identifier" {
				name = nodeText(c, src)
				break
			}
		}
	}
	if name == "" {
		return nil
	}

	// Determine kind based on what the typedef wraps.
	kind := types.KindType
	for i := 0; i < int(n.ChildCount()); i++ {
		switch n.Child(i).Type() {
		case "struct_specifier":
			kind = types.KindStruct
		case "enum_specifier":
			kind = types.KindType // enums map to KindType
		}
	}

	sig := strings.TrimRight(firstLine(nodeText(n, src)), " \t;")
	return &types.Symbol{
		ID:         symbolID(filePath, name, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: cLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

// cStructSymbol extracts a symbol from a named struct_specifier at top level.
func cStructSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "type_identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	sig := firstLine(nodeText(n, src))
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}
	return &types.Symbol{
		ID:        symbolID(filePath, name, types.KindStruct),
		File:      filePath,
		Name:      name,
		Kind:      types.KindStruct,
		Signature: sig,
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
	}
}

// cEnumSymbol extracts a symbol from a named enum_specifier at top level.
func cEnumSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
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
		Signature: "enum " + name,
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
	}
}

// cMacroSymbol extracts a function-like macro definition.
func cMacroSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	return &types.Symbol{
		ID:        symbolID(filePath, name, types.KindFunction),
		File:      filePath,
		Name:      name,
		Kind:      types.KindFunction,
		Signature: firstLine(nodeText(n, src)),
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
	}
}

// cExtractInclude extracts an #include edge.
func cExtractInclude(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	// preproc_include: "#include" (system_lib_string | string_literal)
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		switch c.Type() {
		case "system_lib_string", "string_literal":
			path := strings.Trim(nodeText(c, src), "<>\"")
			return &types.ImportEdge{
				ImporterFile: filePath,
				ImportedPath: path,
			}
		}
	}
	return nil
}

// cExtractCalls walks a function body collecting call_expression nodes.
func cExtractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(*sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "call_expression" {
			funcNode := node.ChildByFieldName("function")
			if funcNode != nil {
				callee := cCalleeName(funcNode, src)
				if callee != "" {
					edges = append(edges, types.CallEdge{
						CallerFile:   filePath,
						CallerSymbol: callerSymbol,
						CalleeSymbol: callee,
						Line:         int(node.StartPoint().Row) + 1,
					})
				}
			}
		}
		for i := 0; i < int(node.ChildCount()); i++ {
			walk(node.Child(i))
		}
	}
	walk(n)
	return edges
}

func cCalleeName(n *sitter.Node, src []byte) string {
	switch n.Type() {
	case "identifier":
		return nodeText(n, src)
	case "field_expression":
		// obj->method or obj.method — return field name.
		field := n.ChildByFieldName("field")
		if field != nil {
			return nodeText(field, src)
		}
	}
	return ""
}

// cDeclaratorName extracts the function name from various declarator node shapes:
//
//	identifier                         → direct name
//	type_identifier / field_identifier → direct name
//	qualified_identifier               → full qualified name (e.g. "Widget::render")
//	function_declarator                → recurse on "declarator" field
//	pointer_declarator                 → recurse on "declarator" field
//	parenthesized_declarator           → recurse on child
func cDeclaratorName(n *sitter.Node, src []byte) string {
	switch n.Type() {
	case "identifier", "type_identifier", "field_identifier":
		return nodeText(n, src)
	case "qualified_identifier":
		// Build the full qualified name (e.g. "Widget::render") by joining the
		// text of all non-punctuation named children. The grammar emits nodes
		// like: namespace_identifier "::" identifier.
		return cQualifiedName(n, src)
	case "function_declarator", "pointer_declarator", "abstract_pointer_declarator",
		"parenthesized_declarator", "abstract_function_declarator":
		decl := n.ChildByFieldName("declarator")
		if decl != nil {
			return cDeclaratorName(decl, src)
		}
	}
	// Generic fallback: search children for identifier.
	for i := 0; i < int(n.ChildCount()); i++ {
		if name := cDeclaratorName(n.Child(i), src); name != "" {
			return name
		}
	}
	return ""
}

// cQualifiedName reconstructs a fully-qualified C++ name from a
// qualified_identifier node, e.g. "Widget::render" or "std::string".
// It concatenates all child node texts separated by "::".
func cQualifiedName(n *sitter.Node, src []byte) string {
	var parts []string
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		switch child.Type() {
		case "namespace_identifier", "identifier", "type_identifier":
			parts = append(parts, nodeText(child, src))
		case "qualified_identifier":
			// nested qualifier
			parts = append(parts, cQualifiedName(child, src))
		case "::":
			// separator — skip, we add it between parts
		}
	}
	return strings.Join(parts, "::")
}

// cLeadingComment returns // or /* comments immediately preceding the node.
func cLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && (prev.Type() == "comment") {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
