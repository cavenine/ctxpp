package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/rust"

	"github.com/cavenine/ctxpp/internal/types"
)

// RustParser implements Parser for Rust source files using tree-sitter.
// It extracts functions, structs, enums, traits, type aliases, and impl methods.
type RustParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewRustParser constructs a RustParser with a pooled tree-sitter parser.
func NewRustParser() *RustParser {
	lang := rust.GetLanguage()
	return &RustParser{
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

func (p *RustParser) Language() string     { return "rust" }
func (p *RustParser) Extensions() []string { return []string{".rs"} }

// Parse extracts symbols and edges from a Rust source file.
func (p *RustParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("rust parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	rustWalkDecls(tree.RootNode(), src, filePath, "", &res)
	return res, nil
}

// rustWalkDecls walks top-level (and nested) declarations extracting symbols.
// receiver is the name of any enclosing impl/trait context for method attribution.
func rustWalkDecls(n *sitter.Node, src []byte, filePath, receiver string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		switch child.Type() {
		case "function_item":
			sym := rustFunctionSymbol(child, src, filePath, receiver)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				res.CallEdges = append(res.CallEdges, rustExtractCalls(child, src, filePath, sym.Name)...)
			}

		case "struct_item":
			sym := rustNamedSymbol(child, src, filePath, types.KindStruct)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
			}

		case "enum_item":
			sym := rustNamedSymbol(child, src, filePath, types.KindType)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
			}

		case "trait_item":
			sym := rustNamedSymbol(child, src, filePath, types.KindInterface)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				// Walk trait body for method signatures.
				rustWalkDecls(child, src, filePath, sym.Name, res)
			}

		case "impl_item":
			// impl Foo { ... } or impl Trait for Foo { ... }
			// The type being implemented serves as the receiver for methods.
			implReceiver := rustImplReceiver(child, src)
			rustWalkDecls(child, src, filePath, implReceiver, res)

		case "type_item":
			sym := rustTypeAliasSymbol(child, src, filePath)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
			}

		case "use_declaration":
			edge := rustExtractImport(child, src, filePath)
			if edge != nil {
				res.ImportEdges = append(res.ImportEdges, *edge)
			}

		case "mod_item":
			// Recurse into inline modules.
			rustWalkDecls(child, src, filePath, receiver, res)

		case "declaration_list", "field_declaration_list":
			// Body nodes: recurse without changing receiver.
			rustWalkDecls(child, src, filePath, receiver, res)
		}
	}
}

func rustFunctionSymbol(n *sitter.Node, src []byte, filePath, receiver string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	kind := types.KindFunction
	if receiver != "" {
		kind = types.KindMethod
	}
	// Build signature: everything up to the body block.
	sig := firstLine(nodeText(n, src))
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}
	idName := qualifiedMemberName(receiver, name)
	return &types.Symbol{
		ID:         symbolID(filePath, idName, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: rustLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   receiver,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

// rustNamedSymbol extracts a struct/enum/trait declaration by looking for its
// "name" field node (an identifier or type_identifier).
func rustNamedSymbol(n *sitter.Node, src []byte, filePath string, kind types.SymbolKind) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "type_identifier")
		if nameNode == nil {
			nameNode = childByType(n, "identifier")
		}
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
		ID:         symbolID(filePath, name, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: rustLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func rustTypeAliasSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
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
	}
}

// rustImplReceiver returns the type name from an impl_item node.
// For "impl Foo { }" it returns "Foo". For "impl Bar for Foo { }" it returns "Foo".
func rustImplReceiver(n *sitter.Node, src []byte) string {
	// The "type" field in impl_item is the type being implemented.
	typeNode := n.ChildByFieldName("type")
	if typeNode != nil {
		return nodeText(typeNode, src)
	}
	// Fallback: look for type_identifier.
	tn := childByType(n, "type_identifier")
	if tn != nil {
		return nodeText(tn, src)
	}
	return ""
}

func rustExtractImport(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	// use_declaration contains a scoped_identifier, use_list, or identifier.
	// We grab the text of the whole use path as the imported path.
	if argument := n.ChildByFieldName("argument"); argument != nil {
		path := rustImportPath(argument, src)
		if path != "" {
			return &types.ImportEdge{ImporterFile: filePath, ImportedPath: path}
		}
	}
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		switch c.Type() {
		case "scoped_identifier", "identifier", "use_wildcard", "use_as_clause":
			path := rustImportPath(c, src)
			if path != "" {
				return &types.ImportEdge{ImporterFile: filePath, ImportedPath: path}
			}
		}
	}
	return nil
}

func rustImportPath(n *sitter.Node, src []byte) string {
	if n == nil {
		return ""
	}
	if n.Type() == "use_as_clause" {
		if path := n.ChildByFieldName("path"); path != nil {
			return strings.TrimSuffix(nodeText(path, src), "::*")
		}
	}
	if n.Type() == "use_wildcard" {
		return strings.TrimSuffix(nodeText(n, src), "::*")
	}
	return strings.TrimSuffix(nodeText(n, src), "::*")
}

// rustExtractCalls walks a function body collecting call_expression nodes.
func rustExtractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "call_expression" {
			funcNode := node.ChildByFieldName("function")
			if funcNode != nil {
				callee := rustCalleeName(funcNode, src)
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

func rustCalleeName(n *sitter.Node, src []byte) string {
	switch n.Type() {
	case "identifier":
		return nodeText(n, src)
	case "scoped_identifier":
		// e.g. module::function — return the last segment.
		last := childByType(n, "identifier")
		if last != nil {
			return nodeText(last, src)
		}
	case "field_expression":
		// e.g. self.method() — return the field name.
		field := n.ChildByFieldName("field")
		if field != nil {
			return nodeText(field, src)
		}
	}
	return ""
}

// rustLeadingComment returns /// or // or /* comments immediately preceding the node.
func rustLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && (prev.Type() == "line_comment" || prev.Type() == "block_comment" || prev.Type() == "doc_comment") {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
