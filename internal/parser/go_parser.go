package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"

	"github.com/cavenine/ctxpp/internal/types"
)

// GoParser implements Parser for Go source files.
// It pools tree-sitter parser instances via sync.Pool to avoid per-call allocation.
type GoParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewGoParser constructs a GoParser with a pooled tree-sitter parser.
func NewGoParser() *GoParser {
	lang := golang.GetLanguage()
	return &GoParser{
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

func (p *GoParser) Language() string     { return "go" }
func (p *GoParser) Extensions() []string { return []string{".go"} }

// Parse extracts symbols and edges from a Go source file.
// The underlying tree-sitter parser is obtained from a pool and returned after use.
func (p *GoParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("go parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	root := tree.RootNode()
	pkgName := extractPackageName(root, src)

	var res Result

	// Walk top-level declarations.
	for i := 0; i < int(root.ChildCount()); i++ {
		child := root.Child(i)
		switch child.Type() {
		case "function_declaration":
			sym := extractFunction(child, src, filePath, pkgName, "")
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				res.CallEdges = append(res.CallEdges, extractCalls(child, src, filePath, sym.Name)...)
			}
		case "method_declaration":
			sym := extractMethod(child, src, filePath, pkgName)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				res.CallEdges = append(res.CallEdges, extractCalls(child, src, filePath, sym.Name)...)
			}
		case "type_declaration":
			syms := extractTypeDecl(child, src, filePath, pkgName)
			res.Symbols = append(res.Symbols, syms...)
		case "const_declaration":
			syms := extractVarConst(child, src, filePath, pkgName, types.KindConst)
			res.Symbols = append(res.Symbols, syms...)
		case "var_declaration":
			syms := extractVarConst(child, src, filePath, pkgName, types.KindVar)
			res.Symbols = append(res.Symbols, syms...)
		case "import_declaration":
			res.ImportEdges = append(res.ImportEdges, extractImports(child, src, filePath)...)
		}
	}

	return res, nil
}

// ---- helpers ---------------------------------------------------------------

func nodeText(n *sitter.Node, src []byte) string {
	return string(src[n.StartByte():n.EndByte()])
}

func childByType(n *sitter.Node, typ string) *sitter.Node {
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == typ {
			return c
		}
	}
	return nil
}

func childrenByType(n *sitter.Node, typ string) []*sitter.Node {
	var out []*sitter.Node
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == typ {
			out = append(out, c)
		}
	}
	return out
}

func extractPackageName(root *sitter.Node, src []byte) string {
	clause := childByType(root, "package_clause")
	if clause == nil {
		return ""
	}
	id := childByType(clause, "package_identifier")
	if id == nil {
		return ""
	}
	return nodeText(id, src)
}

// leadingComment scans backwards from a node to collect a // or /* comment block.
func leadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && prev.Type() == "comment" {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}

func symbolID(filePath, name string, kind types.SymbolKind) string {
	return fmt.Sprintf("%s:%s:%s", filePath, name, kind)
}

// firstLine returns only the first line of a multi-line string.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// maxSnippetBytes is the maximum byte length of a Symbol.Snippet.
// Truncation is UTF-8 safe: we back off to the last valid rune boundary.
const maxSnippetBytes = 500

// truncateSnippet returns the first maxSnippetBytes bytes of s, truncated at
// a UTF-8 boundary. If s is shorter than the limit it is returned unchanged.
func truncateSnippet(s string) string {
	if len(s) <= maxSnippetBytes {
		return s
	}
	// Back off to avoid splitting a multi-byte rune.
	b := s[:maxSnippetBytes]
	for i := len(b) - 1; i >= len(b)-3 && i >= 0; i-- {
		if b[i] < 0x80 || b[i] >= 0xC0 {
			return b[:i+1]
		}
	}
	return b
}

func extractFunction(n *sitter.Node, src []byte, filePath, pkgName, receiver string) *types.Symbol {
	nameNode := childByType(n, "identifier")
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	if name == "" || name[0] < 'A' || name[0] > 'Z' {
		// Skip unexported for now — can make configurable later.
		// Actually include unexported too; useful for call graph.
		// Re-include:
	}

	sig := firstLine(nodeText(n, src))
	// Trim body brace onwards: keep only up to opening '{'.
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}

	kind := types.KindFunction
	if receiver != "" {
		kind = types.KindMethod
	}

	return &types.Symbol{
		ID:         symbolID(filePath, name, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: leadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   receiver,
		Package:    pkgName,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func extractMethod(n *sitter.Node, src []byte, filePath, pkgName string) *types.Symbol {
	// method_declaration: receiver parameter_list function_name parameters result? block
	receiver := ""
	recvNode := childByType(n, "parameter_list")
	if recvNode != nil {
		// The receiver type is the first parameter_declaration inside parameter_list.
		for i := 0; i < int(recvNode.ChildCount()); i++ {
			c := recvNode.Child(i)
			if c.Type() == "parameter_declaration" {
				// type is last named child of parameter_declaration
				for j := int(c.ChildCount()) - 1; j >= 0; j-- {
					tc := c.Child(j)
					if tc.IsNamed() {
						receiver = nodeText(tc, src)
						// Strip pointer prefix.
						receiver = strings.TrimPrefix(receiver, "*")
						break
					}
				}
				break
			}
		}
	}

	nameNode := childByType(n, "field_identifier")
	if nameNode == nil {
		// Some grammars use "identifier" for method name too.
		nameNode = childByType(n, "identifier")
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
		ID:         symbolID(filePath, receiver+"."+name, types.KindMethod),
		File:       filePath,
		Name:       name,
		Kind:       types.KindMethod,
		Signature:  sig,
		DocComment: leadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   receiver,
		Package:    pkgName,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func extractTypeDecl(n *sitter.Node, src []byte, filePath, pkgName string) []types.Symbol {
	var syms []types.Symbol
	for i := 0; i < int(n.ChildCount()); i++ {
		spec := n.Child(i)
		if spec.Type() != "type_spec" {
			continue
		}
		nameNode := childByType(spec, "type_identifier")
		if nameNode == nil {
			continue
		}
		name := nodeText(nameNode, src)

		// Determine kind from the type body node.
		kind := types.KindType
		for j := 0; j < int(spec.ChildCount()); j++ {
			c := spec.Child(j)
			switch c.Type() {
			case "struct_type":
				kind = types.KindStruct
			case "interface_type":
				kind = types.KindInterface
			}
		}

		sig := firstLine(nodeText(spec, src))
		syms = append(syms, types.Symbol{
			ID:         symbolID(filePath, name, kind),
			File:       filePath,
			Name:       name,
			Kind:       kind,
			Signature:  sig,
			DocComment: leadingComment(n, src),
			StartLine:  int(spec.StartPoint().Row) + 1,
			EndLine:    int(spec.EndPoint().Row) + 1,
			Package:    pkgName,
			Snippet:    truncateSnippet(nodeText(spec, src)),
		})
	}
	return syms
}

func extractVarConst(n *sitter.Node, src []byte, filePath, pkgName string, kind types.SymbolKind) []types.Symbol {
	var syms []types.Symbol
	for i := 0; i < int(n.ChildCount()); i++ {
		spec := n.Child(i)
		if spec.Type() != "const_spec" && spec.Type() != "var_spec" {
			continue
		}
		// Each spec may declare multiple names.
		for j := 0; j < int(spec.ChildCount()); j++ {
			c := spec.Child(j)
			if c.Type() != "identifier" {
				continue
			}
			name := nodeText(c, src)
			if name == "_" {
				continue
			}
			syms = append(syms, types.Symbol{
				ID:        symbolID(filePath, name, kind),
				File:      filePath,
				Name:      name,
				Kind:      kind,
				Signature: firstLine(nodeText(spec, src)),
				StartLine: int(spec.StartPoint().Row) + 1,
				EndLine:   int(spec.EndPoint().Row) + 1,
				Package:   pkgName,
				Snippet:   truncateSnippet(nodeText(spec, src)),
			})
		}
	}
	return syms
}

func extractImports(n *sitter.Node, src []byte, filePath string) []types.ImportEdge {
	var edges []types.ImportEdge
	// import_declaration may have import_spec_list or a single import_spec.
	var walkImportSpec func(node *sitter.Node)
	walkImportSpec = func(node *sitter.Node) {
		for i := 0; i < int(node.ChildCount()); i++ {
			c := node.Child(i)
			if c.Type() == "import_spec" {
				pathNode := childByType(c, "interpreted_string_literal")
				if pathNode == nil {
					pathNode = childByType(c, "raw_string_literal")
				}
				if pathNode != nil {
					path := strings.Trim(nodeText(pathNode, src), `"`+"`")
					edges = append(edges, types.ImportEdge{
						ImporterFile: filePath,
						ImportedPath: path,
					})
				}
			} else {
				walkImportSpec(c)
			}
		}
	}
	walkImportSpec(n)
	return edges
}

// extractCalls walks a function/method body and collects call expressions.
func extractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "call_expression" {
			funcNode := node.ChildByFieldName("function")
			if funcNode != nil {
				callee := calleeName(funcNode, src)
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

// calleeName extracts a short callee name from a function node in a call expression.
func calleeName(n *sitter.Node, src []byte) string {
	switch n.Type() {
	case "identifier":
		return nodeText(n, src)
	case "selector_expression":
		// e.g. pkg.Func or receiver.Method
		sel := n.ChildByFieldName("field")
		if sel != nil {
			return nodeText(sel, src)
		}
	}
	return ""
}
