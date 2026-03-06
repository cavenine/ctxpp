package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/csharp"

	"github.com/cavenine/ctxpp/internal/types"
)

// CSharpParser implements Parser for C# source files using tree-sitter.
type CSharpParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewCSharpParser constructs a CSharpParser with a pooled tree-sitter parser.
func NewCSharpParser() *CSharpParser {
	lang := csharp.GetLanguage()
	return &CSharpParser{
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

func (p *CSharpParser) Language() string     { return "csharp" }
func (p *CSharpParser) Extensions() []string { return []string{".cs"} }

// Parse extracts symbols and edges from a C# source file.
func (p *CSharpParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("csharp parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	namespace := csharpNamespace(tree.RootNode(), src)
	var res Result
	csharpWalkDecls(tree.RootNode(), src, filePath, namespace, "", &res)
	return res, nil
}

func csharpWalkDecls(n *sitter.Node, src []byte, filePath, namespaceName, enclosingType string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		switch child.Type() {
		case "class_declaration":
			sym := csharpExtractType(child, src, filePath, namespaceName, types.KindStruct)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				csharpWalkDecls(child, src, filePath, namespaceName, sym.Name, res)
			}
		case "interface_declaration":
			sym := csharpExtractType(child, src, filePath, namespaceName, types.KindInterface)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				csharpWalkDecls(child, src, filePath, namespaceName, sym.Name, res)
			}
		case "method_declaration":
			sym := csharpExtractMethod(child, src, filePath, namespaceName, enclosingType)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				res.CallEdges = append(res.CallEdges, csharpExtractCalls(child, src, filePath, sym.Name)...)
			}
		case "field_declaration":
			res.Symbols = append(res.Symbols, csharpExtractFields(child, src, filePath, namespaceName, enclosingType)...)
		case "using_directive":
			edge := csharpExtractImport(child, src, filePath)
			if edge != nil {
				res.ImportEdges = append(res.ImportEdges, *edge)
			}
		default:
			csharpWalkDecls(child, src, filePath, namespaceName, enclosingType, res)
		}
	}
}

func csharpNamespace(root *sitter.Node, src []byte) string {
	for i := 0; i < int(root.ChildCount()); i++ {
		child := root.Child(i)
		switch child.Type() {
		case "namespace_declaration", "file_scoped_namespace_declaration":
			name := csharpQualifiedName(child, src)
			if name != "" {
				return name
			}
		}
	}
	return ""
}

func csharpExtractType(n *sitter.Node, src []byte, filePath, namespaceName string, kind types.SymbolKind) *types.Symbol {
	nameNode := childByType(n, "identifier")
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
		DocComment: csharpLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Package:    namespaceName,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func csharpExtractMethod(n *sitter.Node, src []byte, filePath, namespaceName, enclosingType string) *types.Symbol {
	nameNode := childByType(n, "identifier")
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	sig := firstLine(nodeText(n, src))
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}
	return &types.Symbol{
		ID:         symbolID(filePath, name, types.KindMethod),
		File:       filePath,
		Name:       name,
		Kind:       types.KindMethod,
		Signature:  sig,
		DocComment: csharpLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   enclosingType,
		Package:    namespaceName,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func csharpExtractFields(n *sitter.Node, src []byte, filePath, namespaceName, enclosingType string) []types.Symbol {
	var out []types.Symbol
	varDecl := childByType(n, "variable_declaration")
	if varDecl == nil {
		return nil
	}
	for i := 0; i < int(varDecl.ChildCount()); i++ {
		child := varDecl.Child(i)
		if child.Type() != "variable_declarator" {
			continue
		}
		nameNode := childByType(child, "identifier")
		if nameNode == nil {
			continue
		}
		name := nodeText(nameNode, src)
		out = append(out, types.Symbol{
			ID:        symbolID(filePath, name, types.KindField),
			File:      filePath,
			Name:      name,
			Kind:      types.KindField,
			Signature: firstLine(nodeText(n, src)),
			StartLine: int(n.StartPoint().Row) + 1,
			EndLine:   int(n.EndPoint().Row) + 1,
			Receiver:  enclosingType,
			Package:   namespaceName,
		})
	}
	return out
}

func csharpExtractImport(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	name := csharpQualifiedName(n, src)
	if name == "" {
		return nil
	}
	return &types.ImportEdge{ImporterFile: filePath, ImportedPath: name}
}

func csharpExtractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "invocation_expression" {
			callee := csharpInvocationName(node, src)
			if callee != "" {
				edges = append(edges, types.CallEdge{
					CallerFile:   filePath,
					CallerSymbol: callerSymbol,
					CalleeSymbol: callee,
					Line:         int(node.StartPoint().Row) + 1,
				})
			}
		}
		for i := 0; i < int(node.ChildCount()); i++ {
			walk(node.Child(i))
		}
	}
	walk(n)
	return edges
}

func csharpQualifiedName(n *sitter.Node, src []byte) string {
	var parts []string
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node == nil {
			return
		}
		if node.Type() == "identifier" {
			parts = append(parts, nodeText(node, src))
		}
		for i := 0; i < int(node.ChildCount()); i++ {
			walk(node.Child(i))
		}
	}
	walk(n)
	return strings.Join(parts, ".")
}

func csharpInvocationName(n *sitter.Node, src []byte) string {
	if member := childByType(n, "member_access_expression"); member != nil {
		last := csharpLastIdentifier(member)
		if last != nil {
			return nodeText(last, src)
		}
	}
	identifiers := csharpIdentifiers(n)
	if len(identifiers) > 0 {
		return nodeText(identifiers[0], src)
	}
	return ""
}

func csharpLastIdentifier(n *sitter.Node) *sitter.Node {
	idents := csharpIdentifiers(n)
	if len(idents) == 0 {
		return nil
	}
	return idents[len(idents)-1]
}

func csharpIdentifiers(n *sitter.Node) []*sitter.Node {
	var out []*sitter.Node
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node == nil {
			return
		}
		if node.Type() == "identifier" {
			out = append(out, node)
		}
		for i := 0; i < int(node.ChildCount()); i++ {
			walk(node.Child(i))
		}
	}
	walk(n)
	return out
}

func csharpLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && prev.Type() == "comment" {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
