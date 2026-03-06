package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/kotlin"

	"github.com/cavenine/ctxpp/internal/types"
)

// KotlinParser implements Parser for Kotlin source files using tree-sitter.
type KotlinParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewKotlinParser constructs a KotlinParser with a pooled tree-sitter parser.
func NewKotlinParser() *KotlinParser {
	lang := kotlin.GetLanguage()
	return &KotlinParser{
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

func (p *KotlinParser) Language() string { return "kotlin" }

func (p *KotlinParser) Extensions() []string {
	return []string{".kt", ".kts"}
}

// Parse extracts symbols and edges from a Kotlin source file.
func (p *KotlinParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("kotlin parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	packageName := kotlinPackageName(tree.RootNode(), src)
	var res Result
	kotlinWalkDecls(tree.RootNode(), src, filePath, packageName, "", &res)
	return res, nil
}

func kotlinWalkDecls(n *sitter.Node, src []byte, filePath, packageName, enclosingType string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		switch child.Type() {
		case "class_declaration":
			sym := kotlinExtractClassLike(child, src, filePath, packageName)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				nextEnclosing := enclosingType
				if sym.Kind == types.KindStruct || sym.Kind == types.KindInterface {
					nextEnclosing = sym.Name
				}
				kotlinWalkDecls(child, src, filePath, packageName, nextEnclosing, res)
			}
		case "function_declaration":
			sym := kotlinExtractFunction(child, src, filePath, packageName, enclosingType)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				res.CallEdges = append(res.CallEdges, kotlinExtractCalls(child, src, filePath, sym.Name)...)
			}
		case "property_declaration":
			res.Symbols = append(res.Symbols, kotlinExtractProperties(child, src, filePath, packageName, enclosingType)...)
		case "import_header":
			edge := kotlinExtractImport(child, src, filePath)
			if edge != nil {
				res.ImportEdges = append(res.ImportEdges, *edge)
			}
		default:
			kotlinWalkDecls(child, src, filePath, packageName, enclosingType, res)
		}
	}
}

func kotlinPackageName(root *sitter.Node, src []byte) string {
	for i := 0; i < int(root.ChildCount()); i++ {
		child := root.Child(i)
		if child.Type() != "package_header" {
			continue
		}
		id := childByType(child, "identifier")
		if id == nil {
			return ""
		}
		return nodeText(id, src)
	}
	return ""
}

func kotlinExtractClassLike(n *sitter.Node, src []byte, filePath, packageName string) *types.Symbol {
	nameNode := childByType(n, "type_identifier")
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	kind := types.KindStruct
	if hasChildType(n, "interface") {
		kind = types.KindInterface
	}
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
		DocComment: kotlinLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Package:    packageName,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func kotlinExtractFunction(n *sitter.Node, src []byte, filePath, packageName, enclosingType string) *types.Symbol {
	nameNode := childByType(n, "simple_identifier")
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	kind := types.KindFunction
	if enclosingType != "" {
		kind = types.KindMethod
	}
	sig := firstLine(nodeText(n, src))
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}
	idName := qualifiedMemberName(enclosingType, name)
	return &types.Symbol{
		ID:         symbolID(filePath, idName, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: kotlinLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   enclosingType,
		Package:    packageName,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func kotlinExtractProperties(n *sitter.Node, src []byte, filePath, packageName, enclosingType string) []types.Symbol {
	var out []types.Symbol
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		if child.Type() != "variable_declaration" {
			continue
		}
		for j := 0; j < int(child.ChildCount()); j++ {
			decl := child.Child(j)
			if decl.Type() != "simple_identifier" {
				continue
			}
			kind := types.KindVar
			if enclosingType != "" {
				kind = types.KindField
			}
			out = append(out, types.Symbol{
				ID:        symbolID(filePath, qualifiedMemberName(enclosingType, nodeText(decl, src)), kind),
				File:      filePath,
				Name:      nodeText(decl, src),
				Kind:      kind,
				Signature: firstLine(nodeText(n, src)),
				StartLine: int(n.StartPoint().Row) + 1,
				EndLine:   int(n.EndPoint().Row) + 1,
				Receiver:  enclosingType,
				Package:   packageName,
			})
		}
	}
	return out
}

func kotlinExtractImport(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	id := childByType(n, "identifier")
	if id == nil {
		return nil
	}
	return &types.ImportEdge{ImporterFile: filePath, ImportedPath: nodeText(id, src)}
}

func kotlinExtractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "call_expression" {
			calleeNode := childByType(node, "simple_identifier")
			if calleeNode != nil {
				callee := nodeText(calleeNode, src)
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

func kotlinLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && (prev.Type() == "line_comment" || prev.Type() == "multiline_comment") {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}

func hasChildType(n *sitter.Node, want string) bool {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		if child.Type() == want {
			return true
		}
	}
	return false
}

func qualifiedMemberName(receiver, name string) string {
	if receiver == "" {
		return name
	}
	return receiver + "." + name
}
