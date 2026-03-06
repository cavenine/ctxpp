package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/java"

	"github.com/cavenine/ctxpp/internal/types"
)

// JavaParser implements Parser for Java source files using tree-sitter.
type JavaParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewJavaParser constructs a JavaParser with a pooled tree-sitter parser.
func NewJavaParser() *JavaParser {
	lang := java.GetLanguage()
	return &JavaParser{
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

func (p *JavaParser) Language() string     { return "java" }
func (p *JavaParser) Extensions() []string { return []string{".java"} }

// Parse extracts symbols and edges from a Java source file.
func (p *JavaParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("java parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	javaWalkDecls(tree.RootNode(), src, filePath, "", &res)
	return res, nil
}

// javaWalkDecls recursively walks the tree looking for class/interface/enum
// declarations and method/field declarations inside them. pkg is the current
// enclosing class name (used as Package on symbols).
func javaWalkDecls(n *sitter.Node, src []byte, filePath, enclosingClass string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		switch child.Type() {
		case "class_declaration", "record_declaration":
			sym := javaExtractClass(child, src, filePath, enclosingClass)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				javaWalkDecls(child, src, filePath, sym.Name, res)
			}
		case "interface_declaration":
			sym := javaExtractInterface(child, src, filePath, enclosingClass)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				javaWalkDecls(child, src, filePath, sym.Name, res)
			}
		case "enum_declaration":
			sym := javaExtractEnum(child, src, filePath, enclosingClass)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
			}
		case "method_declaration", "constructor_declaration":
			sym := javaExtractMethod(child, src, filePath, enclosingClass)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
				res.CallEdges = append(res.CallEdges, javaExtractCalls(child, src, filePath, sym.Name)...)
			}
		case "field_declaration":
			syms := javaExtractFields(child, src, filePath, enclosingClass)
			res.Symbols = append(res.Symbols, syms...)
		case "import_declaration":
			edge := javaExtractImport(child, src, filePath)
			if edge != nil {
				res.ImportEdges = append(res.ImportEdges, *edge)
			}
		case "class_body", "interface_body", "enum_body":
			// Recurse into the body node.
			javaWalkDecls(child, src, filePath, enclosingClass, res)
		default:
			javaWalkDecls(child, src, filePath, enclosingClass, res)
		}
	}
}

func javaExtractClass(n *sitter.Node, src []byte, filePath, pkg string) *types.Symbol {
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
		ID:         symbolID(filePath, name, types.KindStruct),
		File:       filePath,
		Name:       name,
		Kind:       types.KindStruct,
		Signature:  sig,
		DocComment: javaLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Package:    pkg,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func javaExtractInterface(n *sitter.Node, src []byte, filePath, pkg string) *types.Symbol {
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
		ID:         symbolID(filePath, name, types.KindInterface),
		File:       filePath,
		Name:       name,
		Kind:       types.KindInterface,
		Signature:  sig,
		DocComment: javaLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Package:    pkg,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func javaExtractEnum(n *sitter.Node, src []byte, filePath, pkg string) *types.Symbol {
	nameNode := childByType(n, "identifier")
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
		Package:   pkg,
	}
}

func javaExtractMethod(n *sitter.Node, src []byte, filePath, receiver string) *types.Symbol {
	nameNode := childByType(n, "identifier")
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	sig := firstLine(nodeText(n, src))
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}
	kind := types.KindMethod
	if n.Type() == "constructor_declaration" {
		kind = types.KindFunction
	}
	idName := qualifiedMemberName(receiver, name)
	return &types.Symbol{
		ID:         symbolID(filePath, idName, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: javaLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   receiver,
		Package:    receiver,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func javaExtractFields(n *sitter.Node, src []byte, filePath, pkg string) []types.Symbol {
	var syms []types.Symbol
	// field_declaration: modifiers? type declarator_list ;
	// variable_declarator nodes hold each field name.
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() != "variable_declarator" {
			continue
		}
		nameNode := childByType(c, "identifier")
		if nameNode == nil {
			continue
		}
		name := nodeText(nameNode, src)
		syms = append(syms, types.Symbol{
			ID:        symbolID(filePath, qualifiedMemberName(pkg, name), types.KindField),
			File:      filePath,
			Name:      name,
			Kind:      types.KindField,
			Signature: firstLine(nodeText(n, src)),
			StartLine: int(n.StartPoint().Row) + 1,
			EndLine:   int(n.EndPoint().Row) + 1,
			Receiver:  pkg,
			Package:   pkg,
		})
	}
	return syms
}

func javaExtractImport(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	// import_declaration contains a scoped_identifier or identifier.
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == "scoped_identifier" || c.Type() == "identifier" {
			return &types.ImportEdge{
				ImporterFile: filePath,
				ImportedPath: nodeText(c, src),
			}
		}
	}
	return nil
}

// javaExtractCalls walks a method body and collects method invocations.
func javaExtractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "method_invocation" {
			// method_invocation: [object "." ] name "(" args ")"
			// The method name is the identifier child that is not after a '.' at
			// the argument list position — tree-sitter names it "name".
			nameNode := node.ChildByFieldName("name")
			if nameNode == nil {
				nameNode = childByType(node, "identifier")
			}
			if nameNode != nil {
				callee := nodeText(nameNode, src)
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

// javaLeadingComment returns a Javadoc or line comment block immediately
// preceding the node.
func javaLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && (prev.Type() == "line_comment" || prev.Type() == "block_comment") {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
