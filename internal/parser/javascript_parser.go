package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/javascript"

	"github.com/cavenine/ctxpp/internal/types"
)

// JavaScriptParser implements Parser for JavaScript source files using tree-sitter.
// It extracts functions, classes, methods, and arrow-function variable declarations.
type JavaScriptParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewJavaScriptParser constructs a JavaScriptParser with a pooled tree-sitter parser.
func NewJavaScriptParser() *JavaScriptParser {
	lang := javascript.GetLanguage()
	return &JavaScriptParser{
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

func (p *JavaScriptParser) Language() string     { return "javascript" }
func (p *JavaScriptParser) Extensions() []string { return []string{".js", ".mjs", ".cjs", ".jsx"} }

// Parse extracts symbols and edges from a JavaScript source file.
func (p *JavaScriptParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("javascript parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	jsWalkTopLevel(tree.RootNode(), src, filePath, &res)
	return res, nil
}

// jsWalkTopLevel walks top-level and export-wrapped declarations in a JS/TS file.
// It is shared between JavaScriptParser and TypeScriptParser.
func jsWalkTopLevel(n *sitter.Node, src []byte, filePath string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		jsHandleDecl(child, src, filePath, res)
	}
}

// jsHandleDecl dispatches a single declaration node, unwrapping export wrappers.
func jsHandleDecl(n *sitter.Node, src []byte, filePath string, res *Result) {
	switch n.Type() {
	case "export_statement":
		// Unwrap: export default? <declaration>
		for i := 0; i < int(n.ChildCount()); i++ {
			jsHandleDecl(n.Child(i), src, filePath, res)
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
		// const Foo = () => {} or const Foo = function() {}
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
	}
}

func jsFunctionSymbol(n *sitter.Node, src []byte, filePath, receiver string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "identifier")
	}
	if nameNode == nil {
		return nil
	}
	name := nodeText(nameNode, src)
	if name == "" {
		return nil
	}
	kind := types.KindFunction
	if receiver != "" {
		kind = types.KindMethod
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
		DocComment: jsLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   receiver,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

func jsClassSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "identifier")
		if nameNode == nil {
			nameNode = childByType(n, "type_identifier")
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
		ID:         symbolID(filePath, name, types.KindStruct),
		File:       filePath,
		Name:       name,
		Kind:       types.KindStruct,
		Signature:  sig,
		DocComment: jsLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

// jsClassMembers extracts method definitions from a class body.
func jsClassMembers(classNode *sitter.Node, src []byte, filePath, className string) []types.Symbol {
	var syms []types.Symbol
	body := classNode.ChildByFieldName("body")
	if body == nil {
		body = childByType(classNode, "class_body")
	}
	if body == nil {
		return nil
	}
	for i := 0; i < int(body.ChildCount()); i++ {
		member := body.Child(i)
		switch member.Type() {
		case "method_definition":
			nameNode := member.ChildByFieldName("name")
			if nameNode == nil {
				continue
			}
			name := nodeText(nameNode, src)
			if name == "" {
				continue
			}
			sig := firstLine(nodeText(member, src))
			if idx := strings.Index(sig, "{"); idx >= 0 {
				sig = strings.TrimSpace(sig[:idx])
			}
			syms = append(syms, types.Symbol{
				ID:        symbolID(filePath, qualifiedMemberName(className, name), types.KindMethod),
				File:      filePath,
				Name:      name,
				Kind:      types.KindMethod,
				Signature: sig,
				StartLine: int(member.StartPoint().Row) + 1,
				EndLine:   int(member.EndPoint().Row) + 1,
				Receiver:  className,
				Snippet:   truncateSnippet(nodeText(member, src)),
			})
		}
	}
	return syms
}

// jsVarDeclSymbols extracts symbols from const/let/var declarations where the
// RHS is a function or arrow function.
func jsVarDeclSymbols(n *sitter.Node, src []byte, filePath string) []types.Symbol {
	var syms []types.Symbol
	for i := 0; i < int(n.ChildCount()); i++ {
		decl := n.Child(i)
		if decl.Type() != "variable_declarator" {
			continue
		}
		nameNode := decl.ChildByFieldName("name")
		if nameNode == nil {
			continue
		}
		name := nodeText(nameNode, src)
		valNode := decl.ChildByFieldName("value")
		if valNode == nil {
			continue
		}
		switch valNode.Type() {
		case "arrow_function", "function", "function_expression", "generator_function":
			sig := firstLine(nodeText(n, src))
			if idx := strings.Index(sig, "=>"); idx >= 0 {
				// Include up to and including "=>" but not the body.
				rhs := sig[idx+2:]
				if bi := strings.Index(rhs, "{"); bi >= 0 {
					sig = strings.TrimSpace(sig[:idx+2])
				} else {
					sig = strings.TrimSpace(sig)
				}
			} else if idx := strings.Index(sig, "{"); idx >= 0 {
				sig = strings.TrimSpace(sig[:idx])
			}
			syms = append(syms, types.Symbol{
				ID:         symbolID(filePath, name, types.KindFunction),
				File:       filePath,
				Name:       name,
				Kind:       types.KindFunction,
				Signature:  sig,
				DocComment: jsLeadingComment(n, src),
				StartLine:  int(n.StartPoint().Row) + 1,
				EndLine:    int(n.EndPoint().Row) + 1,
				Snippet:    truncateSnippet(nodeText(valNode, src)),
			})
		}
	}
	return syms
}

func jsExtractImport(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	// import_statement: "import" ... "from" string_literal
	sourceNode := n.ChildByFieldName("source")
	if sourceNode == nil {
		// Fall back: last string node
		for i := int(n.ChildCount()) - 1; i >= 0; i-- {
			c := n.Child(i)
			if c.Type() == "string" {
				sourceNode = c
				break
			}
		}
	}
	if sourceNode == nil {
		return nil
	}
	path := strings.Trim(nodeText(sourceNode, src), `"'`)
	return &types.ImportEdge{
		ImporterFile: filePath,
		ImportedPath: path,
	}
}

// jsExtractCalls walks a subtree and collects call_expression nodes.
func jsExtractCalls(n *sitter.Node, src []byte, filePath, callerSymbol string) []types.CallEdge {
	var edges []types.CallEdge
	var walk func(node *sitter.Node)
	walk = func(node *sitter.Node) {
		if node.Type() == "call_expression" {
			funcNode := node.ChildByFieldName("function")
			if funcNode != nil {
				callee := jsCalleeName(funcNode, src)
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

func jsCalleeName(n *sitter.Node, src []byte) string {
	switch n.Type() {
	case "identifier":
		return nodeText(n, src)
	case "member_expression":
		// e.g. obj.method — use the property (right-hand) name.
		prop := n.ChildByFieldName("property")
		if prop != nil {
			return nodeText(prop, src)
		}
	}
	return ""
}

// jsLeadingComment returns a // or /* comment block immediately preceding the node.
func jsLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && (prev.Type() == "comment") {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
