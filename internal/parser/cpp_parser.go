package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/cpp"

	"github.com/cavenine/ctxpp/internal/types"
)

// CppParser implements Parser for C++ source files using tree-sitter.
// It handles all C constructs (via cHandleDecl) plus C++-specific nodes:
// class declarations, namespaces, templates, and out-of-line method definitions.
type CppParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewCppParser constructs a CppParser with a pooled tree-sitter parser.
func NewCppParser() *CppParser {
	lang := cpp.GetLanguage()
	return &CppParser{
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

func (p *CppParser) Language() string { return "cpp" }
func (p *CppParser) Extensions() []string {
	return []string{".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++"}
}

// Parse extracts symbols and edges from a C++ source file.
func (p *CppParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("cpp parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	cppWalkDecls(tree.RootNode(), src, filePath, "", &res)
	return res, nil
}

// cppWalkDecls iterates child nodes dispatching to cppHandleDecl. receiver is
// the enclosing class/struct name used to attribute methods.
func cppWalkDecls(n *sitter.Node, src []byte, filePath, receiver string, res *Result) {
	for i := 0; i < int(n.ChildCount()); i++ {
		cppHandleDecl(n.Child(i), src, filePath, receiver, res)
	}
}

// cppHandleDecl handles a single declaration node, covering C-compatible and
// C++-specific node types.
func cppHandleDecl(n *sitter.Node, src []byte, filePath, receiver string, res *Result) {
	switch n.Type() {
	// --- Shared with C (delegate to c_parser helpers) ---
	case "function_definition":
		sym := cppFunctionSymbol(n, src, filePath, receiver)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
			res.CallEdges = append(res.CallEdges, cExtractCalls(n, src, filePath, sym.Name)...)
		}

	case "declaration":
		syms := cDeclSymbols(n, src, filePath)
		res.Symbols = append(res.Symbols, syms...)

	case "type_definition":
		sym := cTypedefSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "enum_specifier":
		sym := cEnumSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "preproc_function_def":
		sym := cMacroSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}

	case "preproc_include":
		edge := cExtractInclude(n, src, filePath)
		if edge != nil {
			res.ImportEdges = append(res.ImportEdges, *edge)
		}

	// --- C++-specific ---
	case "class_specifier", "struct_specifier":
		sym := cppClassSymbol(n, src, filePath, receiver)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
			// Recurse into the class body to extract method declarations/definitions.
			body := n.ChildByFieldName("body")
			if body == nil {
				body = childByType(n, "field_declaration_list")
			}
			if body != nil {
				cppWalkDecls(body, src, filePath, sym.Name, res)
			}
		}

	case "field_declaration":
		// Method declarations inside a class body that are pure virtual or
		// simple member functions appear as field_declaration in the C++ grammar.
		// We surface them only if they look like function declarations (have a
		// function_declarator).
		syms := cppFieldDeclSymbols(n, src, filePath, receiver)
		res.Symbols = append(res.Symbols, syms...)

	case "namespace_definition":
		// namespace foo { ... }
		sym := cppNamespaceSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}
		body := n.ChildByFieldName("body")
		if body == nil {
			body = childByType(n, "declaration_list")
		}
		if body != nil {
			cppWalkDecls(body, src, filePath, receiver, res)
		}

	case "template_declaration":
		// template<...> function_definition | class_specifier
		// Recurse: the interesting node is the child after the parameter list.
		cppWalkDecls(n, src, filePath, receiver, res)

	case "access_specifier":
		// public:, private:, protected: — skip.

	case "using_declaration", "using_directive":
		edge := cppExtractUsing(n, src, filePath)
		if edge != nil {
			res.ImportEdges = append(res.ImportEdges, *edge)
		}

	case "enum_class_specifier":
		// enum class Color { ... };
		sym := cppEnumClassSymbol(n, src, filePath)
		if sym != nil {
			res.Symbols = append(res.Symbols, *sym)
		}
	}
}

// cppFunctionSymbol extracts a function from a function_definition. In C++ the
// declarator may carry a qualified name (e.g. "Widget::render"), in which case we
// extract the last segment as the symbol name and the qualifier as Receiver.
func cppFunctionSymbol(n *sitter.Node, src []byte, filePath, enclosingClass string) *types.Symbol {
	declNode := n.ChildByFieldName("declarator")
	if declNode == nil {
		return nil
	}
	fullName := cDeclaratorName(declNode, src)
	if fullName == "" {
		return nil
	}

	name, recv := cppSplitQualified(fullName)
	if recv == "" {
		recv = enclosingClass
	}

	kind := types.KindFunction
	if recv != "" {
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
		DocComment: cLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Receiver:   recv,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

// cppClassSymbol extracts a class_specifier or struct_specifier symbol.
func cppClassSymbol(n *sitter.Node, src []byte, filePath, pkg string) *types.Symbol {
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
		ID:         symbolID(filePath, name, types.KindStruct),
		File:       filePath,
		Name:       name,
		Kind:       types.KindStruct,
		Signature:  sig,
		DocComment: cLeadingComment(n, src),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
		Package:    pkg,
		Snippet:    truncateSnippet(nodeText(n, src)),
	}
}

// cppFieldDeclSymbols surfaces method declarations inside a class body.
// In the C++ grammar a member function declaration appears as field_declaration
// with a function_declarator nested inside its declarator.
func cppFieldDeclSymbols(n *sitter.Node, src []byte, filePath, receiver string) []types.Symbol {
	var syms []types.Symbol
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		if !containsFuncDeclarator(child) {
			continue
		}
		name := cDeclaratorName(child, src)
		if name == "" {
			continue
		}
		sig := strings.TrimRight(firstLine(nodeText(n, src)), " \t;")
		kind := types.KindMethod
		if receiver == "" {
			kind = types.KindFunction
		}
		syms = append(syms, types.Symbol{
			ID:        symbolID(filePath, name, kind),
			File:      filePath,
			Name:      name,
			Kind:      kind,
			Signature: sig,
			StartLine: int(n.StartPoint().Row) + 1,
			EndLine:   int(n.EndPoint().Row) + 1,
			Receiver:  receiver,
		})
	}
	return syms
}

// cppNamespaceSymbol extracts a namespace symbol.
func cppNamespaceSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		nameNode = childByType(n, "identifier")
	}
	if nameNode == nil {
		return nil // anonymous namespace
	}
	name := nodeText(nameNode, src)
	return &types.Symbol{
		ID:        symbolID(filePath, name, types.KindType),
		File:      filePath,
		Name:      name,
		Kind:      types.KindType,
		Signature: "namespace " + name,
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
	}
}

// cppEnumClassSymbol extracts an enum class / enum struct symbol.
func cppEnumClassSymbol(n *sitter.Node, src []byte, filePath string) *types.Symbol {
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
		Signature: "enum class " + name,
		StartLine: int(n.StartPoint().Row) + 1,
		EndLine:   int(n.EndPoint().Row) + 1,
	}
}

// cppExtractUsing extracts a using_declaration or using_directive as an import edge.
func cppExtractUsing(n *sitter.Node, src []byte, filePath string) *types.ImportEdge {
	// using std::string;  or  using namespace std;
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		switch c.Type() {
		case "qualified_identifier", "identifier", "namespace_identifier":
			return &types.ImportEdge{
				ImporterFile: filePath,
				ImportedPath: nodeText(c, src),
			}
		}
	}
	return nil
}

// cppSplitQualified splits a qualified name like "Widget::render" into
// ("render", "Widget"). If no qualifier is present it returns (name, "").
func cppSplitQualified(name string) (string, string) {
	idx := strings.LastIndex(name, "::")
	if idx < 0 {
		return name, ""
	}
	return name[idx+2:], name[:idx]
}
