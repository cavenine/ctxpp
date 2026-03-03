package parser

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/sql"

	"github.com/cavenine/ctxpp/internal/types"
)

// SQLParser implements Parser for SQL files using tree-sitter's SQL grammar.
// It extracts CREATE TABLE, VIEW, INDEX, FUNCTION, TRIGGER statements from
// the AST. CREATE PROCEDURE is handled via regex fallback since tree-sitter-sql
// does not have a dedicated node type for it.
type SQLParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewSQLParser constructs a SQLParser with a pooled tree-sitter parser.
func NewSQLParser() *SQLParser {
	lang := sql.GetLanguage()
	return &SQLParser{
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

func (p *SQLParser) Language() string     { return "sql" }
func (p *SQLParser) Extensions() []string { return []string{".sql"} }

// Parse extracts DDL symbols from SQL source using tree-sitter.
func (p *SQLParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("sql parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	root := tree.RootNode()
	var res Result

	for i := 0; i < int(root.ChildCount()); i++ {
		child := root.Child(i)

		switch child.Type() {
		case "statement":
			// A statement wraps one of the create_* node types.
			if child.NamedChildCount() == 0 {
				continue
			}
			inner := child.NamedChild(0)
			sym := p.extractCreateSymbol(inner, src, filePath, root, i)
			if sym != nil {
				res.Symbols = append(res.Symbols, *sym)
			}
		case "ERROR":
			// Fallback: try regex on ERROR node text for CREATE PROCEDURE
			// and other statements tree-sitter-sql can't parse.
			syms := p.extractFromError(child, src, filePath, root, i)
			res.Symbols = append(res.Symbols, syms...)
		}
	}

	return res, nil
}

// extractCreateSymbol maps a tree-sitter create_* node to a Symbol.
func (p *SQLParser) extractCreateSymbol(n *sitter.Node, src []byte, filePath string, root *sitter.Node, sibIdx int) *types.Symbol {
	var kind types.SymbolKind
	var name string

	switch n.Type() {
	case "create_table":
		kind = types.KindTable
		name = sqlObjectName(n, src)
	case "create_view", "create_materialized_view":
		kind = types.KindView
		name = sqlObjectName(n, src)
	case "create_index":
		kind = types.KindIndex
		// create_index has a direct identifier child for the index name.
		name = sqlDirectIdentifier(n, src)
	case "create_function":
		kind = types.KindFunction
		name = sqlObjectName(n, src)
	case "create_trigger":
		kind = types.KindTrigger
		name = sqlObjectName(n, src)
	default:
		return nil
	}

	if name == "" {
		return nil
	}

	sig := firstLine(nodeText(n, src))
	if len(sig) > 200 {
		sig = sig[:200]
	}

	doc := sqlLeadingComment(root, sibIdx, src)

	return &types.Symbol{
		ID:         fmt.Sprintf("%s:%s:%s", filePath, name, kind),
		File:       filePath,
		Name:       name,
		Kind:       kind,
		Signature:  sig,
		DocComment: doc,
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
	}
}

// sqlObjectName extracts the name from the first object_reference child.
func sqlObjectName(n *sitter.Node, src []byte) string {
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == "object_reference" {
			// object_reference contains one or more identifiers (schema.name).
			// Take the last identifier as the unqualified name.
			var last *sitter.Node
			for j := 0; j < int(c.ChildCount()); j++ {
				gc := c.Child(j)
				if gc.Type() == "identifier" {
					last = gc
				}
			}
			if last != nil {
				return nodeText(last, src)
			}
		}
	}
	return ""
}

// sqlDirectIdentifier extracts the first direct identifier child (used by create_index).
func sqlDirectIdentifier(n *sitter.Node, src []byte) string {
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == "identifier" {
			return nodeText(c, src)
		}
	}
	return ""
}

// sqlLeadingComment scans siblings before sibIdx in root to collect comment nodes.
func sqlLeadingComment(root *sitter.Node, sibIdx int, src []byte) string {
	var lines []string
	for j := sibIdx - 1; j >= 0; j-- {
		sib := root.Child(j)
		if sib.Type() == "comment" {
			lines = append([]string{nodeText(sib, src)}, lines...)
		} else {
			break
		}
	}
	return strings.Join(lines, "\n")
}

// createProcedurePattern matches CREATE [OR REPLACE] PROCEDURE [schema.]<name>
// as a fallback for ERROR nodes from tree-sitter-sql.
var createProcedurePattern = regexp.MustCompile(
	`(?i)^CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(?:\w+\.)?(\w+)`,
)

// extractFromError handles ERROR nodes by applying regex fallback.
func (p *SQLParser) extractFromError(n *sitter.Node, src []byte, filePath string, root *sitter.Node, sibIdx int) []types.Symbol {
	text := nodeText(n, src)
	m := createProcedurePattern.FindStringSubmatch(text)
	if m == nil {
		return nil
	}

	name := m[1]
	sig := firstLine(text)
	if len(sig) > 200 {
		sig = sig[:200]
	}

	doc := sqlLeadingComment(root, sibIdx, src)

	// EndLine: scan forward to find the statement end (semicolons following
	// the ERROR node in the tree, or use the last sibling's end row).
	endRow := int(n.EndPoint().Row)
	// Look at subsequent siblings for block/semicolon to get a better end.
	for j := sibIdx + 1; j < int(root.ChildCount()); j++ {
		sib := root.Child(j)
		if sib.Type() == ";" {
			endRow = int(sib.EndPoint().Row)
			break
		}
		if sib.Type() == "block" || sib.Type() == "ERROR" {
			endRow = int(sib.EndPoint().Row)
			// Keep scanning for the trailing semicolon.
			continue
		}
		break
	}

	return []types.Symbol{{
		ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindProcedure),
		File:       filePath,
		Name:       name,
		Kind:       types.KindProcedure,
		Signature:  sig,
		DocComment: doc,
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    endRow + 1,
	}}
}
