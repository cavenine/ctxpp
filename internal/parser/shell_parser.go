package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/bash"

	"github.com/cavenine/ctxpp/internal/types"
)

// ShellParser implements Parser for shell script files using tree-sitter's
// Bash grammar. It extracts function definitions with their leading comments.
type ShellParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewShellParser constructs a ShellParser with a pooled tree-sitter parser.
func NewShellParser() *ShellParser {
	lang := bash.GetLanguage()
	return &ShellParser{
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

func (p *ShellParser) Language() string     { return "shell" }
func (p *ShellParser) Extensions() []string { return []string{".sh", ".bash", ".dash", ".zsh"} }

// Parse extracts function definitions from shell scripts using tree-sitter.
func (p *ShellParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("shell parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	root := tree.RootNode()
	var res Result

	for i := 0; i < int(root.ChildCount()); i++ {
		child := root.Child(i)
		if child.Type() != "function_definition" {
			continue
		}

		name := shellFuncName(child, src)
		if name == "" {
			continue
		}

		// Build signature from the first line of the function definition.
		sig := firstLine(nodeText(child, src))

		// Collect leading comments by scanning previous siblings.
		doc := shellLeadingComment(child, src)

		res.Symbols = append(res.Symbols, types.Symbol{
			ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindFunction),
			File:       filePath,
			Name:       name,
			Kind:       types.KindFunction,
			Signature:  sig,
			DocComment: doc,
			StartLine:  int(child.StartPoint().Row) + 1,
			EndLine:    int(child.EndPoint().Row) + 1,
		})
	}

	return res, nil
}

// shellFuncName extracts the function name from a function_definition node.
// The name is stored in a "word" child that is not inside compound_statement.
func shellFuncName(n *sitter.Node, src []byte) string {
	// In tree-sitter-bash, function_definition children include:
	//   optional "function" keyword, "word" (name), optional "(", ")",
	//   "compound_statement" (body)
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == "word" {
			return nodeText(c, src)
		}
	}
	return ""
}

// shellLeadingComment collects consecutive comment siblings immediately
// preceding a function_definition node.
func shellLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && prev.Type() == "comment" {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
