package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/html"

	"github.com/cavenine/ctxpp/internal/types"
)

// HTMLParser implements Parser for HTML files using tree-sitter's HTML grammar.
// It extracts <title> and <h1>-<h6> heading elements as symbols.
type HTMLParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewHTMLParser constructs an HTMLParser with a pooled tree-sitter parser.
func NewHTMLParser() *HTMLParser {
	lang := html.GetLanguage()
	return &HTMLParser{
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

func (p *HTMLParser) Language() string     { return "html" }
func (p *HTMLParser) Extensions() []string { return []string{".html", ".htm"} }

// Parse extracts title and heading symbols from HTML source using tree-sitter.
func (p *HTMLParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("html parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	var res Result
	p.walkHTML(tree.RootNode(), src, filePath, &res)
	return res, nil
}

// walkHTML recursively walks the HTML AST, extracting title and heading symbols.
func (p *HTMLParser) walkHTML(n *sitter.Node, src []byte, filePath string, res *Result) {
	if n.Type() == "element" {
		tagName := htmlTagName(n, src)
		switch {
		case tagName == "title":
			content := htmlTextContent(n, src)
			if content != "" {
				res.Symbols = append(res.Symbols, types.Symbol{
					ID:        fmt.Sprintf("%s:%s:%s", filePath, content, types.KindElement),
					File:      filePath,
					Name:      content,
					Kind:      types.KindElement,
					Signature: fmt.Sprintf("<title>%s</title>", content),
					StartLine: int(n.StartPoint().Row) + 1,
					EndLine:   int(n.EndPoint().Row) + 1,
				})
			}
		case isHeadingTag(tagName):
			content := htmlTextContent(n, src)
			if content != "" {
				res.Symbols = append(res.Symbols, types.Symbol{
					ID:        fmt.Sprintf("%s:%s:%s", filePath, content, types.KindSection),
					File:      filePath,
					Name:      content,
					Kind:      types.KindSection,
					Signature: fmt.Sprintf("<%s>%s</%s>", tagName, content, tagName),
					StartLine: int(n.StartPoint().Row) + 1,
					EndLine:   int(n.EndPoint().Row) + 1,
				})
			}
		}
	}

	// Recurse into children.
	for i := 0; i < int(n.ChildCount()); i++ {
		p.walkHTML(n.Child(i), src, filePath, res)
	}
}

// htmlTagName extracts the tag name from an element's start_tag child.
func htmlTagName(elem *sitter.Node, src []byte) string {
	for i := 0; i < int(elem.ChildCount()); i++ {
		c := elem.Child(i)
		if c.Type() == "start_tag" || c.Type() == "self_closing_tag" {
			for j := 0; j < int(c.ChildCount()); j++ {
				gc := c.Child(j)
				if gc.Type() == "tag_name" {
					return strings.ToLower(nodeText(gc, src))
				}
			}
		}
	}
	return ""
}

// htmlTextContent extracts the trimmed text content from an element's text children.
// Strips any nested HTML tags by only collecting "text" nodes.
func htmlTextContent(elem *sitter.Node, src []byte) string {
	var parts []string
	for i := 0; i < int(elem.ChildCount()); i++ {
		c := elem.Child(i)
		if c.Type() == "text" {
			parts = append(parts, strings.TrimSpace(nodeText(c, src)))
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

// isHeadingTag returns true if the tag name is h1-h6.
func isHeadingTag(tag string) bool {
	switch tag {
	case "h1", "h2", "h3", "h4", "h5", "h6":
		return true
	}
	return false
}
