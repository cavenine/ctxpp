package parser

import (
	"context"
	"fmt"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	tree_sitter_markdown "github.com/smacker/go-tree-sitter/markdown/tree-sitter-markdown"

	"github.com/cavenine/ctxpp/internal/types"
)

// MarkdownParser implements Parser for Markdown files using tree-sitter's
// Markdown grammar. It extracts ATX headings (# through ######) as section
// symbols, with body text under each heading captured as DocComment.
type MarkdownParser struct{}

// NewMarkdownParser constructs a MarkdownParser.
func NewMarkdownParser() *MarkdownParser {
	return &MarkdownParser{}
}

func (p *MarkdownParser) Language() string     { return "markdown" }
func (p *MarkdownParser) Extensions() []string { return []string{".md", ".mdx"} }

// mdHeading is used internally during markdown parsing to track heading
// positions and body ranges before final Symbol construction.
type mdHeading struct {
	sym       types.Symbol
	bodyStart int // 0-indexed line where body begins (after heading)
}

// Parse extracts heading sections from Markdown source using tree-sitter.
func (p *MarkdownParser) Parse(filePath string, src []byte) (Result, error) {
	// tree-sitter-markdown block grammar is sufficient for heading extraction;
	// inline text content is accessible as raw bytes from the block tree.
	tsParser := sitter.NewParser()
	tsParser.SetLanguage(tree_sitter_markdown.GetLanguage())

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("markdown parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	root := tree.RootNode()

	// Collect all headings with their positions.
	var headings []mdHeading
	p.collectHeadings(root, src, filePath, &headings)

	// Determine body ranges: each heading's body extends from the line
	// after the heading to the line before the next heading (or EOF).
	lines := strings.Split(string(src), "\n")
	for i := range headings {
		bodyStart := headings[i].bodyStart
		var bodyEnd int
		if i+1 < len(headings) {
			bodyEnd = headings[i+1].sym.StartLine - 1 // 0-indexed line before next heading
		} else {
			bodyEnd = len(lines)
		}

		if bodyStart < bodyEnd {
			var body []string
			for j := bodyStart; j < bodyEnd; j++ {
				body = append(body, lines[j])
			}
			headings[i].sym.DocComment = strings.TrimSpace(strings.Join(body, "\n"))
		}

		headings[i].sym.EndLine = bodyEnd
		if headings[i].sym.EndLine < headings[i].sym.StartLine {
			headings[i].sym.EndLine = headings[i].sym.StartLine
		}
	}

	var res Result
	for _, h := range headings {
		res.Symbols = append(res.Symbols, h.sym)
	}
	return res, nil
}

// collectHeadings recursively walks the markdown block tree collecting
// atx_heading nodes.
func (p *MarkdownParser) collectHeadings(n *sitter.Node, src []byte, filePath string, out *[]mdHeading) {
	if n.Type() == "atx_heading" {
		title, sig := mdHeadingText(n, src)
		if title != "" {
			h := mdHeading{
				sym: types.Symbol{
					ID:        fmt.Sprintf("%s:%s:%s", filePath, title, types.KindSection),
					File:      filePath,
					Name:      title,
					Kind:      types.KindSection,
					Signature: sig,
					StartLine: int(n.StartPoint().Row) + 1,
				},
			}
			// If the heading ends with a newline (column 0 on next row),
			// body starts on EndPoint row; otherwise the next line.
			if n.EndPoint().Column == 0 {
				h.bodyStart = int(n.EndPoint().Row)
			} else {
				h.bodyStart = int(n.EndPoint().Row) + 1
			}
			*out = append(*out, h)
		}
		return // don't recurse into heading children
	}

	for i := 0; i < int(n.ChildCount()); i++ {
		p.collectHeadings(n.Child(i), src, filePath, out)
	}
}

// mdHeadingText extracts the heading title and full signature from an atx_heading node.
// Returns (title, signature) where signature includes the # markers.
func mdHeadingText(n *sitter.Node, src []byte) (string, string) {
	// The atx_heading node contains:
	//   - atx_h1_marker .. atx_h6_marker
	//   - inline (the heading text content)
	// The full node text gives us the signature (e.g. "## Foo").
	fullText := strings.TrimSpace(nodeText(n, src))

	// Extract just the inline text content.
	for i := 0; i < int(n.ChildCount()); i++ {
		c := n.Child(i)
		if c.Type() == "inline" {
			title := strings.TrimSpace(nodeText(c, src))
			return title, fullText
		}
	}

	return "", ""
}
