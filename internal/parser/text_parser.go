package parser

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/cavenine/ctxpp/internal/types"
)

// TextParser implements Parser for plain text and configuration files.
// It emits a single "document" symbol per file, with the first few lines
// as the signature and the full content as the doc comment for embedding.
type TextParser struct{}

// NewTextParser constructs a TextParser.
func NewTextParser() *TextParser {
	return &TextParser{}
}

func (p *TextParser) Language() string { return "text" }
func (p *TextParser) Extensions() []string {
	return []string{".txt", ".env", ".cfg", ".conf", ".ini", ".properties"}
}

// Filenames returns exact basenames this parser handles (extensionless files).
func (p *TextParser) Filenames() []string {
	return []string{"Makefile", "Dockerfile", "LICENSE", "Rakefile", "Vagrantfile", "Procfile"}
}

// Parse emits a single document symbol for the entire file.
func (p *TextParser) Parse(filePath string, src []byte) (Result, error) {
	content := strings.TrimSpace(string(src))
	if content == "" {
		return Result{}, nil
	}

	lines := strings.Split(content, "\n")
	name := filepath.Base(filePath)

	// Signature: first 5 lines or fewer.
	sigLines := lines
	if len(sigLines) > 5 {
		sigLines = sigLines[:5]
	}
	sig := strings.Join(sigLines, "\n")
	if len(sig) > 300 {
		sig = sig[:300]
	}

	// DocComment: full content (for embedding), capped at a reasonable size.
	doc := content
	if len(doc) > 4000 {
		doc = doc[:4000]
	}

	return Result{
		Symbols: []types.Symbol{
			{
				ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindDocument),
				File:       filePath,
				Name:       name,
				Kind:       types.KindDocument,
				Signature:  sig,
				DocComment: doc,
				StartLine:  1,
				EndLine:    len(lines),
			},
		},
	}, nil
}
