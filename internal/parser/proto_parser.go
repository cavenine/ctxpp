package parser

import (
	"context"
	"fmt"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/protobuf"

	"github.com/cavenine/ctxpp/internal/types"
)

// ProtoParser implements Parser for Protocol Buffer definition files using
// tree-sitter's protobuf grammar. It extracts service, rpc, message, and
// enum definitions with their leading comments.
type ProtoParser struct {
	lang *sitter.Language
	pool sync.Pool
}

// NewProtoParser constructs a ProtoParser with a pooled tree-sitter parser.
func NewProtoParser() *ProtoParser {
	lang := protobuf.GetLanguage()
	return &ProtoParser{
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

func (p *ProtoParser) Language() string     { return "protobuf" }
func (p *ProtoParser) Extensions() []string { return []string{".proto"} }

// Parse extracts symbols from a .proto file.
//
// Extracted symbol types:
//   - service  → KindInterface (each service definition)
//   - rpc      → KindMethod    (each rpc within a service, named "Service.Method")
//   - message  → KindType      (each message definition)
//   - enum     → KindType      (each enum definition)
func (p *ProtoParser) Parse(filePath string, src []byte) (Result, error) {
	tsParser := p.pool.Get().(*sitter.Parser)
	defer p.pool.Put(tsParser)

	tree, err := tsParser.ParseCtx(context.Background(), nil, src)
	if err != nil {
		return Result{}, fmt.Errorf("proto parser: parse %s: %w", filePath, err)
	}
	defer tree.Close()

	root := tree.RootNode()
	var res Result

	for i := 0; i < int(root.ChildCount()); i++ {
		child := root.Child(i)
		switch child.Type() {
		case "service":
			p.extractService(filePath, child, src, &res)
		case "message":
			p.extractMessage(filePath, child, src, &res)
		case "enum":
			p.extractEnum(filePath, child, src, &res)
		}
	}

	return res, nil
}

// extractService creates a symbol for the service and symbols for each rpc method.
func (p *ProtoParser) extractService(filePath string, n *sitter.Node, src []byte, res *Result) {
	nameNode := childByType(n, "service_name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	doc := protoLeadingComment(n, src)

	res.Symbols = append(res.Symbols, types.Symbol{
		ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindInterface),
		File:       filePath,
		Name:       name,
		Kind:       types.KindInterface,
		Signature:  "service " + name,
		DocComment: doc,
		Snippet:    truncateSnippet(nodeText(n, src)),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
	})

	// Extract rpc methods within the service.
	for i := 0; i < int(n.ChildCount()); i++ {
		child := n.Child(i)
		if child.Type() == "rpc" {
			p.extractRPC(filePath, name, child, src, res)
		}
	}
}

// extractRPC creates a symbol for an rpc method, qualified with its service name.
func (p *ProtoParser) extractRPC(filePath, serviceName string, n *sitter.Node, src []byte, res *Result) {
	nameNode := childByType(n, "rpc_name")
	if nameNode == nil {
		return
	}
	rpcName := nodeText(nameNode, src)
	qualifiedName := serviceName + "." + rpcName
	doc := protoLeadingComment(n, src)
	sig := firstLine(nodeText(n, src))

	res.Symbols = append(res.Symbols, types.Symbol{
		ID:         fmt.Sprintf("%s:%s:%s", filePath, qualifiedName, types.KindMethod),
		File:       filePath,
		Name:       qualifiedName,
		Kind:       types.KindMethod,
		Signature:  sig,
		DocComment: doc,
		Receiver:   serviceName,
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
	})
}

// extractMessage creates a symbol for a message definition.
func (p *ProtoParser) extractMessage(filePath string, n *sitter.Node, src []byte, res *Result) {
	nameNode := childByType(n, "message_name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	doc := protoLeadingComment(n, src)

	res.Symbols = append(res.Symbols, types.Symbol{
		ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindType),
		File:       filePath,
		Name:       name,
		Kind:       types.KindType,
		Signature:  "message " + name,
		DocComment: doc,
		Snippet:    truncateSnippet(nodeText(n, src)),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
	})
}

// extractEnum creates a symbol for an enum definition.
func (p *ProtoParser) extractEnum(filePath string, n *sitter.Node, src []byte, res *Result) {
	nameNode := childByType(n, "enum_name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	doc := protoLeadingComment(n, src)

	res.Symbols = append(res.Symbols, types.Symbol{
		ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindType),
		File:       filePath,
		Name:       name,
		Kind:       types.KindType,
		Signature:  "enum " + name,
		DocComment: doc,
		Snippet:    truncateSnippet(nodeText(n, src)),
		StartLine:  int(n.StartPoint().Row) + 1,
		EndLine:    int(n.EndPoint().Row) + 1,
	})
}

// protoLeadingComment collects consecutive // comment lines immediately
// preceding a node. In protobuf tree-sitter, comments are siblings of the
// node they document (not children).
func protoLeadingComment(n *sitter.Node, src []byte) string {
	var lines []string
	prev := n.PrevNamedSibling()
	for prev != nil && prev.Type() == "comment" {
		lines = append([]string{nodeText(prev, src)}, lines...)
		prev = prev.PrevNamedSibling()
	}
	return strings.Join(lines, "\n")
}
