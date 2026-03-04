package parser

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/cavenine/ctxpp/internal/types"
)

// HTTPParser implements Parser for .http / .rest files (REST Client format).
// It extracts HTTP request definitions (GET, POST, PUT, DELETE, PATCH, etc.)
// as function symbols, using the method + path as the symbol name.
type HTTPParser struct{}

// NewHTTPParser constructs an HTTPParser.
func NewHTTPParser() *HTTPParser {
	return &HTTPParser{}
}

func (p *HTTPParser) Language() string     { return "http" }
func (p *HTTPParser) Extensions() []string { return []string{".http", ".rest"} }

// httpMethodPattern matches a line starting with an HTTP method and URL.
var httpMethodPattern = regexp.MustCompile(
	`^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(https?://\S+)`,
)

// Parse extracts HTTP request definitions from .http files.
// Requests are separated by ### lines. Each request becomes a function symbol
// with name "METHOD /path".
func (p *HTTPParser) Parse(filePath string, src []byte) (Result, error) {
	lines := strings.Split(string(src), "\n")
	var res Result
	var pendingComments []string

	for i := 0; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])

		// ### separators reset comment context.
		if strings.HasPrefix(trimmed, "###") {
			// Capture the ### line as a comment if it has text after ###.
			text := strings.TrimSpace(strings.TrimPrefix(trimmed, "###"))
			if text != "" {
				pendingComments = append(pendingComments, text)
			}
			continue
		}

		// Collect # comments.
		if strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "//") {
			pendingComments = append(pendingComments, trimmed)
			continue
		}

		// Skip blank lines.
		if trimmed == "" {
			continue
		}

		m := httpMethodPattern.FindStringSubmatch(trimmed)
		if m == nil {
			// Non-comment, non-method line — could be headers or body.
			// Don't reset comments here; they belong to the next request.
			continue
		}

		method := m[1]
		rawURL := m[2]

		// Extract path from URL, stripping query params for the name.
		path := extractPath(rawURL)
		name := method + " " + path

		startLine := i + 1 // 1-based

		// Find the end of this request block (next ### or next request line or EOF).
		endLine := findRequestEnd(lines, i+1)

		doc := strings.Join(pendingComments, "\n")

		res.Symbols = append(res.Symbols, types.Symbol{
			ID:         fmt.Sprintf("%s:%s:%s", filePath, name, types.KindFunction),
			File:       filePath,
			Name:       name,
			Kind:       types.KindFunction,
			Signature:  trimmed,
			DocComment: doc,
			StartLine:  startLine,
			EndLine:    endLine + 1, // 1-based
		})

		pendingComments = nil
	}

	return res, nil
}

// extractPath parses a URL and returns just the path component.
func extractPath(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		// Fallback: strip scheme+host manually.
		idx := strings.Index(rawURL, "://")
		if idx >= 0 {
			rest := rawURL[idx+3:]
			slash := strings.Index(rest, "/")
			if slash >= 0 {
				return rest[slash:]
			}
		}
		return rawURL
	}
	if u.Path == "" {
		return "/"
	}
	return u.Path
}

// findRequestEnd scans forward from startLine to find where the current
// request block ends (before the next ### separator or next HTTP method line).
func findRequestEnd(lines []string, startLine int) int {
	for i := startLine; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		if strings.HasPrefix(trimmed, "###") {
			return i - 1
		}
		if httpMethodPattern.MatchString(trimmed) {
			return i - 1
		}
	}
	return len(lines) - 1
}
