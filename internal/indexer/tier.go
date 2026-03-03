package indexer

import (
	"path/filepath"
	"strings"

	"github.com/cavenine/ctxpp/internal/types"
)

// classifySourceTier determines the ranking tier for all symbols in a file
// based on its repo-relative path. The returned tier controls a multiplicative
// weight applied during semantic search scoring.
//
// Classification rules (evaluated in priority order):
//
//  1. TierLowSignal (4): changelogs, generated code, test fixtures
//  2. TierVendor (3): vendored / third-party dependencies
//  3. TierDocs (2): documentation, configs, tests, examples
//  4. TierCode (1): everything else (project source code)
func classifySourceTier(relPath string) types.SourceTier {
	lower := strings.ToLower(relPath)
	base := strings.ToLower(filepath.Base(relPath))

	// ---- TierLowSignal: changelogs, generated code, test fixtures ----

	// Changelog files (any directory).
	if strings.HasPrefix(base, "changelog") || base == "changes" || base == "history.md" {
		return types.TierLowSignal
	}

	// Generated code patterns.
	if strings.HasPrefix(base, "zz_generated") ||
		strings.Contains(base, "_generated.") ||
		strings.Contains(base, ".generated.") ||
		strings.HasPrefix(base, "generated.pb") ||
		strings.HasSuffix(base, ".pb.go") ||
		strings.HasSuffix(base, "_string.go") {
		return types.TierLowSignal
	}

	// ---- TierVendor: vendored / third-party dependencies ----

	if hasPathComponent(lower, "vendor") ||
		hasPathComponent(lower, "node_modules") {
		return types.TierVendor
	}

	// ---- TierDocs: documentation, tests, examples, configs ----

	if hasPathComponent(lower, "testdata") ||
		hasPathComponent(lower, "test") ||
		hasPathComponent(lower, "tests") ||
		hasPathComponent(lower, "examples") ||
		hasPathComponent(lower, "example") ||
		hasPathComponent(lower, "docs") ||
		hasPathComponent(lower, "doc") ||
		hasPathComponent(lower, "hack") {
		return types.TierDocs
	}

	ext := strings.ToLower(filepath.Ext(relPath))
	if ext == ".md" || ext == ".rst" || ext == ".txt" || ext == ".yaml" || ext == ".yml" || ext == ".toml" {
		return types.TierDocs
	}

	// Go test files.
	if strings.HasSuffix(lower, "_test.go") {
		return types.TierDocs
	}

	// ---- TierCode: project source ----
	return types.TierCode
}

// hasPathComponent reports whether the lowercased path contains the given
// directory name as a complete path component (not a substring of another
// component). For example, hasPathComponent("pkg/vendor/foo.go", "vendor")
// returns true, but hasPathComponent("pkg/revendor/foo.go", "vendor") returns
// false.
func hasPathComponent(lowerPath, component string) bool {
	sep := string(filepath.Separator)
	// Check prefix: "vendor/..."
	if strings.HasPrefix(lowerPath, component+sep) {
		return true
	}
	// Check interior: ".../vendor/..."
	if strings.Contains(lowerPath, sep+component+sep) {
		return true
	}
	// Check exact match (the path is just "vendor").
	if lowerPath == component {
		return true
	}
	return false
}
