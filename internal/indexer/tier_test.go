package indexer

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestClassifySourceTier(t *testing.T) {
	tests := []struct {
		name string
		path string
		want types.SourceTier
	}{
		// ---- TierCode (default project source) ----
		{name: "normal go file", path: "cmd/server/main.go", want: types.TierCode},
		{name: "internal package", path: "internal/store/store.go", want: types.TierCode},
		{name: "root go file", path: "main.go", want: types.TierCode},
		{name: "nested source", path: "pkg/api/handler.go", want: types.TierCode},

		// ---- TierDocs (docs, tests, configs) ----
		{name: "markdown file", path: "README.md", want: types.TierDocs},
		{name: "yaml config", path: "config.yaml", want: types.TierDocs},
		{name: "yml config", path: "deploy.yml", want: types.TierDocs},
		{name: "toml config", path: "Cargo.toml", want: types.TierDocs},
		{name: "txt file", path: "NOTICE.txt", want: types.TierDocs},
		{name: "rst file", path: "docs/guide.rst", want: types.TierDocs},
		{name: "testdata dir", path: "internal/parser/testdata/sample.go", want: types.TierDocs},
		{name: "test dir", path: "test/integration/run.go", want: types.TierDocs},
		{name: "tests dir", path: "tests/e2e/suite.go", want: types.TierDocs},
		{name: "examples dir", path: "examples/basic/main.go", want: types.TierDocs},
		{name: "example dir", path: "example/main.go", want: types.TierDocs},
		{name: "docs dir", path: "docs/api/types.go", want: types.TierDocs},
		{name: "doc dir", path: "doc/design.go", want: types.TierDocs},
		{name: "hack dir", path: "hack/update-codegen.go", want: types.TierDocs},
		{name: "go test file", path: "pkg/api/handler_test.go", want: types.TierDocs},
		{name: "nested md", path: "internal/parser/README.md", want: types.TierDocs},

		// ---- TierVendor (vendored / third-party) ----
		{name: "vendor dir", path: "vendor/github.com/foo/bar.go", want: types.TierVendor},
		{name: "nested vendor", path: "pkg/vendor/lib/lib.go", want: types.TierVendor},
		{name: "third_party is code", path: "third_party/proto/types.go", want: types.TierCode},
		{name: "node_modules", path: "node_modules/express/index.js", want: types.TierVendor},
		{name: "staging src is code", path: "staging/src/k8s.io/api/core/v1/types.go", want: types.TierCode},

		// ---- TierLowSignal (changelogs, generated code) ----
		{name: "changelog", path: "CHANGELOG.md", want: types.TierLowSignal},
		{name: "changelog lowercase", path: "changelog.md", want: types.TierLowSignal},
		{name: "changelog no ext", path: "CHANGELOG", want: types.TierLowSignal},
		{name: "nested changelog", path: "pkg/CHANGELOG.md", want: types.TierLowSignal},
		{name: "changes file", path: "CHANGES", want: types.TierLowSignal},
		{name: "history md", path: "HISTORY.md", want: types.TierLowSignal},
		{name: "zz_generated", path: "pkg/apis/zz_generated.deepcopy.go", want: types.TierLowSignal},
		{name: "underscore generated", path: "types_generated.go", want: types.TierLowSignal},
		{name: "dot generated", path: "schema.generated.go", want: types.TierLowSignal},
		{name: "protobuf generated", path: "api/generated.pb.go", want: types.TierLowSignal},
		{name: "pb.go file", path: "proto/message.pb.go", want: types.TierLowSignal},
		{name: "string generated", path: "pkg/kind_string.go", want: types.TierLowSignal},

		// ---- Edge cases: priority order matters ----
		{name: "generated in vendor (low signal wins)", path: "vendor/foo/zz_generated.go", want: types.TierLowSignal},
		{name: "changelog in vendor (low signal wins)", path: "vendor/foo/CHANGELOG.md", want: types.TierLowSignal},
		{name: "test file not in test dir", path: "pkg/api/handler_test.go", want: types.TierDocs},
		{name: "vendor not as component", path: "pkg/revendor/lib.go", want: types.TierCode},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := classifySourceTier(tc.path)
			if got != tc.want {
				t.Errorf("classifySourceTier(%q) = %d, want %d", tc.path, got, tc.want)
			}
		})
	}
}

func TestHasPathComponent(t *testing.T) {
	tests := []struct {
		name      string
		path      string
		component string
		want      bool
	}{
		{name: "prefix match", path: "vendor/foo/bar.go", component: "vendor", want: true},
		{name: "interior match", path: "pkg/vendor/lib.go", component: "vendor", want: true},
		{name: "exact match", path: "vendor", component: "vendor", want: true},
		{name: "no match substring", path: "pkg/revendor/lib.go", component: "vendor", want: false},
		{name: "no match at all", path: "cmd/server/main.go", component: "vendor", want: false},
		{name: "suffix only no sep", path: "myvendor", component: "vendor", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := hasPathComponent(tc.path, tc.component)
			if got != tc.want {
				t.Errorf("hasPathComponent(%q, %q) = %v, want %v", tc.path, tc.component, got, tc.want)
			}
		})
	}
}
