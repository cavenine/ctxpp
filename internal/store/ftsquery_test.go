package store

import "testing"

func TestExtractKeywords(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "strips stopwords",
			input: "etcd storage and watch mechanism",
			want:  "etcd storage watch mechanism",
		},
		{
			name:  "preserves discriminative terms only",
			input: "how is the authentication and login flow handled",
			want:  "authentication login flow handled",
		},
		{
			name:  "deduplicates tokens",
			input: "pod scheduling pod affinity pod",
			want:  "pod scheduling affinity",
		},
		{
			name:  "filters short tokens",
			input: "a CRI go runtime",
			want:  "cri runtime",
		},
		{
			name:  "filters long tokens",
			input: "abcdefghijklmnopqrstuvwxyz01234567890 etcd",
			want:  "etcd",
		},
		{
			name:  "limits to max terms",
			input: "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike",
			want:  "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
		},
		{
			name:  "empty input",
			input: "",
			want:  "",
		},
		{
			name:  "all stopwords returns empty",
			input: "the and is of for in to a",
			want:  "",
		},
		{
			name:  "handles compound tokens with hyphens",
			input: "pod-affinity and anti-affinity rules",
			want:  "pod-affinity anti-affinity rules",
		},
		{
			name:  "handles dotted paths",
			input: "import net.http and fmt",
			want:  "import net.http fmt",
		},
		{
			name:  "case insensitive",
			input: "RBAC Authorization RoleBinding",
			want:  "rbac authorization rolebinding",
		},
		{
			name:  "mixed case and stopwords",
			input: "What is the API server admission controller webhook",
			want:  "api server admission controller webhook",
		},
		{
			name:  "real benchmark query Q9",
			input: "etcd storage and watch mechanism",
			want:  "etcd storage watch mechanism",
		},
		{
			name:  "real benchmark query Q7",
			input: "API server admission controller webhook",
			want:  "api server admission controller webhook",
		},
		{
			name:  "agent filler stripped",
			input: "help me find the code for file handling",
			want:  "find handling",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractKeywords(tc.input)
			if got != tc.want {
				t.Errorf("ExtractKeywords(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
