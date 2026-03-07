package cmd

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/store"
)

func TestParseStoreOpenOptions_DefaultsToAuto(t *testing.T) {
	t.Setenv("CTXPP_VECTOR_INDEX", "")
	opts, err := parseStoreOpenOptionsFromEnv()
	if err != nil {
		t.Fatalf("parseStoreOpenOptionsFromEnv() error = %v", err)
	}
	if opts.SemanticMode != store.SemanticModeAuto {
		t.Fatalf("SemanticMode = %q, want %q", opts.SemanticMode, store.SemanticModeAuto)
	}
}

func TestParseStoreOpenOptions_ParsesANNMode(t *testing.T) {
	t.Setenv("CTXPP_VECTOR_INDEX", "ann")
	opts, err := parseStoreOpenOptionsFromEnv()
	if err != nil {
		t.Fatalf("parseStoreOpenOptionsFromEnv() error = %v", err)
	}
	if opts.SemanticMode != store.SemanticModeANN {
		t.Fatalf("SemanticMode = %q, want %q", opts.SemanticMode, store.SemanticModeANN)
	}
}

func TestParseStoreOpenOptions_RejectsInvalidMode(t *testing.T) {
	t.Setenv("CTXPP_VECTOR_INDEX", "bogus")
	_, err := parseStoreOpenOptionsFromEnv()
	if err == nil {
		t.Fatal("parseStoreOpenOptionsFromEnv() error = nil, want error")
	}
}
