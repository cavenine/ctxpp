package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/cavenine/ctxpp/internal/store"
)

func parseStoreOpenOptionsFromEnv() (store.OpenOptions, error) {
	raw := strings.TrimSpace(os.Getenv("CTXPP_VECTOR_INDEX"))
	if raw == "" {
		return store.OpenOptions{SemanticMode: store.SemanticModeAuto}, nil
	}
	switch strings.ToLower(raw) {
	case string(store.SemanticModeAuto):
		return store.OpenOptions{SemanticMode: store.SemanticModeAuto}, nil
	case string(store.SemanticModeBruteForce):
		return store.OpenOptions{SemanticMode: store.SemanticModeBruteForce}, nil
	case string(store.SemanticModeANN):
		return store.OpenOptions{SemanticMode: store.SemanticModeANN}, nil
	default:
		return store.OpenOptions{}, fmt.Errorf("invalid CTXPP_VECTOR_INDEX %q", raw)
	}
}
