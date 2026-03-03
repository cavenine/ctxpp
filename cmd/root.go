package cmd

import (
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "ctxpp",
	Short: "ctx++ — fast, local codebase intelligence for AI agents",
	Long: `ctx++ is a local MCP server that provides AI agents with deep codebase
intelligence via symbol extraction, full-text and vector search, incremental
indexing, and call-graph traversal.`,
}

// Execute is the entry point called from main.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.AddCommand(newMCPCmd())
	rootCmd.AddCommand(newIndexCmd())
	rootCmd.AddCommand(newBackfillCmd())
	rootCmd.AddCommand(newVersionCmd())
}
