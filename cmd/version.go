package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// Version is set at build time via:
//
//	go build -ldflags "-X github.com/cavenine/ctxpp/cmd.Version=v0.0.3"
var Version = "dev"

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the ctx++ version",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("ctx++ %s\n", Version)
		},
	}
}
