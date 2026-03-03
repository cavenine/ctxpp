package widget

import (
	"fmt"
	"io"
)

// Widget holds configuration for a UI element.
type Widget struct {
	Name  string
	Width int
}

// Renderer draws things to a writer.
type Renderer interface {
	Render(w io.Writer) error
}

// New creates a Widget with the given name.
func New(name string) *Widget {
	return &Widget{Name: name}
}

// Render writes the widget to w.
func (wg *Widget) Render(w io.Writer) error {
	_, err := fmt.Fprintf(w, "<widget>%s</widget>", wg.Name)
	return err
}

// unexported is intentionally not exported.
func unexported() {}

const Version = "1.0"

var DefaultWidth = 100
