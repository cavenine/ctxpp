# Adding Language Support

This document is a template and checklist for adding a new language to ctx++.

Use it as a copy/paste starting point for issue planning and implementation PRs.

## Goals

- Add parser support for a new language/file type.
- Produce high-signal symbols, call edges, and import edges.
- Integrate with indexing, incremental updates, and watch mode.
- Keep behavior aligned with existing quality and testing standards.

## Principles

- Tree-sitter first: if a grammar exists, use it before regex/line parsing.
- Keep parser interfaces small and consistent with `internal/parser`.
- Do not regress existing languages.
- Prefer incremental, test-first vertical slices.

---

## Implementation Template

Replace `<LANG>` and placeholders with your target language.

### 1) Scope and file mapping

- Language: `<LANG>`
- File extensions: `<.ext1, .ext2>`
- Optional exact filenames: `<Makefile-like names if any>`
- Initial symbol kinds in scope:
  - `<functions>`
  - `<classes/interfaces/structs>`
  - `<methods/properties/fields>`
  - `<modules/packages/namespaces>`
- Out of scope for v1 parser slice:
  - `<advanced language feature(s)>`

### 2) Parser implementation

- Add parser under `internal/parser/`:
  - `<lang>_parser.go`
  - optional helpers for AST traversal
- Implement existing parser interfaces used by indexer:
  - `Language()`
  - `Extensions()`
  - `Parse(...)`
  - `Filenames()` if filename-based parsing is needed
- Parse output should populate:
  - `[]types.Symbol`
  - `[]types.CallEdge`
  - `[]types.ImportEdge`

### 3) Symbol extraction rules

Document concrete extraction rules before coding:

- Symbol ID format: `<file:name:kind>` (match project conventions)
- Signature style: `<how signatures are rendered>`
- Doc comment mapping: `<how docs/comments are captured>`
- Receiver/container handling: `<methods, namespace, package semantics>`
- Line ranges: `<how start/end lines are computed>`

### 4) Indexer integration

- Wire parser into parser registration location(s) used by commands.
- Confirm extension and/or filename routing resolves to new parser.
- Verify unchanged-file SHA skip behavior still works.
- Verify watch mode behavior:
  - create/modify/delete for target language files
  - create directory + new files inside directory
  - startup reconcile catches offline changes

### 5) Tests (required)

Use table-driven tests and `t.Run`.

- Parser tests in `internal/parser/`:
  - happy path symbol extraction
  - edge cases (empty file, malformed snippets, nested declarations)
  - calls/imports extraction where applicable
- Indexer integration tests in `internal/indexer/`:
  - file is indexed for new extension(s)
  - incremental reindex skips unchanged content
  - watch-mode create/modify/delete coverage
- Fixtures:
  - add small, readable fixtures under `testdata/`

### 6) Performance and correctness checks

- Run:
  - `go test ./...`
  - `go test -race ./...` (if practical for your change)
  - `go vet ./...`
- Spot-check symbol counts and call/import graph quality on a realistic sample repo.

### 7) Documentation updates

- Update `README.md` language support list.
- Update architecture/design docs if parser architecture changed.
- Add notes for known limitations in first release.

---

## PR Checklist Template

Copy this into the PR description:

```md
## Summary
- Add `<LANG>` parser and wire extension routing for `<.exts>`.
- Extract `<key symbol kinds>` plus call/import edges.
- Add parser/indexer/watch tests and fixtures.

## Design Notes
- Grammar strategy: `<tree-sitter/custom fallback>`
- Known limitations: `<list>`

## Validation
- [ ] go test ./...
- [ ] go vet ./...
- [ ] Watch-mode scenarios tested

## Follow-ups (optional)
- `<future quality/perf improvements>`
```

## Definition of Done

- New language files are indexed automatically.
- Keyword/semantic search can return symbols for the new language.
- Call graph and blast-radius tools return meaningful language-specific relationships.
- Incremental indexing and watcher flows work without manual reindex.
- Tests cover parser behavior and indexer integration.
