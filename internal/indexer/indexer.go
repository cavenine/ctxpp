// Package indexer walks a project tree, parses source files, embeds symbols,
// and persists everything to the store. It supports incremental reindexing
// (skip unchanged files by SHA-256) and live updates via fsnotify.
package indexer

import (
	"context"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/fsnotify/fsnotify"
	gitignore "github.com/sabhiram/go-gitignore"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/parser"
	"github.com/cavenine/ctxpp/internal/store"
	"github.com/cavenine/ctxpp/internal/types"
)

// Config holds indexer options.
type Config struct {
	// ProjectRoot is the absolute path to the project being indexed.
	ProjectRoot string

	// Workers is the number of parallel parse workers (CPU-bound).
	// Defaults to runtime.NumCPU().
	Workers int

	// EmbedConcurrency is the max number of concurrent embed HTTP calls.
	// Defaults to 8.
	EmbedConcurrency int

	// WatchDebounce is how long to wait after the last fs event before
	// reindexing changed files. Defaults to 500ms.
	WatchDebounce time.Duration

	// Logger is the structured logger used by the indexer.
	// Defaults to slog.Default() if nil.
	Logger *slog.Logger
}

func (c *Config) setDefaults() {
	if c.Workers <= 0 {
		c.Workers = runtime.NumCPU()
	}
	if c.EmbedConcurrency <= 0 {
		// Allow override via environment variable, useful for tuning Bedrock
		// concurrency (e.g. CTXPP_EMBED_CONCURRENCY=30).
		if v, err := strconv.Atoi(os.Getenv("CTXPP_EMBED_CONCURRENCY")); err == nil && v > 0 {
			c.EmbedConcurrency = v
		} else {
			c.EmbedConcurrency = 8
		}
	}
	if c.WatchDebounce <= 0 {
		c.WatchDebounce = 500 * time.Millisecond
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// IndexStats reports what happened during an Index() run.
type IndexStats struct {
	FilesIndexed   int
	FilesSkipped   int
	SymbolsIndexed int
	Duration       time.Duration
}

// Indexer orchestrates file walking, parsing, embedding, and storage.
type Indexer struct {
	cfg       Config
	store     *store.Store
	parsers   map[string]parser.Parser // keyed by extension
	filenames map[string]parser.Parser // keyed by exact basename (e.g. "Makefile")
	embedder  embed.Embedder
	log       *slog.Logger
}

// New creates an Indexer.
// parsers is the list of language parsers to use; the indexer routes files by extension.
// Parsers that implement FilenameParser also match by exact basename.
func New(cfg Config, st *store.Store, parsers []parser.Parser, embedder embed.Embedder) *Indexer {
	cfg.setDefaults()
	byExt := make(map[string]parser.Parser)
	byName := make(map[string]parser.Parser)
	for _, p := range parsers {
		for _, ext := range p.Extensions() {
			byExt[ext] = p
		}
		if fp, ok := p.(parser.FilenameParser); ok {
			for _, name := range fp.Filenames() {
				byName[name] = p
			}
		}
	}
	return &Indexer{
		cfg:       cfg,
		store:     st,
		parsers:   byExt,
		filenames: byName,
		embedder:  embedder,
		log:       cfg.Logger.With("component", "indexer"),
	}
}

// ---- Pipeline types --------------------------------------------------------

// walkJob is sent from the walker to parse workers.
type walkJob struct {
	absPath string
	relPath string
	ext     string
}

// parsedFile is sent from parse workers to the store writer.
type parsedFile struct {
	relPath     string
	sha         string
	modTime     int64
	lang        string
	symbols     []types.Symbol
	callEdges   []types.CallEdge
	importEdges []types.ImportEdge
	skipped     bool // true = SHA unchanged, nothing to persist
}

// embedJob is sent from the store writer to embed workers.
type embedJob struct {
	symbolID string
	text     string
}

// embedResult is sent from embed workers to the embed writer.
type embedResult struct {
	symbolID string
	vec      []float32
}

// ---- Index (pipeline) ------------------------------------------------------

// Index walks the project root using a multi-stage channel pipeline:
//
//	Stage 1 (walk):        single goroutine walks the file tree      → walkJobs
//	Stage 2 (parse):       N workers read, hash, parse (CPU-bound)   → parsedFiles
//	Stage 3 (store write): single goroutine writes all DB mutations  → embedJobs
//	Stage 4 (embed):       M workers call Ollama HTTP (I/O-bound)    → embedResults
//	Stage 5 (embed write): single goroutine batch-upserts embeddings
//
// This design keeps SQLite writes contention-free (single writer), maximises
// CPU utilisation on parsing, and overlaps embed HTTP calls with both.
func (idx *Indexer) Index(ctx context.Context) (IndexStats, error) {
	start := time.Now()
	gi := loadGitignore(idx.cfg.ProjectRoot)

	// Skip the entire embed pipeline when the embedder is a non-functional stub.
	// This avoids writing millions of zero-vector bytes to SQLite and removes
	// the dominant bottleneck for first-index latency.
	skipEmbed := idx.embedder.Model() == embed.BundledModel

	walkJobs := make(chan walkJob, idx.cfg.Workers*4)
	parsed := make(chan parsedFile, idx.cfg.Workers*4)
	embedJobs := make(chan embedJob, idx.cfg.EmbedConcurrency*4)
	embedResults := make(chan embedResult, idx.cfg.EmbedConcurrency*4)

	// ---- Stage 1: walk (single goroutine) ----
	go func() {
		defer close(walkJobs)
		_ = filepath.WalkDir(idx.cfg.ProjectRoot, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}

			rel, _ := filepath.Rel(idx.cfg.ProjectRoot, path)

			if strings.HasPrefix(rel, ".ctxpp") {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}

			if d.IsDir() {
				base := d.Name()
				if base != "." && (strings.HasPrefix(base, ".") || base == "node_modules") {
					return filepath.SkipDir
				}
				return nil
			}

			if gi != nil && gi.MatchesPath(rel) {
				return nil
			}

			ext := strings.ToLower(filepath.Ext(path))
			if _, ok := idx.parsers[ext]; !ok {
				// Check filename match (e.g. Makefile, Dockerfile).
				if _, ok := idx.filenames[d.Name()]; !ok {
					return nil
				}
			}

			select {
			case walkJobs <- walkJob{absPath: path, relPath: rel, ext: ext}:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		})
	}()

	// ---- Stage 2: parse workers (CPU-bound, no DB) ----
	var parseWG sync.WaitGroup
	for i := 0; i < idx.cfg.Workers; i++ {
		parseWG.Add(1)
		go func() {
			defer parseWG.Done()
			for j := range walkJobs {
				if ctx.Err() != nil {
					return
				}
				pf, err := idx.parseFile(j.absPath, j.relPath, j.ext)
				if err != nil {
					idx.log.Error("parse file", "path", j.relPath, "err", err)
					continue
				}
				select {
				case parsed <- pf:
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	go func() {
		parseWG.Wait()
		close(parsed)
	}()

	// ---- Stage 3: store writer (single goroutine, all DB writes) ----
	//
	// Batches multiple parsed files into a single transaction to amortise
	// WAL commit overhead. Previously each file incurred ~5 separate commits
	// (UpsertFile, DeleteSymbols, UpsertSymbols, UpsertCallEdges,
	// UpsertImportEdges). Now we flush one transaction per batch.
	const storeBatchSize = 32

	statsCh := make(chan IndexStats, 1)
	go func() {
		var stats IndexStats
		var batch []store.ParsedFileData
		// pendingSyms tracks symbols from the current batch that need
		// to be sent to embed workers after the batch is flushed.
		var pendingSyms []types.Symbol
		// pendingCallEdges and pendingImportEdges collect edges from
		// the current batch so we can build enriched embed text.
		var pendingCallEdges []types.CallEdge
		var pendingImportEdges []types.ImportEdge

		flushBatch := func() {
			if len(batch) == 0 {
				return
			}
			if err := idx.store.UpsertParsedFileBatch(batch); err != nil {
				idx.log.Error("upsert parsed file batch", "files", len(batch), "err", err)
				// On batch failure, none of the files in this batch are counted.
				batch = batch[:0]
				pendingSyms = pendingSyms[:0]
				pendingCallEdges = pendingCallEdges[:0]
				pendingImportEdges = pendingImportEdges[:0]
				return
			}

			// Batch committed successfully — update stats and feed embed jobs.
			for _, pfd := range batch {
				stats.FilesIndexed++
				stats.SymbolsIndexed += len(pfd.Symbols)
				idx.log.Info("indexed", "path", pfd.File.Path, "symbols", len(pfd.Symbols))
			}

			if !skipEmbed {
				// Build enrichment maps from accumulated edges.
				callsBySymbol := buildCallMap(pendingCallEdges)
				importsByFile := buildImportMap(pendingImportEdges)

				for _, sym := range pendingSyms {
					// Key: "file:name" to match CallerFile:CallerSymbol.
					key := sym.File + ":" + sym.Name
					calls := callsBySymbol[key]
					imports := importsByFile[sym.File]
					text := buildEnrichedEmbedText(sym, calls, imports)
					select {
					case embedJobs <- embedJob{symbolID: sym.ID, text: text}:
					case <-ctx.Done():
					}
				}
			}

			batch = batch[:0]
			pendingSyms = pendingSyms[:0]
			pendingCallEdges = pendingCallEdges[:0]
			pendingImportEdges = pendingImportEdges[:0]
		}

		for pf := range parsed {
			if pf.skipped {
				stats.FilesSkipped++
				continue
			}

			batch = append(batch, store.ParsedFileData{
				File: types.FileRecord{
					Path:    pf.relPath,
					SHA256:  pf.sha,
					ModTime: pf.modTime,
					Lang:    pf.lang,
				},
				Symbols:     pf.symbols,
				CallEdges:   pf.callEdges,
				ImportEdges: pf.importEdges,
			})
			pendingSyms = append(pendingSyms, pf.symbols...)
			pendingCallEdges = append(pendingCallEdges, pf.callEdges...)
			pendingImportEdges = append(pendingImportEdges, pf.importEdges...)

			if len(batch) >= storeBatchSize {
				flushBatch()
			}
		}
		flushBatch() // flush remaining

		close(embedJobs) // no more embed work once all files are stored
		stats.Duration = time.Since(start)
		statsCh <- stats
	}()

	// ---- Stage 4: embed batcher (collects jobs → batch HTTP calls) ----
	//
	// Instead of N concurrent single-embed HTTP calls, we collect jobs into
	// batches and call EmbedBatch (if the embedder supports it). This reduces
	// HTTP round-trips from 181k to ~1.8k (at batch size 100) and enables
	// GPU batching inside Ollama.
	const embedBatchSize = 2000

	go func() {
		defer close(embedResults)

		batcher, hasBatch := idx.embedder.(embed.BatchEmbedder)

		if !hasBatch {
			// Fallback: fan out single Embed calls with bounded concurrency.
			var wg sync.WaitGroup
			sem := make(chan struct{}, idx.cfg.EmbedConcurrency)
			for ej := range embedJobs {
				if ctx.Err() != nil {
					break
				}
				wg.Add(1)
				ej := ej
				sem <- struct{}{}
				go func() {
					defer wg.Done()
					defer func() { <-sem }()
					vec, err := idx.embedder.Embed(ctx, ej.text)
					if err != nil {
						idx.log.Warn("embed failed", "symbol", ej.symbolID, "err", err)
						return
					}
					select {
					case embedResults <- embedResult{symbolID: ej.symbolID, vec: vec}:
					case <-ctx.Done():
					}
				}()
			}
			wg.Wait()
			return
		}

		// Batch path: collect jobs into batches and dispatch overlapping
		// EmbedBatch calls. We allow up to embedInflight concurrent Ollama
		// requests so that batch N+1 is already inflight while batch N's
		// results are being pushed to embedResults / written to SQLite.
		// This keeps the GPU busier by hiding the inter-batch gap.
		const embedInflight = 2

		var flushWg sync.WaitGroup
		sem := make(chan struct{}, embedInflight)

		flushBatch := func(jobs []embedJob) {
			if len(jobs) == 0 {
				return
			}
			flushWg.Add(1)
			sem <- struct{}{} // limit concurrency
			go func() {
				defer flushWg.Done()
				defer func() { <-sem }()

				texts := make([]string, len(jobs))
				for i, ej := range jobs {
					texts[i] = ej.text
				}
				vecs, err := batcher.EmbedBatch(ctx, texts)
				if err != nil {
					idx.log.Warn("embed batch error; retrying failed items individually", "size", len(jobs), "err", err)
				}

				failed := make([]embedJob, 0, len(jobs))
				for i, ej := range jobs {
					if i < len(vecs) && vecs[i] != nil {
						select {
						case embedResults <- embedResult{symbolID: ej.symbolID, vec: vecs[i]}:
						case <-ctx.Done():
							return
						}
						continue
					}
					failed = append(failed, ej)
				}

				if len(failed) == 0 {
					return
				}

				// Self-heal path: retry failed batch items individually so a single
				// batch error does not discard successful work for the entire batch.
				idx.log.Warn("embed batch partial failure", "failed", len(failed), "size", len(jobs))
				for _, ej := range failed {
					if ctx.Err() != nil {
						return
					}
					vec, eerr := idx.embedder.Embed(ctx, ej.text)
					if eerr != nil {
						idx.log.Warn("embed retry failed", "symbol", ej.symbolID, "err", eerr)
						continue
					}
					select {
					case embedResults <- embedResult{symbolID: ej.symbolID, vec: vec}:
					case <-ctx.Done():
						return
					}
				}
			}()
		}

		var batch []embedJob
		for ej := range embedJobs {
			if ctx.Err() != nil {
				break
			}
			batch = append(batch, ej)
			if len(batch) >= embedBatchSize {
				// Dispatch this batch and start collecting the next one
				// immediately. The old batch slice is owned by the goroutine.
				flushBatch(batch)
				batch = nil
			}
		}
		flushBatch(batch) // remaining
		flushWg.Wait()
	}()

	// ---- Stage 5: embed writer (single goroutine, batch DB upsert) ----
	embedDone := make(chan struct{})
	go func() {
		defer close(embedDone)
		model := idx.embedder.Model()
		var batch []store.EmbeddingItem
		const batchSize = 64

		flush := func() {
			if len(batch) == 0 {
				return
			}
			if err := idx.store.UpsertEmbeddingsBatch(batch); err != nil {
				idx.log.Warn("batch upsert embeddings", "err", err)
			}
			batch = batch[:0]
		}

		for er := range embedResults {
			batch = append(batch, store.EmbeddingItem{
				SymbolID: er.symbolID,
				Model:    model,
				Vector:   er.vec,
			})
			if len(batch) >= batchSize {
				flush()
			}
		}
		flush() // remaining items
	}()

	// Wait for pipeline to drain.
	stats := <-statsCh
	<-embedDone

	return stats, ctx.Err()
}

// parseFile reads, hashes, and parses a single file. No DB access.
// Returns a parsedFile with skipped=true if the SHA matches the stored record.
func (idx *Indexer) parseFile(absPath, relPath, ext string) (parsedFile, error) {
	src, err := os.ReadFile(absPath)
	if err != nil {
		return parsedFile{}, fmt.Errorf("read %s: %w", relPath, err)
	}

	sha := hashBytes(src)

	// Check SHA against store (read-only, safe from multiple goroutines with WAL mode).
	stored, err := idx.store.GetFileSHA(relPath)
	if err != nil {
		return parsedFile{}, fmt.Errorf("get sha %s: %w", relPath, err)
	}
	if stored == sha {
		idx.log.Debug("skip unchanged", "path", relPath)
		return parsedFile{relPath: relPath, skipped: true}, nil
	}

	p := idx.parsers[ext]
	if p == nil {
		p = idx.filenames[filepath.Base(absPath)]
	}
	result, err := p.Parse(relPath, src)
	if err != nil {
		return parsedFile{}, fmt.Errorf("parse %s: %w", relPath, err)
	}

	// Classify all symbols in this file by provenance tier.
	tier := classifySourceTier(relPath)
	for i := range result.Symbols {
		result.Symbols[i].SourceTier = tier
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return parsedFile{}, fmt.Errorf("stat %s: %w", relPath, err)
	}

	return parsedFile{
		relPath:     relPath,
		sha:         sha,
		modTime:     info.ModTime().UnixNano(),
		lang:        p.Language(),
		symbols:     result.Symbols,
		callEdges:   result.CallEdges,
		importEdges: result.ImportEdges,
	}, nil
}

// indexFile indexes a single file synchronously (used by the Watch path).
// Returns (symbolCount, skipped, error).
func (idx *Indexer) indexFile(ctx context.Context, absPath, relPath, ext string) (int, bool, error) {
	pf, err := idx.parseFile(absPath, relPath, ext)
	if err != nil {
		return 0, false, err
	}
	if pf.skipped {
		return 0, true, nil
	}

	if err := idx.store.UpsertParsedFile(store.ParsedFileData{
		File: types.FileRecord{
			Path:    pf.relPath,
			SHA256:  pf.sha,
			ModTime: pf.modTime,
			Lang:    pf.lang,
		},
		Symbols:     pf.symbols,
		CallEdges:   pf.callEdges,
		ImportEdges: pf.importEdges,
	}); err != nil {
		return 0, false, fmt.Errorf("upsert parsed file %s: %w", relPath, err)
	}

	// Embed symbols concurrently, then batch-upsert.
	idx.embedSymbols(ctx, pf.symbols, pf.callEdges, pf.importEdges)

	idx.log.Info("indexed", "path", relPath, "symbols", len(pf.symbols))
	return len(pf.symbols), false, nil
}

// ---- Watch -----------------------------------------------------------------

// Watch starts a filesystem watcher and reindexes changed files.
// It blocks until ctx is cancelled.
// Goroutine ownership: Watch owns the watcher goroutine; it is cleaned up on return.
func (idx *Indexer) Watch(ctx context.Context) error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("indexer: watch: %w", err)
	}
	defer watcher.Close()

	gi := loadGitignore(idx.cfg.ProjectRoot)

	// Walk and add all directories.
	if err := filepath.WalkDir(idx.cfg.ProjectRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		base := d.Name()
		if base != "." && (strings.HasPrefix(base, ".") || base == "node_modules") {
			return filepath.SkipDir
		}
		return watcher.Add(path)
	}); err != nil {
		return fmt.Errorf("indexer: watch walk: %w", err)
	}

	idx.reconcileStartup(ctx, gi)

	debounce := make(map[string]*time.Timer)
	var mu sync.Mutex

	reindex := func(absPath string) {
		rel, err := filepath.Rel(idx.cfg.ProjectRoot, absPath)
		if err != nil {
			return
		}
		if gi != nil && gi.MatchesPath(rel) {
			return
		}
		ext := strings.ToLower(filepath.Ext(absPath))
		if _, ok := idx.parsers[ext]; !ok {
			if _, ok := idx.filenames[filepath.Base(absPath)]; !ok {
				return
			}
		}
		if _, _, err := idx.indexFile(ctx, absPath, rel, ext); err != nil {
			idx.log.Error("watch reindex", "path", rel, "err", err)
		}
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}
			if event.Has(fsnotify.Remove) || event.Has(fsnotify.Rename) {
				rel, _ := filepath.Rel(idx.cfg.ProjectRoot, event.Name)
				if err := idx.store.DeleteFile(rel); err != nil {
					idx.log.Warn("watch delete", "path", rel, "err", err)
				}
				continue
			}
			if event.Has(fsnotify.Create) {
				info, err := os.Stat(event.Name)
				if err == nil && info.IsDir() {
					idx.addWatchDirsRecursive(ctx, watcher, event.Name, gi)
					continue
				}
				if err != nil && !os.IsNotExist(err) {
					idx.log.Warn("watch stat create path", "path", event.Name, "err", err)
				}
			}
			if !event.Has(fsnotify.Write) && !event.Has(fsnotify.Create) {
				continue
			}
			absPath := event.Name
			mu.Lock()
			if t, ok := debounce[absPath]; ok {
				t.Stop()
			}
			debounce[absPath] = time.AfterFunc(idx.cfg.WatchDebounce, func() {
				reindex(absPath)
				mu.Lock()
				delete(debounce, absPath)
				mu.Unlock()
			})
			mu.Unlock()
		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			idx.log.Warn("watcher error", "err", err)
		}
	}
}

func (idx *Indexer) addWatchDirsRecursive(ctx context.Context, watcher *fsnotify.Watcher, root string, gi *gitignore.GitIgnore) {
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			idx.log.Warn("watch walk new dir", "path", path, "err", err)
			return nil
		}
		if d.IsDir() {
			if idx.shouldSkipWatchDir(path, d.Name(), gi) {
				return filepath.SkipDir
			}
			if err := watcher.Add(path); err != nil {
				if os.IsNotExist(err) {
					return nil
				}
				idx.log.Warn("watch add dir", "path", path, "err", err)
			}
			return nil
		}
		rel, err := filepath.Rel(idx.cfg.ProjectRoot, path)
		if err != nil {
			idx.log.Warn("watch rel file", "path", path, "err", err)
			return nil
		}
		if gi != nil && gi.MatchesPath(rel) {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if _, ok := idx.parsers[ext]; !ok {
			if _, ok := idx.filenames[filepath.Base(path)]; !ok {
				return nil
			}
		}
		if _, _, err := idx.indexFile(ctx, path, rel, ext); err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			idx.log.Warn("watch index existing file", "path", rel, "err", err)
		}
		return nil
	})
}

func (idx *Indexer) shouldSkipWatchDir(path, base string, gi *gitignore.GitIgnore) bool {
	if base != "." && (strings.HasPrefix(base, ".") || base == "node_modules") {
		return true
	}
	rel, err := filepath.Rel(idx.cfg.ProjectRoot, path)
	if err != nil {
		return true
	}
	if rel == "." {
		return false
	}
	if rel == ".ctxpp" || strings.HasPrefix(rel, ".ctxpp/") {
		return true
	}
	if gi != nil && gi.MatchesPath(rel) {
		return true
	}
	return false
}

func (idx *Indexer) reconcileStartup(ctx context.Context, gi *gitignore.GitIgnore) {
	seen := make(map[string]struct{})

	_ = filepath.WalkDir(idx.cfg.ProjectRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			idx.log.Warn("watch reconcile walk", "path", path, "err", err)
			return nil
		}
		if d.IsDir() {
			if idx.shouldSkipWatchDir(path, d.Name(), gi) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(idx.cfg.ProjectRoot, path)
		if err != nil {
			idx.log.Warn("watch reconcile rel", "path", path, "err", err)
			return nil
		}
		if gi != nil && gi.MatchesPath(rel) {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if _, ok := idx.parsers[ext]; !ok {
			if _, ok := idx.filenames[filepath.Base(path)]; !ok {
				return nil
			}
		}
		seen[rel] = struct{}{}
		if _, _, err := idx.indexFile(ctx, path, rel, ext); err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			idx.log.Warn("watch reconcile index", "path", rel, "err", err)
		}
		return nil
	})

	indexedFiles, err := idx.store.ListFiles()
	if err != nil {
		idx.log.Warn("watch reconcile list files", "err", err)
		return
	}
	for _, rel := range indexedFiles {
		if _, ok := seen[rel]; ok {
			continue
		}
		absPath := filepath.Join(idx.cfg.ProjectRoot, rel)
		if _, err := os.Stat(absPath); err == nil {
			continue
		} else if !os.IsNotExist(err) {
			idx.log.Warn("watch reconcile stat indexed file", "path", rel, "err", err)
			continue
		}
		if err := idx.store.DeleteFile(rel); err != nil {
			idx.log.Warn("watch reconcile delete stale", "path", rel, "err", err)
		}
	}
}

// ---- helpers ---------------------------------------------------------------

func hashBytes(b []byte) string {
	return strconv.FormatUint(xxhash.Sum64(b), 16)
}

func loadGitignore(root string) *gitignore.GitIgnore {
	gi, _ := gitignore.CompileIgnoreFile(filepath.Join(root, ".gitignore"))
	return gi
}

// buildEmbedText produces the text sent to the embedder for a symbol.
// It combines the signature and doc comment for richer semantic signal.
func buildEmbedText(sym types.Symbol) string {
	return buildEnrichedEmbedText(sym, nil, nil)
}

// BuildEmbedText is the exported form of buildEmbedText, for use by the
// backfill command and other callers outside this package.
func BuildEmbedText(sym types.Symbol) string {
	return buildEnrichedEmbedText(sym, nil, nil)
}

// maxEnrichCalls is the maximum number of call targets appended to embed text.
const maxEnrichCalls = 10

// maxEnrichImports is the maximum number of import paths appended to embed text.
const maxEnrichImports = 6

// buildEnrichedEmbedText produces the text sent to the embedder, enriched with
// the repo-relative file path, a truncated body snippet, call-target names, and
// import paths extracted during parsing. The extra context helps the embedding
// model associate a symbol with the concepts it references (similar to
// chunk-level context) without needing a separate chunk embedding.
//
// Format:
//
//	<file> <kind> [Receiver.]Name[: Signature]
//	DocComment (if any)
//	<snippet body, truncated>
//	calls: FuncA, FuncB, FuncC (deduped, truncated to maxEnrichCalls)
//	imports: pkg/x, pkg/y (deduped, truncated to maxEnrichImports)
func buildEnrichedEmbedText(sym types.Symbol, calls []string, imports []string) string {
	var b strings.Builder

	// File path token for domain/topic signal.
	if sym.File != "" {
		b.WriteString(sym.File)
		b.WriteByte(' ')
	}

	b.WriteString(string(sym.Kind))
	b.WriteByte(' ')
	if sym.Receiver != "" {
		b.WriteString(sym.Receiver)
		b.WriteByte('.')
	}
	b.WriteString(sym.Name)
	if sym.Signature != "" {
		b.WriteString(": ")
		b.WriteString(sym.Signature)
	}
	if sym.DocComment != "" {
		b.WriteByte('\n')
		b.WriteString(sym.DocComment)
	}

	if sym.Snippet != "" {
		b.WriteByte('\n')
		b.WriteString(sym.Snippet)
	}

	if len(calls) > 0 {
		deduped := dedupStrings(calls)
		if len(deduped) > maxEnrichCalls {
			deduped = deduped[:maxEnrichCalls]
		}
		b.WriteString("\ncalls: ")
		b.WriteString(strings.Join(deduped, ", "))
	}

	if len(imports) > 0 {
		deduped := dedupStrings(imports)
		if len(deduped) > maxEnrichImports {
			deduped = deduped[:maxEnrichImports]
		}
		b.WriteString("\nimports: ")
		b.WriteString(strings.Join(deduped, ", "))
	}

	return b.String()
}

// dedupStrings returns a new slice with duplicates removed, preserving order.
func dedupStrings(ss []string) []string {
	seen := make(map[string]struct{}, len(ss))
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

// buildCallMap groups call-edge callee names by "callerFile:callerSymbol" key.
// This lets the store writer look up call targets for each symbol in O(1).
func buildCallMap(edges []types.CallEdge) map[string][]string {
	m := make(map[string][]string, len(edges)/4+1)
	for _, e := range edges {
		key := e.CallerFile + ":" + e.CallerSymbol
		m[key] = append(m[key], e.CalleeSymbol)
	}
	return m
}

// buildImportMap groups imported paths by importer file.
func buildImportMap(edges []types.ImportEdge) map[string][]string {
	m := make(map[string][]string, len(edges)/4+1)
	for _, e := range edges {
		m[e.ImporterFile] = append(m[e.ImporterFile], e.ImportedPath)
	}
	return m
}

// embedSymbols fans out embed HTTP calls concurrently (bounded by cfg.EmbedConcurrency),
// then batch-upserts all resulting embeddings in one transaction.
// Used by the Watch/single-file path; the Index pipeline uses stages 4+5 instead.
func (idx *Indexer) embedSymbols(ctx context.Context, syms []types.Symbol, callEdges []types.CallEdge, importEdges []types.ImportEdge) {
	if len(syms) == 0 {
		return
	}
	// Skip when using the non-functional stub embedder.
	if idx.embedder.Model() == embed.BundledModel {
		return
	}

	callsBySymbol := buildCallMap(callEdges)
	importsByFile := buildImportMap(importEdges)

	type result struct {
		symbolID string
		vec      []float32
	}

	results := make(chan result, len(syms))
	sem := make(chan struct{}, idx.cfg.EmbedConcurrency)
	var wg sync.WaitGroup

	for _, sym := range syms {
		wg.Add(1)
		go func(s types.Symbol) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				return
			}

			key := s.File + ":" + s.Name
			text := buildEnrichedEmbedText(s, callsBySymbol[key], importsByFile[s.File])
			vec, err := idx.embedder.Embed(ctx, text)
			if err != nil {
				idx.log.Warn("embed failed", "symbol", s.ID, "err", err)
				return
			}
			results <- result{symbolID: s.ID, vec: vec}
		}(sym)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	model := idx.embedder.Model()
	var items []store.EmbeddingItem
	for r := range results {
		items = append(items, store.EmbeddingItem{
			SymbolID: r.symbolID,
			Model:    model,
			Vector:   r.vec,
		})
	}

	if len(items) > 0 {
		if err := idx.store.UpsertEmbeddingsBatch(items); err != nil {
			idx.log.Warn("batch upsert embeddings", "err", err)
		}
	}
}
