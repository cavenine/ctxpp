// Package store provides a SQLite-backed persistence layer for ctx++.
// Schema: files, symbols, embeddings, call_edges, import_edges, FTS5 virtual table.
package store

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/cavenine/ctxpp/internal/types"
	_ "modernc.org/sqlite"
)

const schema = `
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-64000;
PRAGMA mmap_size=268435456;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS files (
	path      TEXT PRIMARY KEY,
	sha256    TEXT NOT NULL,
	mod_time  INTEGER NOT NULL,
	lang      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS symbols (
	id          TEXT PRIMARY KEY,
	file        TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
	name        TEXT NOT NULL,
	kind        TEXT NOT NULL,
	signature   TEXT NOT NULL DEFAULT '',
	doc_comment TEXT NOT NULL DEFAULT '',
	start_line  INTEGER NOT NULL DEFAULT 0,
	end_line    INTEGER NOT NULL DEFAULT 0,
	receiver    TEXT NOT NULL DEFAULT '',
	package     TEXT NOT NULL DEFAULT '',
	source_tier INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file, start_line);

CREATE TABLE IF NOT EXISTS embeddings (
	symbol_id   TEXT PRIMARY KEY REFERENCES symbols(id) ON DELETE CASCADE,
	model       TEXT NOT NULL,
	dims        INTEGER NOT NULL,
	vector      BLOB NOT NULL  -- little-endian float32 array
);

CREATE TABLE IF NOT EXISTS call_edges (
	caller_file   TEXT NOT NULL,
	caller_symbol TEXT NOT NULL,
	callee_file   TEXT NOT NULL DEFAULT '',
	callee_symbol TEXT NOT NULL,
	line          INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (caller_file, caller_symbol, callee_symbol, line)
);

CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(callee_symbol);
CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_symbol);

CREATE TABLE IF NOT EXISTS import_edges (
	importer_file TEXT NOT NULL,
	imported_path TEXT NOT NULL,
	PRIMARY KEY (importer_file, imported_path)
);

CREATE INDEX IF NOT EXISTS idx_import_edges_imported ON import_edges(imported_path);

-- FTS5 for fast keyword search over symbol name, signature, and doc comment.
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
	symbol_id UNINDEXED,
	name,
	signature,
	doc_comment,
	content='symbols',
	content_rowid='rowid'
);

-- Keep FTS in sync with symbols table.
CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
	INSERT INTO symbols_fts(rowid, symbol_id, name, signature, doc_comment)
	VALUES (new.rowid, new.id, new.name, new.signature, new.doc_comment);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
	INSERT INTO symbols_fts(symbols_fts, rowid, symbol_id, name, signature, doc_comment)
	VALUES ('delete', old.rowid, old.id, old.name, old.signature, old.doc_comment);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
	INSERT INTO symbols_fts(symbols_fts, rowid, symbol_id, name, signature, doc_comment)
	VALUES ('delete', old.rowid, old.id, old.name, old.signature, old.doc_comment);
	INSERT INTO symbols_fts(rowid, symbol_id, name, signature, doc_comment)
	VALUES (new.rowid, new.id, new.name, new.signature, new.doc_comment);
END;
`

// semanticScratch holds reusable buffers for SearchSemantic to avoid
// per-query allocations. Obtained from semanticPool.
type semanticScratch struct {
	topK  []semanticHit
	args  []any
	ids   []string
	order map[int64]int
}

type semanticHit struct {
	rowid int64
	score float32
}

type semanticCandidate struct {
	rowid int64
	score float32
}

type semanticSearcher interface {
	Search(queryVec []float32, limit int) ([]semanticCandidate, error)
}

type SemanticMode string

const (
	SemanticModeAuto       SemanticMode = "auto"
	SemanticModeBruteForce SemanticMode = "bruteforce"
	SemanticModeANN        SemanticMode = "ann"
)

type OpenOptions struct {
	SemanticMode SemanticMode
}

type ANNState string

const (
	ANNStateDisabled   ANNState = "disabled"
	ANNStateHealthy    ANNState = "healthy"
	ANNStateRebuilding ANNState = "rebuilding"
	ANNStateMissing    ANNState = "missing"
)

type ANNStatus struct {
	Mode             SemanticMode
	State            ANNState
	ArtifactsPresent bool
}

var semanticPool = sync.Pool{
	New: func() any {
		return &semanticScratch{
			order: make(map[int64]int, 32),
		}
	},
}

var newANNSemanticSearcher = newHNSWSemanticSearcher

func (sc *semanticScratch) reset(limit int) {
	sc.topK = sc.topK[:0]
	sc.args = sc.args[:0]
	sc.ids = sc.ids[:0]
	clear(sc.order)
}

// Store wraps a SQLite database with typed operations.
// It maintains two connection pools: a single-writer pool (db) for mutations
// and a multi-reader pool (rdb) for concurrent read-only queries.
// This avoids contention where N parse workers block on a single connection
// just to check SHA hashes while the store writer holds the connection for
// batch inserts.
type Store struct {
	db                 *sql.DB // write path: MaxOpenConns=1, _txlock=immediate
	rdb                *sql.DB // read path: MaxOpenConns=4, read-only queries
	path               string
	semanticMode       SemanticMode
	semanticMu         sync.RWMutex
	semanticSearcher   semanticSearcher
	annSyncMu          sync.Mutex
	annSyncDepth       int
	annSyncDirty       bool
	annSyncRebuild     bool
	annPendingUpsert   map[string]struct{}
	annPendingDelete   map[int64]struct{}
	annRebuildInFlight bool
	annRebuildDone     chan struct{}
}

// Open opens (or creates) a Store at the given file path.
func Open(path string) (*Store, error) {
	return OpenWithOptions(path, OpenOptions{})
}

// OpenWithOptions opens (or creates) a Store at the given file path with
// optional semantic search configuration.
func OpenWithOptions(path string, opts OpenOptions) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("store: create db dir: %w", err)
	}
	// Write connection: single writer with immediate transaction locking.
	db, err := sql.Open("sqlite", path+"?_busy_timeout=5000&_txlock=immediate")
	if err != nil {
		return nil, fmt.Errorf("store: open write db: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite: single writer

	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: apply schema: %w", err)
	}

	// Migrate: add source_tier column if missing (existing databases).
	db.Exec(`ALTER TABLE symbols ADD COLUMN source_tier INTEGER NOT NULL DEFAULT 1`) //nolint:errcheck

	// Read connection pool: multiple concurrent readers (WAL mode allows this).
	// No _txlock=immediate since we only do SELECT queries on this pool.
	rdb, err := sql.Open("sqlite", path+"?_busy_timeout=5000&mode=ro")
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("store: open read db: %w", err)
	}
	rdb.SetMaxOpenConns(4) // enough for parse workers to not block each other

	mode := opts.SemanticMode
	if mode == "" {
		mode = SemanticModeAuto
	}
	done := make(chan struct{})
	close(done)
	st := &Store{db: db, rdb: rdb, path: path, semanticMode: mode, annRebuildDone: done}
	if err := st.configureSemanticSearcher(); err != nil {
		rdb.Close()
		db.Close()
		return nil, err
	}
	return st, nil
}

func (s *Store) configureSemanticSearcher() error {
	bruteForce := &bruteForceSemanticSearcher{store: s}
	tryANN := func() (semanticSearcher, error) {
		searcher, err := newANNSemanticSearcher(s)
		if err != nil {
			if buildErr := s.BuildANNArtifacts(); buildErr == nil {
				searcher, err = newANNSemanticSearcher(s)
			}
		}
		if err != nil {
			return nil, err
		}
		return searcher, nil
	}
	switch s.semanticMode {
	case SemanticModeANN:
		searcher, err := tryANN()
		if err != nil {
			s.semanticMu.Lock()
			s.semanticSearcher = bruteForce
			s.semanticMu.Unlock()
			s.semanticMode = SemanticModeBruteForce
			return nil
		}
		s.semanticMu.Lock()
		s.semanticSearcher = searcher
		s.semanticMu.Unlock()
		return nil
	case SemanticModeAuto:
		searcher, err := tryANN()
		if err == nil {
			s.semanticMu.Lock()
			s.semanticSearcher = searcher
			s.semanticMu.Unlock()
			s.semanticMode = SemanticModeANN
			return nil
		}
		s.semanticMu.Lock()
		s.semanticSearcher = bruteForce
		s.semanticMu.Unlock()
		s.semanticMode = SemanticModeBruteForce
		return nil
	case SemanticModeBruteForce:
		s.semanticMu.Lock()
		s.semanticSearcher = bruteForce
		s.semanticMu.Unlock()
		s.semanticMode = SemanticModeBruteForce
		return nil
	default:
		return fmt.Errorf("store: unsupported semantic mode %q", s.semanticMode)
	}
}

func (s *Store) currentSemanticSearcher() semanticSearcher {
	s.semanticMu.RLock()
	defer s.semanticMu.RUnlock()
	return s.semanticSearcher
}

func (s *Store) replaceSemanticSearcher(searcher semanticSearcher) {
	s.semanticMu.Lock()
	s.semanticSearcher = searcher
	s.semanticMu.Unlock()
}

func (s *Store) fallbackToBruteForceSearcher() semanticSearcher {
	searcher := &bruteForceSemanticSearcher{store: s}
	s.semanticMu.Lock()
	s.semanticSearcher = searcher
	s.semanticMode = SemanticModeBruteForce
	s.semanticMu.Unlock()
	return searcher
}

func (s *Store) safeSemanticSearch(searcher semanticSearcher, queryVec []float32, limit int) (candidates []semanticCandidate, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("semantic search panic: %v", r)
		}
	}()
	return searcher.Search(queryVec, limit)
}

func (s *Store) waitForBackgroundANNRebuild() error {
	s.annSyncMu.Lock()
	done := s.annRebuildDone
	s.annSyncMu.Unlock()
	<-done
	return nil
}

func (s *Store) ANNStatus() ANNStatus {
	status := ANNStatus{Mode: s.semanticMode}
	paths := annArtifactPaths(s.path)
	status.ArtifactsPresent = annArtifactsExist(paths)
	if s.semanticMode != SemanticModeANN {
		status.State = ANNStateDisabled
		return status
	}

	s.annSyncMu.Lock()
	rebuilding := s.annRebuildInFlight
	s.annSyncMu.Unlock()
	if rebuilding {
		status.State = ANNStateRebuilding
		return status
	}
	if !status.ArtifactsPresent {
		status.State = ANNStateMissing
		return status
	}
	status.State = ANNStateHealthy
	return status
}

// Close closes both the write and read database connections.
func (s *Store) Close() error {
	rerr := s.rdb.Close()
	werr := s.db.Close()
	if werr != nil {
		return werr
	}
	return rerr
}

// DB exposes the raw *sql.DB for advanced use.
func (s *Store) DB() *sql.DB { return s.db }

// BeginDeferredANNSync defers ANN artifact refreshes until EndDeferredANNSync is
// called. Nested calls are supported.
func (s *Store) BeginDeferredANNSync() {
	s.annSyncMu.Lock()
	s.annSyncDepth++
	s.annSyncMu.Unlock()
}

// EndDeferredANNSync ends a deferred ANN sync scope and flushes one pending
// artifact refresh if needed when the outermost scope exits.
func (s *Store) EndDeferredANNSync() error {
	s.annSyncMu.Lock()
	if s.annSyncDepth == 0 {
		s.annSyncMu.Unlock()
		return fmt.Errorf("store: deferred ann sync underflow")
	}
	s.annSyncDepth--
	shouldSync := s.annSyncDepth == 0 && s.annSyncDirty
	rebuild := s.annSyncRebuild
	upserts := make([]string, 0, len(s.annPendingUpsert))
	for symbolID := range s.annPendingUpsert {
		upserts = append(upserts, symbolID)
	}
	deletes := make([]int64, 0, len(s.annPendingDelete))
	for rowID := range s.annPendingDelete {
		deletes = append(deletes, rowID)
	}
	if shouldSync {
		s.annSyncDirty = false
		s.annSyncRebuild = false
		s.annPendingUpsert = nil
		s.annPendingDelete = nil
	}
	s.annSyncMu.Unlock()
	if !shouldSync {
		return nil
	}
	if err := s.flushDeferredANNSync(rebuild, upserts, deletes); err != nil {
		return fmt.Errorf("flush deferred ann sync: %w", err)
	}
	return nil
}

// ---- Files ----------------------------------------------------------------

// GetFileSHA returns the stored SHA256 for a file, or ("", nil) if not indexed.
// Uses the read-only connection pool for concurrent access from parse workers.
func (s *Store) GetFileSHA(path string) (string, error) {
	var sha string
	err := s.rdb.QueryRow(`SELECT sha256 FROM files WHERE path=?`, path).Scan(&sha)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return sha, err
}

// UpsertFile inserts or updates a file record.
func (s *Store) UpsertFile(f types.FileRecord) error {
	_, err := s.db.Exec(`
		INSERT INTO files(path,sha256,mod_time,lang) VALUES(?,?,?,?)
		ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mod_time=excluded.mod_time, lang=excluded.lang`,
		f.Path, f.SHA256, f.ModTime, f.Lang)
	return err
}

// DeleteFile removes a file and all its symbols (via CASCADE).
func (s *Store) DeleteFile(path string) error {
	_, err := s.db.Exec(`DELETE FROM files WHERE path=?`, path)
	if err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return err
	}
	return nil
}

// ListFiles returns all indexed file paths.
func (s *Store) ListFiles() ([]string, error) {
	rows, err := s.rdb.Query(`SELECT path FROM files`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var paths []string
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return paths, nil
}

// ---- Symbols ---------------------------------------------------------------

// DeleteSymbolsByFile removes all symbols for a file before re-inserting.
func (s *Store) DeleteSymbolsByFile(filePath string) error {
	_, err := s.db.Exec(`DELETE FROM symbols WHERE file=?`, filePath)
	if err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return err
	}
	return nil
}

// UpsertSymbol inserts or replaces a symbol.
func (s *Store) UpsertSymbol(sym types.Symbol) error {
	_, err := s.db.Exec(`
		INSERT INTO symbols(id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET
			file=excluded.file, name=excluded.name, kind=excluded.kind,
			signature=excluded.signature, doc_comment=excluded.doc_comment,
			start_line=excluded.start_line, end_line=excluded.end_line,
			receiver=excluded.receiver, package=excluded.package,
			source_tier=excluded.source_tier`,
		sym.ID, sym.File, sym.Name, string(sym.Kind), sym.Signature,
		sym.DocComment, sym.StartLine, sym.EndLine, sym.Receiver, sym.Package,
		sym.SourceTier)
	return err
}

// GetSymbol returns a single symbol by ID.
func (s *Store) GetSymbol(id string) (*types.Symbol, error) {
	row := s.rdb.QueryRow(`
		SELECT id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier
		FROM symbols WHERE id=?`, id)
	return scanSymbol(row)
}

// GetSymbolsByFile returns all symbols for a file, ordered by start_line.
func (s *Store) GetSymbolsByFile(filePath string) ([]types.Symbol, error) {
	rows, err := s.rdb.Query(`
		SELECT id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier
		FROM symbols WHERE file=? ORDER BY start_line`, filePath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanSymbols(rows)
}

// SearchKeyword performs FTS5 keyword search.
func (s *Store) SearchKeyword(query string, limit int) ([]types.Symbol, error) {
	if limit <= 0 {
		limit = 10
	}
	if query == "" {
		return nil, nil
	}
	// FTS5 content tables must be joined on rowid, not on UNINDEXED columns.
	rows, err := s.rdb.Query(`
		SELECT s.id,s.file,s.name,s.kind,s.signature,s.doc_comment,s.start_line,s.end_line,s.receiver,s.package,s.source_tier
		FROM symbols_fts f
		JOIN symbols s ON s.rowid = f.rowid
		WHERE symbols_fts MATCH ?
		ORDER BY rank
		LIMIT ?`, query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanSymbols(rows)
}

// ---- Embeddings ------------------------------------------------------------

// UpsertEmbedding stores a float32 embedding vector for a symbol.
func (s *Store) UpsertEmbedding(symbolID, model string, vec []float32) error {
	blob := encodeFloat32s(vec)
	_, err := s.db.Exec(`
		INSERT INTO embeddings(symbol_id,model,dims,vector) VALUES(?,?,?,?)
		ON CONFLICT(symbol_id) DO UPDATE SET model=excluded.model, dims=excluded.dims, vector=excluded.vector`,
		symbolID, model, len(vec), blob)
	if err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return err
	}
	return nil
}

// EmbeddingItem groups the fields needed for a single embedding upsert.
type EmbeddingItem struct {
	SymbolID string
	Model    string
	Vector   []float32
}

// UpsertEmbeddingsBatch inserts or replaces multiple embeddings in a single transaction.
func (s *Store) UpsertEmbeddingsBatch(items []EmbeddingItem) error {
	if len(items) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck
	stmt, err := tx.Prepare(`
		INSERT INTO embeddings(symbol_id,model,dims,vector) VALUES(?,?,?,?)
		ON CONFLICT(symbol_id) DO UPDATE SET model=excluded.model, dims=excluded.dims, vector=excluded.vector`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, item := range items {
		blob := encodeFloat32s(item.Vector)
		if _, err := stmt.Exec(item.SymbolID, item.Model, len(item.Vector), blob); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	symbolIDs := make([]string, 0, len(items))
	for _, item := range items {
		symbolIDs = append(symbolIDs, item.SymbolID)
	}
	if err := s.syncANNEmbeddingUpsertsIfPresent(symbolIDs); err != nil {
		return err
	}
	return nil
}

// deduplicateByNameKind removes duplicate symbols that share the same Name and
// Kind, keeping only the first (highest-ranked) occurrence of each pair. This
// prevents identical symbols from multiple files (e.g. the same SQL table
// defined across several migration files) from consuming multiple result slots.
func deduplicateByNameKind(syms []types.Symbol) []types.Symbol {
	seen := make(map[string]struct{}, len(syms))
	out := syms[:0:0] // reuse backing array header; avoids allocation when no dups
	for _, s := range syms {
		key := s.Name + "\x00" + string(s.Kind)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, s)
	}
	return out
}

type bruteForceSemanticSearcher struct {
	store *Store
}

type SemanticDebugCandidate struct {
	RowID int64   `json:"rowid"`
	Score float32 `json:"score"`
}

type SemanticDebugResult struct {
	ConfiguredMode  SemanticMode             `json:"configured_mode"`
	EffectiveMode   SemanticMode             `json:"effective_mode"`
	CandidateLimit  int                      `json:"candidate_limit"`
	FallbackApplied bool                     `json:"fallback_applied"`
	Candidates      []SemanticDebugCandidate `json:"candidates"`
	Results         []types.Symbol           `json:"results"`
}

// SearchSemantic performs cosine similarity search using the configured
// semantic searcher, then applies the existing exact ranking pipeline:
// source-tier weighting, symbol hydration, and deduplication. The default
// searcher is a brute-force scan, but this orchestration layer is the seam for
// future ANN candidate generation.
func (s *Store) SearchSemantic(queryVec []float32, limit int) ([]types.Symbol, error) {
	debug, err := s.DebugSemanticQuery(queryVec, limit)
	if err != nil {
		return nil, err
	}
	return debug.Results, nil
}

func (s *Store) DebugSemanticQuery(queryVec []float32, limit int) (SemanticDebugResult, error) {
	return s.debugSemanticQuery(queryVec, limit, 0)
}

func (s *Store) DebugSemanticQueryWithCandidateLimit(queryVec []float32, limit int, candidateLimit int) (SemanticDebugResult, error) {
	return s.debugSemanticQuery(queryVec, limit, candidateLimit)
}

func (s *Store) debugSemanticQuery(queryVec []float32, limit int, candidateLimit int) (SemanticDebugResult, error) {
	if limit <= 0 {
		limit = 10
	}
	configuredMode := s.semanticMode
	if candidateLimit <= 0 {
		candidateLimit = s.semanticCandidateLimit(limit)
	}
	searcher := s.currentSemanticSearcher()
	candidates, err := s.safeSemanticSearch(searcher, queryVec, candidateLimit)
	fallbackApplied := false
	if err != nil && s.semanticMode == SemanticModeANN {
		fallbackApplied = true
		searcher = s.fallbackToBruteForceSearcher()
		if candidateLimit <= 0 {
			candidateLimit = s.semanticCandidateLimit(limit)
		}
		candidates, err = s.safeSemanticSearch(searcher, queryVec, candidateLimit)
	}
	if err != nil {
		return SemanticDebugResult{}, err
	}
	results, err := s.hydrateSemanticCandidates(queryVec, candidates, limit)
	if err != nil {
		return SemanticDebugResult{}, err
	}
	debugCandidates := make([]SemanticDebugCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		debugCandidates = append(debugCandidates, SemanticDebugCandidate{RowID: candidate.rowid, Score: candidate.score})
	}
	return SemanticDebugResult{
		ConfiguredMode:  configuredMode,
		EffectiveMode:   s.semanticMode,
		CandidateLimit:  candidateLimit,
		FallbackApplied: fallbackApplied,
		Candidates:      debugCandidates,
		Results:         results,
	}, nil
}

func (s *Store) semanticCandidateLimit(limit int) int {
	if limit <= 0 {
		limit = 10
	}
	if s.semanticMode == SemanticModeANN {
		candidateLimit := limit * 400
		if candidateLimit < 4096 {
			candidateLimit = 4096
		}
		return candidateLimit
	}
	return limit * 3
}

// Search performs brute-force cosine similarity search by comparing query
// vectors against all stored embeddings. It returns candidate row IDs and raw
// cosine scores only; Store.SearchSemantic handles the higher-level ranking and
// hydration steps.
//
// Two-phase approach to minimise allocations:
//
//	Phase 1: scan only (rowid, vector blob) using RawBytes and integer rowid
//	         to avoid per-row string and blob-copy allocations. Compute cosine
//	         similarity and track top-K rowids.
//	Phase 2: caller applies source-tier weighting, symbol hydration, and final
//	         limit trimming.
//
// All scratch slices and maps are obtained from a sync.Pool (semanticPool) to
// eliminate per-query allocations on the hot path.
func (b *bruteForceSemanticSearcher) Search(queryVec []float32, limit int) ([]semanticCandidate, error) {
	if limit <= 0 {
		limit = 10
	}

	sc := semanticPool.Get().(*semanticScratch)
	defer semanticPool.Put(sc)
	sc.reset(limit)

	// Phase 1: scan embeddings by rowid (int64, no string alloc) and vector
	// (RawBytes, avoids bytes.Clone). We only keep the top candidates.
	rows, err := b.store.rdb.Query(`SELECT rowid, vector FROM embeddings`)
	if err != nil {
		return nil, err
	}

	minIdx := 0

	for rows.Next() {
		var rid int64
		var blob sql.RawBytes
		if err := rows.Scan(&rid, &blob); err != nil {
			rows.Close()
			return nil, err
		}

		score := cosineSimilarityBlob(queryVec, []byte(blob))

		if len(sc.topK) < limit {
			sc.topK = append(sc.topK, semanticHit{rid, score})
			if score < sc.topK[minIdx].score {
				minIdx = len(sc.topK) - 1
			}
		} else if score > sc.topK[minIdx].score {
			sc.topK[minIdx] = semanticHit{rid, score}
			minIdx = 0
			for i := 1; i < len(sc.topK); i++ {
				if sc.topK[i].score < sc.topK[minIdx].score {
					minIdx = i
				}
			}
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if len(sc.topK) == 0 {
		return nil, nil
	}

	out := make([]semanticCandidate, len(sc.topK))
	for i, hit := range sc.topK {
		out[i] = semanticCandidate{rowid: hit.rowid, score: hit.score}
	}
	return out, nil
}

func (s *Store) hydrateSemanticCandidates(queryVec []float32, candidates []semanticCandidate, limit int) ([]types.Symbol, error) {
	if limit <= 0 {
		limit = 10
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	sc := semanticPool.Get().(*semanticScratch)
	defer semanticPool.Put(sc)
	sc.reset(len(candidates))
	for _, candidate := range candidates {
		sc.topK = append(sc.topK, semanticHit{rowid: candidate.rowid, score: candidate.score})
	}

	// Phase 2: look up symbol_id and source_tier for each candidate rowid.
	placeholders := strings.Repeat("?,", len(sc.topK))
	placeholders = placeholders[:len(placeholders)-1]

	// Reuse pooled slices for args and the rowid->order map.
	for i, h := range sc.topK {
		sc.args = append(sc.args, h.rowid)
		sc.order[h.rowid] = i
	}

	// Pre-size ids slice to match topK length.
	for range sc.topK {
		sc.ids = append(sc.ids, "")
	}

	// Fetch symbol_id, source_tier, and vector in one join so candidate scores can
	// be re-ranked using exact cosine similarity even when the underlying searcher
	// is approximate.
	idRows, err := s.rdb.Query(`
		SELECT e.rowid, e.symbol_id, s.source_tier, e.vector
		FROM embeddings e
		JOIN symbols s ON s.id = e.symbol_id
		WHERE e.rowid IN (`+placeholders+`)`, sc.args...)
	if err != nil {
		return nil, err
	}
	defer idRows.Close()
	for idRows.Next() {
		var rid int64
		var sid string
		var tier types.SourceTier
		var blob sql.RawBytes
		if err := idRows.Scan(&rid, &sid, &tier, &blob); err != nil {
			return nil, err
		}
		if idx, ok := sc.order[rid]; ok {
			sc.ids[idx] = sid
			sc.topK[idx].score = cosineSimilarityBlob(queryVec, []byte(blob)) * tier.TierWeight()
		}
	}
	if err := idRows.Err(); err != nil {
		return nil, err
	}

	// Re-sort by tier-adjusted score (insertion sort, optimal for small N).
	for i := 1; i < len(sc.topK); i++ {
		key := sc.topK[i]
		j := i - 1
		for j >= 0 && sc.topK[j].score < key.score {
			sc.topK[j+1] = sc.topK[j]
			j--
		}
		sc.topK[j+1] = key
	}

	// Truncate to final limit.
	if len(sc.topK) > limit {
		sc.topK = sc.topK[:limit]
	}

	// Copy ids out of the pooled slice in adjusted-score order.
	ids := make([]string, len(sc.topK))
	for i, h := range sc.topK {
		if idx, ok := sc.order[h.rowid]; ok {
			ids[i] = sc.ids[idx]
		}
	}

	symMap, err := s.getSymbolsByIDMap(ids)
	if err != nil {
		return nil, err
	}

	syms := make([]types.Symbol, 0, len(ids))
	for _, id := range ids {
		if sym, ok := symMap[id]; ok {
			syms = append(syms, sym)
		}
	}
	return deduplicateByNameKind(syms), nil
}

// getSymbolsByIDMap returns a map of symbol ID -> Symbol for the given IDs.
func (s *Store) getSymbolsByIDMap(ids []string) (map[string]types.Symbol, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	placeholders := strings.Repeat("?,", len(ids))
	placeholders = placeholders[:len(placeholders)-1]
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	rows, err := s.rdb.Query(`
		SELECT id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier
		FROM symbols WHERE id IN (`+placeholders+`)`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	m := make(map[string]types.Symbol, len(ids))
	for rows.Next() {
		var sym types.Symbol
		var kindStr string
		if err := rows.Scan(&sym.ID, &sym.File, &sym.Name, &kindStr,
			&sym.Signature, &sym.DocComment, &sym.StartLine, &sym.EndLine,
			&sym.Receiver, &sym.Package, &sym.SourceTier); err != nil {
			return nil, err
		}
		sym.Kind = types.SymbolKind(kindStr)
		m[sym.ID] = sym
	}
	return m, rows.Err()
}

// ---- Call Edges ------------------------------------------------------------

// UpsertCallEdges replaces all call edges for a given caller file.
func (s *Store) UpsertCallEdges(filePath string, edges []types.CallEdge) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck
	if _, err := tx.Exec(`DELETE FROM call_edges WHERE caller_file=?`, filePath); err != nil {
		return err
	}
	for _, e := range edges {
		if _, err := tx.Exec(`
			INSERT OR IGNORE INTO call_edges(caller_file,caller_symbol,callee_file,callee_symbol,line)
			VALUES(?,?,?,?,?)`,
			e.CallerFile, e.CallerSymbol, e.CalleeFile, e.CalleeSymbol, e.Line); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return fmt.Errorf("sync ann artifacts: %w", err)
	}
	return nil
}

// Callees returns symbols called by the given symbol.
func (s *Store) CalleeSymbols(callerSymbol string) ([]string, error) {
	rows, err := s.rdb.Query(`SELECT DISTINCT callee_symbol FROM call_edges WHERE caller_symbol=?`, callerSymbol)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanStrings(rows)
}

// CallerSymbols returns symbols that call the given symbol.
func (s *Store) CallerSymbols(calleeSymbol string) ([]string, error) {
	rows, err := s.rdb.Query(`SELECT DISTINCT caller_symbol FROM call_edges WHERE callee_symbol=?`, calleeSymbol)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanStrings(rows)
}

// ---- Import Edges ----------------------------------------------------------

// UpsertImportEdges replaces all import edges for a given file.
func (s *Store) UpsertImportEdges(filePath string, edges []types.ImportEdge) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck
	if _, err := tx.Exec(`DELETE FROM import_edges WHERE importer_file=?`, filePath); err != nil {
		return err
	}
	for _, e := range edges {
		if _, err := tx.Exec(`
			INSERT OR IGNORE INTO import_edges(importer_file,imported_path) VALUES(?,?)`,
			e.ImporterFile, e.ImportedPath); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return fmt.Errorf("sync ann artifacts: %w", err)
	}
	return nil
}

// ---- Scanning helpers ------------------------------------------------------

func scanSymbol(row *sql.Row) (*types.Symbol, error) {
	var sym types.Symbol
	var kindStr string
	err := row.Scan(&sym.ID, &sym.File, &sym.Name, &kindStr,
		&sym.Signature, &sym.DocComment, &sym.StartLine, &sym.EndLine,
		&sym.Receiver, &sym.Package, &sym.SourceTier)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	sym.Kind = types.SymbolKind(kindStr)
	return &sym, nil
}

func scanSymbols(rows *sql.Rows) ([]types.Symbol, error) {
	var syms []types.Symbol
	for rows.Next() {
		var sym types.Symbol
		var kindStr string
		if err := rows.Scan(&sym.ID, &sym.File, &sym.Name, &kindStr,
			&sym.Signature, &sym.DocComment, &sym.StartLine, &sym.EndLine,
			&sym.Receiver, &sym.Package, &sym.SourceTier); err != nil {
			return nil, err
		}
		sym.Kind = types.SymbolKind(kindStr)
		syms = append(syms, sym)
	}
	return syms, rows.Err()
}

func scanStrings(rows *sql.Rows) ([]string, error) {
	var out []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// ---- Vector helpers --------------------------------------------------------

func encodeFloat32s(vec []float32) []byte {
	buf := make([]byte, 4*len(vec))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func decodeFloat32s(buf []byte) []float32 {
	n := len(buf) / 4
	vec := make([]float32, n)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return vec
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}

// cosineSimilarityBlob computes cosine similarity between a float32 slice and
// a little-endian float32 blob without allocating a decoded vector.
func cosineSimilarityBlob(a []float32, blob []byte) float32 {
	n := len(blob) / 4
	if n != len(a) || n == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:])))
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}

// scoredSymbol pairs a symbol with a similarity score for ranking.
type scoredSymbol struct {
	sym   types.Symbol
	score float32
}

// sortScored sorts descending by score using a simple insertion sort.
// For typical result sets (<=limit, e.g. 10) this is optimal.
func sortScored(results []scoredSymbol) {
	n := len(results)
	for i := 1; i < n; i++ {
		key := results[i]
		j := i - 1
		for j >= 0 && results[j].score < key.score {
			results[j+1] = results[j]
			j--
		}
		results[j+1] = key
	}
}

// ---- Batch helpers ---------------------------------------------------------

// ParsedFileData bundles all the data for a single parsed file that needs to
// be persisted atomically. Used by UpsertParsedFile to execute all per-file
// mutations in a single transaction, reducing WAL commits from ~5 to 1.
type ParsedFileData struct {
	File        types.FileRecord
	Symbols     []types.Symbol
	CallEdges   []types.CallEdge
	ImportEdges []types.ImportEdge
}

// UpsertParsedFile persists all data for a single parsed file in one transaction.
// This replaces the previous pattern of calling UpsertFile, DeleteSymbolsByFile,
// UpsertSymbolsBatch, UpsertCallEdges, and UpsertImportEdges separately, which
// incurred ~5 WAL commits per file. Now it is 1 commit per file.
func (s *Store) UpsertParsedFile(pf ParsedFileData) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	// 1. Upsert file record.
	if _, err := tx.Exec(`
		INSERT INTO files(path,sha256,mod_time,lang) VALUES(?,?,?,?)
		ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mod_time=excluded.mod_time, lang=excluded.lang`,
		pf.File.Path, pf.File.SHA256, pf.File.ModTime, pf.File.Lang); err != nil {
		return fmt.Errorf("upsert file: %w", err)
	}

	// 2. Delete stale embeddings and symbols for this file.
	if _, err := tx.Exec(`DELETE FROM embeddings WHERE symbol_id IN (SELECT id FROM symbols WHERE file=?)`, pf.File.Path); err != nil {
		return fmt.Errorf("delete embeddings: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM symbols WHERE file=?`, pf.File.Path); err != nil {
		return fmt.Errorf("delete symbols: %w", err)
	}

	// 3. Insert new symbols.
	if len(pf.Symbols) > 0 {
		symStmt, err := tx.Prepare(`
			INSERT INTO symbols(id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier)
			VALUES(?,?,?,?,?,?,?,?,?,?,?)
			ON CONFLICT(id) DO UPDATE SET
				file=excluded.file, name=excluded.name, kind=excluded.kind,
				signature=excluded.signature, doc_comment=excluded.doc_comment,
				start_line=excluded.start_line, end_line=excluded.end_line,
				receiver=excluded.receiver, package=excluded.package,
				source_tier=excluded.source_tier`)
		if err != nil {
			return fmt.Errorf("prepare symbol stmt: %w", err)
		}
		defer symStmt.Close()
		for _, sym := range pf.Symbols {
			if _, err := symStmt.Exec(sym.ID, sym.File, sym.Name, string(sym.Kind), sym.Signature,
				sym.DocComment, sym.StartLine, sym.EndLine, sym.Receiver, sym.Package,
				sym.SourceTier); err != nil {
				return fmt.Errorf("insert symbol %s: %w", sym.ID, err)
			}
		}
	}

	// 4. Replace call edges for this file.
	if _, err := tx.Exec(`DELETE FROM call_edges WHERE caller_file=?`, pf.File.Path); err != nil {
		return fmt.Errorf("delete call edges: %w", err)
	}
	for _, e := range pf.CallEdges {
		if _, err := tx.Exec(`
			INSERT OR IGNORE INTO call_edges(caller_file,caller_symbol,callee_file,callee_symbol,line)
			VALUES(?,?,?,?,?)`,
			e.CallerFile, e.CallerSymbol, e.CalleeFile, e.CalleeSymbol, e.Line); err != nil {
			return fmt.Errorf("insert call edge: %w", err)
		}
	}

	// 5. Replace import edges for this file.
	if _, err := tx.Exec(`DELETE FROM import_edges WHERE importer_file=?`, pf.File.Path); err != nil {
		return fmt.Errorf("delete import edges: %w", err)
	}
	for _, e := range pf.ImportEdges {
		if _, err := tx.Exec(`
			INSERT OR IGNORE INTO import_edges(importer_file,imported_path) VALUES(?,?)`,
			e.ImporterFile, e.ImportedPath); err != nil {
			return fmt.Errorf("insert import edge: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return fmt.Errorf("sync ann artifacts: %w", err)
	}
	return nil
}

// UpsertParsedFileBatch persists multiple parsed files in a single transaction.
// This amortises WAL commit overhead across N files, reducing total commits
// from N (one per file) to 1 for the entire batch.
func (s *Store) UpsertParsedFileBatch(files []ParsedFileData) error {
	if len(files) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	symStmt, err := tx.Prepare(`
		INSERT INTO symbols(id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET
			file=excluded.file, name=excluded.name, kind=excluded.kind,
			signature=excluded.signature, doc_comment=excluded.doc_comment,
			start_line=excluded.start_line, end_line=excluded.end_line,
			receiver=excluded.receiver, package=excluded.package,
			source_tier=excluded.source_tier`)
	if err != nil {
		return fmt.Errorf("prepare symbol stmt: %w", err)
	}
	defer symStmt.Close()

	for _, pf := range files {
		// 1. Upsert file record.
		if _, err := tx.Exec(`
			INSERT INTO files(path,sha256,mod_time,lang) VALUES(?,?,?,?)
			ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mod_time=excluded.mod_time, lang=excluded.lang`,
			pf.File.Path, pf.File.SHA256, pf.File.ModTime, pf.File.Lang); err != nil {
			return fmt.Errorf("upsert file %s: %w", pf.File.Path, err)
		}

		// 2. Delete stale embeddings and symbols.
		if _, err := tx.Exec(`DELETE FROM embeddings WHERE symbol_id IN (SELECT id FROM symbols WHERE file=?)`, pf.File.Path); err != nil {
			return fmt.Errorf("delete embeddings %s: %w", pf.File.Path, err)
		}
		if _, err := tx.Exec(`DELETE FROM symbols WHERE file=?`, pf.File.Path); err != nil {
			return fmt.Errorf("delete symbols %s: %w", pf.File.Path, err)
		}

		// 3. Insert symbols.
		for _, sym := range pf.Symbols {
			if _, err := symStmt.Exec(sym.ID, sym.File, sym.Name, string(sym.Kind), sym.Signature,
				sym.DocComment, sym.StartLine, sym.EndLine, sym.Receiver, sym.Package,
				sym.SourceTier); err != nil {
				return fmt.Errorf("insert symbol %s: %w", sym.ID, err)
			}
		}

		// 4. Replace call edges.
		if _, err := tx.Exec(`DELETE FROM call_edges WHERE caller_file=?`, pf.File.Path); err != nil {
			return fmt.Errorf("delete call edges %s: %w", pf.File.Path, err)
		}
		for _, e := range pf.CallEdges {
			if _, err := tx.Exec(`
				INSERT OR IGNORE INTO call_edges(caller_file,caller_symbol,callee_file,callee_symbol,line)
				VALUES(?,?,?,?,?)`,
				e.CallerFile, e.CallerSymbol, e.CalleeFile, e.CalleeSymbol, e.Line); err != nil {
				return fmt.Errorf("insert call edge: %w", err)
			}
		}

		// 5. Replace import edges.
		if _, err := tx.Exec(`DELETE FROM import_edges WHERE importer_file=?`, pf.File.Path); err != nil {
			return fmt.Errorf("delete import edges %s: %w", pf.File.Path, err)
		}
		for _, e := range pf.ImportEdges {
			if _, err := tx.Exec(`
				INSERT OR IGNORE INTO import_edges(importer_file,imported_path) VALUES(?,?)`,
				e.ImporterFile, e.ImportedPath); err != nil {
				return fmt.Errorf("insert import edge: %w", err)
			}
		}
	}

	if err := tx.Commit(); err != nil {
		return err
	}
	if err := s.syncANNArtifactsIfPresent(); err != nil {
		return fmt.Errorf("sync ann artifacts: %w", err)
	}
	return nil
}

// UpsertSymbolsBatch inserts/replaces multiple symbols in a single transaction.
func (s *Store) UpsertSymbolsBatch(syms []types.Symbol) error {
	if len(syms) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck
	stmt, err := tx.Prepare(`
		INSERT INTO symbols(id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET
			file=excluded.file, name=excluded.name, kind=excluded.kind,
			signature=excluded.signature, doc_comment=excluded.doc_comment,
			start_line=excluded.start_line, end_line=excluded.end_line,
			receiver=excluded.receiver, package=excluded.package,
			source_tier=excluded.source_tier`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, sym := range syms {
		if _, err := stmt.Exec(sym.ID, sym.File, sym.Name, string(sym.Kind), sym.Signature,
			sym.DocComment, sym.StartLine, sym.EndLine, sym.Receiver, sym.Package,
			sym.SourceTier); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// ---- Hybrid Search (RRF) ---------------------------------------------------

// RRF hyperparameters for hybrid search.
const (
	// rrfK is the RRF smoothing constant. Higher values reduce the influence
	// of high-rank positions. 60 is the standard value from the original paper.
	rrfK = 60

	// rrfWeightSemantic is the weight applied to semantic (vector) rank scores.
	rrfWeightSemantic = 0.6

	// rrfWeightFTS is the weight applied to FTS keyword rank scores.
	rrfWeightFTS = 0.4

	// rrfOverselect is the multiplier for how many results to fetch from each
	// sub-search relative to the requested limit.
	rrfOverselect = 3
)

// SearchHybrid combines FTS keyword results with semantic vector results using
// Reciprocal Rank Fusion (RRF). Each sub-search returns an over-selected set of
// candidates; for each candidate appearing in either list, an RRF score is
// computed as:
//
//	score = w_sem / (k + rank_sem) + w_fts / (k + rank_fts)
//
// where rank_x is the 1-based rank in that list (or a large penalty if absent).
// The final results are sorted by descending RRF score and truncated to limit.
func (s *Store) SearchHybrid(queryVec []float32, ftsQuery string, limit int) ([]types.Symbol, error) {
	if limit <= 0 {
		limit = 10
	}
	fetchN := limit * rrfOverselect

	// Preprocess the FTS query: strip stopwords, deduplicate, and extract
	// discriminative keywords so BM25 ranks on meaningful terms only.
	cleanedFTS := ExtractKeywords(ftsQuery)

	type searchResult struct {
		syms []types.Symbol
		err  error
	}

	kwCh := make(chan searchResult, 1)
	semCh := make(chan searchResult, 1)

	go func() {
		syms, err := s.SearchKeyword(cleanedFTS, fetchN)
		kwCh <- searchResult{syms, err}
	}()
	go func() {
		syms, err := s.SearchSemantic(queryVec, fetchN)
		semCh <- searchResult{syms, err}
	}()

	kwR := <-kwCh
	semR := <-semCh

	// If both fail, return the semantic error (more likely to be informative).
	if kwR.err != nil && semR.err != nil {
		return nil, semR.err
	}

	// Build rank maps (symbol ID → 1-based rank).
	semRank := make(map[string]int, len(semR.syms))
	for i, sym := range semR.syms {
		semRank[sym.ID] = i + 1
	}
	ftsRank := make(map[string]int, len(kwR.syms))
	for i, sym := range kwR.syms {
		ftsRank[sym.ID] = i + 1
	}

	// Collect all unique candidates and their Symbol values.
	candidates := make(map[string]types.Symbol)
	for _, sym := range semR.syms {
		candidates[sym.ID] = sym
	}
	for _, sym := range kwR.syms {
		if _, ok := candidates[sym.ID]; !ok {
			candidates[sym.ID] = sym
		}
	}

	// Penalty rank for symbols not present in a given list.
	penaltyRank := fetchN + 100

	// Compute RRF scores.
	type scored struct {
		sym   types.Symbol
		score float64
	}
	results := make([]scored, 0, len(candidates))
	for id, sym := range candidates {
		sr := penaltyRank
		if r, ok := semRank[id]; ok {
			sr = r
		}
		fr := penaltyRank
		if r, ok := ftsRank[id]; ok {
			fr = r
		}
		score := rrfWeightSemantic/float64(rrfK+sr) + rrfWeightFTS/float64(rrfK+fr)
		results = append(results, scored{sym: sym, score: score})
	}

	// Sort by descending RRF score.
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	out := make([]types.Symbol, len(results))
	for i, r := range results {
		out[i] = r.sym
	}
	return deduplicateByNameKind(out), nil
}

// AllSymbolIDs returns all symbol IDs that do not yet have embeddings.
func (s *Store) SymbolIDsWithoutEmbeddings() ([]string, error) {
	rows, err := s.rdb.Query(`
		SELECT id FROM symbols WHERE id NOT IN (SELECT symbol_id FROM embeddings)`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanStrings(rows)
}

// GetSymbolsByIDs returns symbols for a set of IDs.
func (s *Store) GetSymbolsByIDs(ids []string) ([]types.Symbol, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	placeholders := strings.Repeat("?,", len(ids))
	placeholders = placeholders[:len(placeholders)-1]
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	rows, err := s.rdb.Query(`
		SELECT id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier
		FROM symbols WHERE id IN (`+placeholders+`)`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanSymbols(rows)
}

// GetSymbolsByNames returns symbols matching any of the given short names.
func (s *Store) GetSymbolsByNames(names []string) ([]types.Symbol, error) {
	if len(names) == 0 {
		return nil, nil
	}
	placeholders := strings.Repeat("?,", len(names))
	placeholders = placeholders[:len(placeholders)-1]
	args := make([]any, len(names))
	for i, n := range names {
		args[i] = n
	}
	rows, err := s.rdb.Query(`
		SELECT id,file,name,kind,signature,doc_comment,start_line,end_line,receiver,package,source_tier
		FROM symbols WHERE name IN (`+placeholders+`)`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanSymbols(rows)
}

// ---- Call-Graph Re-ranking -------------------------------------------------

// callGraphBoostPerConnection is the number of positions a symbol is promoted
// for each call-graph connection it has to other members of the result set.
// For example, if a symbol has 2 connections, it is promoted 2*2=4 positions.
const callGraphBoostPerConnection = 2

// RerankByCallGraph reorders symbols by boosting those that are densely connected
// (as callers or callees) to other symbols in the result set. This promotes
// symbols that form a cohesive call cluster, improving result quality when the
// query targets a specific feature or workflow.
//
// The reranking is position-based: symbols with call-graph connections to other
// members of the set are promoted forward by callGraphBoostPerConnection
// positions per connection. This is a stable, bounded adjustment.
func (s *Store) RerankByCallGraph(syms []types.Symbol) ([]types.Symbol, error) {
	if len(syms) <= 1 {
		return syms, nil
	}

	// Collect short names of all symbols in the result set.
	nameSet := make(map[string]bool, len(syms))
	for _, sym := range syms {
		nameSet[sym.Name] = true
	}

	// Query call_edges for all connections where both caller and callee are in
	// the result set. We use a single query with two IN clauses, which is
	// efficient because the set is small (typically 10-30 symbols).
	names := make([]string, 0, len(nameSet))
	for n := range nameSet {
		names = append(names, n)
	}
	placeholders := strings.Repeat("?,", len(names))
	placeholders = placeholders[:len(placeholders)-1]

	// Build args: first set for caller_symbol IN, second for callee_symbol IN.
	args := make([]any, 0, len(names)*2)
	for _, n := range names {
		args = append(args, n)
	}
	for _, n := range names {
		args = append(args, n)
	}

	rows, err := s.rdb.Query(`
		SELECT DISTINCT caller_symbol, callee_symbol
		FROM call_edges
		WHERE caller_symbol IN (`+placeholders+`)
		  AND callee_symbol IN (`+placeholders+`)`,
		args...)
	if err != nil {
		return syms, nil // graceful degradation: return original order
	}
	defer rows.Close()

	// Count connections per symbol name (both as caller and callee).
	connectionCount := make(map[string]int, len(names))
	for rows.Next() {
		var caller, callee string
		if err := rows.Scan(&caller, &callee); err != nil {
			return syms, nil
		}
		connectionCount[caller]++
		connectionCount[callee]++
	}
	if err := rows.Err(); err != nil {
		return syms, nil
	}

	// If no connections found, return original order.
	hasConnections := false
	for _, c := range connectionCount {
		if c > 0 {
			hasConnections = true
			break
		}
	}
	if !hasConnections {
		return syms, nil
	}

	// Build scored list: original position minus boost (lower = better).
	type entry struct {
		sym      types.Symbol
		priority float64 // lower = earlier in final result
	}
	entries := make([]entry, len(syms))
	for i, sym := range syms {
		boost := float64(connectionCount[sym.Name]) * callGraphBoostPerConnection
		entries[i] = entry{
			sym:      sym,
			priority: float64(i) - boost,
		}
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].priority < entries[j].priority
	})

	result := make([]types.Symbol, len(entries))
	for i, e := range entries {
		result[i] = e.sym
	}
	return result, nil
}
