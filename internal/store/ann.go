package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/coder/hnsw"
)

const (
	annFormatVersion = 1
	annEngineHNSW    = "hnsw"
)

type annPaths struct {
	Index    string
	Metadata string
}

type annMetadata struct {
	FormatVersion int    `json:"format_version"`
	Engine        string `json:"engine"`
	Model         string `json:"model"`
	Dims          int    `json:"dims"`
	Count         int    `json:"count"`
}

type hnswSemanticSearcher struct {
	paths    annPaths
	metadata annMetadata
	graph    *hnsw.SavedGraph[int64]
}

func annArtifactPaths(dbPath string) annPaths {
	dir := filepath.Dir(dbPath)
	return annPaths{
		Index:    filepath.Join(dir, "ann-hnsw.bin"),
		Metadata: filepath.Join(dir, "ann-hnsw.json"),
	}
}

func writeANNMetadata(path string, meta annMetadata) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal ann metadata: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write ann metadata: %w", err)
	}
	return nil
}

func readANNMetadata(path string) (annMetadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return annMetadata{}, fmt.Errorf("read ann metadata: %w", err)
	}
	var meta annMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return annMetadata{}, fmt.Errorf("decode ann metadata: %w", err)
	}
	return meta, nil
}

func annArtifactsExist(paths annPaths) bool {
	if _, err := os.Stat(paths.Index); err != nil {
		return false
	}
	if _, err := os.Stat(paths.Metadata); err != nil {
		return false
	}
	return true
}

func (s *Store) syncANNArtifactsIfPresent() error {
	s.annSyncMu.Lock()
	if s.annSyncDepth > 0 {
		s.annSyncDirty = true
		s.annSyncRebuild = true
		s.annSyncMu.Unlock()
		return nil
	}
	s.annSyncMu.Unlock()
	return s.forceSyncANNArtifactsIfPresent()
}

func (s *Store) forceSyncANNArtifactsIfPresent() error {

	paths := annArtifactPaths(s.path)
	if !annArtifactsExist(paths) {
		return nil
	}
	meta, err := s.currentANNMetadata()
	if err != nil {
		return fmt.Errorf("sync ann artifacts: %w", err)
	}
	if meta.Count == 0 {
		if err := removeANNArtifacts(paths); err != nil {
			return fmt.Errorf("sync ann artifacts: %w", err)
		}
		return nil
	}
	if err := s.BuildANNArtifacts(); err != nil {
		return fmt.Errorf("sync ann artifacts: %w", err)
	}
	return nil
}

func (s *Store) syncANNEmbeddingUpsertsIfPresent(symbolIDs []string) error {
	s.annSyncMu.Lock()
	if s.annSyncDepth > 0 {
		s.annSyncDirty = true
		if s.annPendingUpsert == nil {
			s.annPendingUpsert = make(map[string]struct{}, len(symbolIDs))
		}
		for _, symbolID := range symbolIDs {
			s.annPendingUpsert[symbolID] = struct{}{}
		}
		s.annSyncMu.Unlock()
		return nil
	}
	s.annSyncMu.Unlock()

	if len(symbolIDs) == 0 {
		return nil
	}
	paths := annArtifactPaths(s.path)
	if !annArtifactsExist(paths) {
		return nil
	}
	current, err := s.currentANNMetadata()
	if err != nil {
		return fmt.Errorf("sync ann embedding upserts: %w", err)
	}
	if current.Count == 0 {
		if err := removeANNArtifacts(paths); err != nil {
			return fmt.Errorf("sync ann embedding upserts: %w", err)
		}
		return nil
	}
	stored, err := readANNMetadata(paths.Metadata)
	if err != nil {
		return s.BuildANNArtifacts()
	}
	if stored.FormatVersion != annFormatVersion || stored.Engine != annEngineHNSW || stored.Model != current.Model || stored.Dims != current.Dims || current.Count < stored.Count {
		return s.BuildANNArtifacts()
	}
	graph, err := hnsw.LoadSavedGraph[int64](paths.Index)
	if err != nil {
		return s.BuildANNArtifacts()
	}
	nodes, err := s.annNodesBySymbolIDs(symbolIDs)
	if err != nil {
		return fmt.Errorf("sync ann embedding upserts: %w", err)
	}
	if len(nodes) == 0 {
		return nil
	}
	if err := addHNSWNodes(graph, nodes); err != nil {
		return s.BuildANNArtifacts()
	}
	if err := graph.Save(); err != nil {
		return fmt.Errorf("sync ann embedding upserts: save graph: %w", err)
	}
	if err := writeANNMetadata(paths.Metadata, current); err != nil {
		return fmt.Errorf("sync ann embedding upserts: write metadata: %w", err)
	}
	return nil
}

func (s *Store) syncANNEmbeddingDeletesIfPresent(rowIDs []int64) error {
	s.annSyncMu.Lock()
	if s.annSyncDepth > 0 {
		s.annSyncDirty = true
		if s.annPendingDelete == nil {
			s.annPendingDelete = make(map[int64]struct{}, len(rowIDs))
		}
		for _, rowID := range rowIDs {
			s.annPendingDelete[rowID] = struct{}{}
		}
		s.annSyncMu.Unlock()
		return nil
	}
	s.annSyncMu.Unlock()

	if len(rowIDs) == 0 {
		return nil
	}
	paths := annArtifactPaths(s.path)
	if !annArtifactsExist(paths) {
		return nil
	}
	current, err := s.currentANNMetadata()
	if err != nil {
		return fmt.Errorf("sync ann embedding deletes: %w", err)
	}
	if current.Count == 0 {
		if err := removeANNArtifacts(paths); err != nil {
			return fmt.Errorf("sync ann embedding deletes: %w", err)
		}
		return nil
	}
	stored, err := readANNMetadata(paths.Metadata)
	if err != nil {
		return s.BuildANNArtifacts()
	}
	if stored.FormatVersion != annFormatVersion || stored.Engine != annEngineHNSW || stored.Model != current.Model || stored.Dims != current.Dims || current.Count > stored.Count {
		return s.BuildANNArtifacts()
	}
	graph, err := hnsw.LoadSavedGraph[int64](paths.Index)
	if err != nil {
		return s.BuildANNArtifacts()
	}
	for _, rowID := range rowIDs {
		graph.Delete(rowID)
	}
	if err := graph.Save(); err != nil {
		return fmt.Errorf("sync ann embedding deletes: save graph: %w", err)
	}
	if err := writeANNMetadata(paths.Metadata, current); err != nil {
		return fmt.Errorf("sync ann embedding deletes: write metadata: %w", err)
	}
	return nil
}

func (s *Store) flushDeferredANNSync(rebuild bool, upserts []string, deletes []int64) error {
	if rebuild {
		return s.forceSyncANNArtifactsIfPresent()
	}
	if len(deletes) > 0 {
		if err := s.syncANNEmbeddingDeletesIfPresent(deletes); err != nil {
			return err
		}
	}
	if len(upserts) > 0 {
		if err := s.syncANNEmbeddingUpsertsIfPresent(upserts); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) embeddingRowIDsByFile(filePath string) ([]int64, error) {
	rows, err := s.db.Query(`
		SELECT e.rowid
		FROM embeddings e
		JOIN symbols s ON s.id = e.symbol_id
		WHERE s.file=?
	`, filePath)
	if err != nil {
		return nil, fmt.Errorf("query embedding rowids by file: %w", err)
	}
	defer rows.Close()
	rowIDs := make([]int64, 0)
	for rows.Next() {
		var rowID int64
		if err := rows.Scan(&rowID); err != nil {
			return nil, fmt.Errorf("scan embedding rowid by file: %w", err)
		}
		rowIDs = append(rowIDs, rowID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate embedding rowids by file: %w", err)
	}
	return rowIDs, nil
}

func (s *Store) annNodesBySymbolIDs(symbolIDs []string) ([]hnsw.Node[int64], error) {
	placeholders := strings.Repeat("?,", len(symbolIDs))
	placeholders = placeholders[:len(placeholders)-1]
	args := make([]any, 0, len(symbolIDs))
	for _, id := range symbolIDs {
		args = append(args, id)
	}
	rows, err := s.db.Query(`SELECT rowid, vector FROM embeddings WHERE symbol_id IN (`+placeholders+`)`, args...)
	if err != nil {
		return nil, fmt.Errorf("query embedding nodes: %w", err)
	}
	defer rows.Close()
	nodes := make([]hnsw.Node[int64], 0, len(symbolIDs))
	for rows.Next() {
		var rowid int64
		var blob []byte
		if err := rows.Scan(&rowid, &blob); err != nil {
			return nil, fmt.Errorf("scan embedding node: %w", err)
		}
		nodes = append(nodes, hnsw.MakeNode(rowid, decodeFloat32s(blob)))
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate embedding nodes: %w", err)
	}
	return nodes, nil
}

func addHNSWNodes(graph *hnsw.SavedGraph[int64], nodes []hnsw.Node[int64]) error {
	const chunkSize = 256
	for start := 0; start < len(nodes); start += chunkSize {
		end := start + chunkSize
		if end > len(nodes) {
			end = len(nodes)
		}
		if err := addHNSWNodeChunk(graph, nodes[start:end]); err != nil {
			return err
		}
	}
	return nil
}

func addHNSWNodeChunk(graph *hnsw.SavedGraph[int64], nodes []hnsw.Node[int64]) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("add hnsw node: %v", r)
		}
	}()
	graph.Add(nodes...)
	return nil
}

func removeANNArtifacts(paths annPaths) error {
	for _, path := range []string{paths.Index, paths.Metadata} {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove %s: %w", path, err)
		}
	}
	return nil
}

func newHNSWSemanticSearcher(s *Store) (semanticSearcher, error) {
	paths := annArtifactPaths(s.path)
	meta, err := readANNMetadata(paths.Metadata)
	if err != nil {
		return nil, fmt.Errorf("load hnsw metadata: %w", err)
	}
	if meta.FormatVersion != annFormatVersion {
		return nil, fmt.Errorf("load hnsw metadata: unsupported format version %d", meta.FormatVersion)
	}
	if meta.Engine != annEngineHNSW {
		return nil, fmt.Errorf("load hnsw metadata: unsupported engine %q", meta.Engine)
	}
	if meta.Dims <= 0 {
		return nil, fmt.Errorf("load hnsw metadata: invalid dims %d", meta.Dims)
	}
	if err := s.validateANNMetadata(meta); err != nil {
		return nil, fmt.Errorf("load hnsw metadata: %w", err)
	}
	if _, err := os.Stat(paths.Index); err != nil {
		return nil, fmt.Errorf("stat hnsw index: %w", err)
	}
	graph, err := hnsw.LoadSavedGraph[int64](paths.Index)
	if err != nil {
		return nil, fmt.Errorf("load hnsw graph: %w", err)
	}
	return &hnswSemanticSearcher{paths: paths, metadata: meta, graph: graph}, nil
}

func (s *Store) validateANNMetadata(meta annMetadata) error {
	current, err := s.currentANNMetadata()
	if err != nil {
		return fmt.Errorf("read embedding state: %w", err)
	}
	if current.Count == 0 {
		return fmt.Errorf("no embeddings available")
	}
	if meta.Model != current.Model {
		return fmt.Errorf("model mismatch: have %q want %q", meta.Model, current.Model)
	}
	if meta.Dims != current.Dims {
		return fmt.Errorf("dims mismatch: have %d want %d", meta.Dims, current.Dims)
	}
	if meta.Count != current.Count {
		return fmt.Errorf("count mismatch: have %d want %d", meta.Count, current.Count)
	}
	return nil
}

func (s *Store) currentANNMetadata() (annMetadata, error) {
	rows, err := s.db.Query(`SELECT model, dims FROM embeddings ORDER BY rowid`)
	if err != nil {
		return annMetadata{}, fmt.Errorf("query embeddings: %w", err)
	}
	defer rows.Close()

	meta := annMetadata{FormatVersion: annFormatVersion, Engine: annEngineHNSW}
	for rows.Next() {
		var model string
		var dims int
		if err := rows.Scan(&model, &dims); err != nil {
			return annMetadata{}, fmt.Errorf("scan embedding metadata: %w", err)
		}
		if meta.Model == "" {
			meta.Model = model
			meta.Dims = dims
		} else {
			if meta.Model != model {
				return annMetadata{}, fmt.Errorf("mixed embedding models %q and %q", meta.Model, model)
			}
			if meta.Dims != dims {
				return annMetadata{}, fmt.Errorf("mixed embedding dims %d and %d", meta.Dims, dims)
			}
		}
		meta.Count++
	}
	if err := rows.Err(); err != nil {
		return annMetadata{}, fmt.Errorf("iterate embeddings: %w", err)
	}
	return meta, nil
}

func (s *Store) BuildANNArtifacts() error {
	meta, err := s.currentANNMetadata()
	if err != nil {
		return fmt.Errorf("build ann artifacts: %w", err)
	}
	if meta.Count == 0 {
		return fmt.Errorf("build ann artifacts: no embeddings available")
	}
	rows, err := s.db.Query(`SELECT rowid, vector FROM embeddings ORDER BY rowid`)
	if err != nil {
		return fmt.Errorf("build ann artifacts: query embeddings: %w", err)
	}
	defer rows.Close()

	graph := hnsw.NewGraph[int64]()
	for rows.Next() {
		var rowid int64
		var blob []byte
		if err := rows.Scan(&rowid, &blob); err != nil {
			return fmt.Errorf("build ann artifacts: scan embedding: %w", err)
		}
		graph.Add(hnsw.MakeNode(rowid, decodeFloat32s(blob)))
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("build ann artifacts: iterate embeddings: %w", err)
	}

	paths := annArtifactPaths(s.path)
	if err := os.MkdirAll(filepath.Dir(paths.Index), 0o755); err != nil {
		return fmt.Errorf("build ann artifacts: create artifact dir: %w", err)
	}
	f, err := os.Create(paths.Index)
	if err != nil {
		return fmt.Errorf("build ann artifacts: create index file: %w", err)
	}
	if err := graph.Export(f); err != nil {
		_ = f.Close()
		return fmt.Errorf("build ann artifacts: export graph: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("build ann artifacts: close index file: %w", err)
	}
	if err := writeANNMetadata(paths.Metadata, meta); err != nil {
		return fmt.Errorf("build ann artifacts: write metadata: %w", err)
	}
	return nil
}

func (h *hnswSemanticSearcher) Search(queryVec []float32, limit int) ([]semanticCandidate, error) {
	if limit <= 0 {
		limit = 10
	}
	if h.graph == nil || h.graph.Len() == 0 {
		return nil, nil
	}
	nodes := h.graph.Search(queryVec, limit)
	if len(nodes) == 0 {
		return nil, nil
	}
	out := make([]semanticCandidate, 0, len(nodes))
	for _, node := range nodes {
		out = append(out, semanticCandidate{
			rowid: node.Key,
			score: cosineSimilarity(queryVec, node.Value),
		})
	}
	return out, nil
}
