package store

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/cavenine/ctxpp/internal/hnsw"
	"github.com/cavenine/ctxpp/internal/types"
)

var annBackgroundRebuildHook func()

const annTierCutoff = types.TierCode

var annExcludedKinds = []types.SymbolKind{
	types.KindConst,
	types.KindVar,
	types.KindDocument,
	types.KindSection,
	types.KindElement,
}

const (
	annFormatVersion = 1
	annEngineHNSW    = "hnsw"
	annGraphM        = 64
	annGraphMl       = 0.25
	annGraphEfSearch = 16384
	annGraphSeed     = 1
	annGraphAltSeed  = 2
	annSearchPasses  = 1
)

type annPaths struct {
	Index    string
	Metadata string
}

type annMetadata struct {
	FormatVersion int     `json:"format_version"`
	Engine        string  `json:"engine"`
	Model         string  `json:"model"`
	Dims          int     `json:"dims"`
	Count         int     `json:"count"`
	M             int     `json:"m,omitempty"`
	Ml            float64 `json:"ml,omitempty"`
	EfSearch      int     `json:"ef_search,omitempty"`
	Seed          int64   `json:"seed,omitempty"`
}

type hnswSemanticSearcher struct {
	paths    annPaths
	metadata annMetadata
	mu       sync.RWMutex
	graph    *hnsw.SavedGraph[int64]
	altGraph *hnsw.Graph[int64]
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
	return s.syncANNArtifactsMaybeAsync()
}

func (s *Store) syncANNArtifactsMaybeAsync() error {
	if s.semanticMode == SemanticModeANN {
		if _, ok := s.currentSemanticSearcher().(*hnswSemanticSearcher); ok {
			s.scheduleBackgroundANNRebuild()
			return nil
		}
	}
	return s.forceSyncANNArtifactsIfPresent()
}

func (s *Store) scheduleBackgroundANNRebuild() {
	s.annSyncMu.Lock()
	if s.annRebuildInFlight {
		s.annSyncMu.Unlock()
		return
	}
	done := make(chan struct{})
	s.annRebuildInFlight = true
	s.annRebuildDone = done
	s.annSyncMu.Unlock()

	go func() {
		defer func() {
			s.annSyncMu.Lock()
			s.annRebuildInFlight = false
			close(done)
			s.annSyncMu.Unlock()
		}()
		if annBackgroundRebuildHook != nil {
			annBackgroundRebuildHook()
		}
		if err := s.BuildANNArtifacts(); err != nil {
			return
		}
		searcher, err := newHNSWSemanticSearcher(s)
		if err != nil {
			return
		}
		s.replaceSemanticSearcher(searcher)
	}()
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
		return s.syncANNArtifactsMaybeAsync()
	}
	if stored.FormatVersion != annFormatVersion || stored.Engine != annEngineHNSW || stored.Model != current.Model || stored.Dims != current.Dims || current.Count < stored.Count || stored.M != annGraphM || stored.Ml != annGraphMl || stored.EfSearch != annGraphEfSearch || stored.Seed != annGraphSeed {
		return s.syncANNArtifactsMaybeAsync()
	}
	graph, err := hnsw.LoadSavedGraph[int64](paths.Index)
	if err != nil {
		return s.syncANNArtifactsMaybeAsync()
	}
	nodes, err := s.annNodesBySymbolIDs(symbolIDs)
	if err != nil {
		return fmt.Errorf("sync ann embedding upserts: %w", err)
	}
	if len(nodes) == 0 {
		return nil
	}
	if err := addHNSWNodes(graph, nodes); err != nil {
		return s.syncANNArtifactsMaybeAsync()
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
		return s.syncANNArtifactsMaybeAsync()
	}
	if stored.FormatVersion != annFormatVersion || stored.Engine != annEngineHNSW || stored.Model != current.Model || stored.Dims != current.Dims || current.Count > stored.Count || stored.M != annGraphM || stored.Ml != annGraphMl || stored.EfSearch != annGraphEfSearch || stored.Seed != annGraphSeed {
		return s.syncANNArtifactsMaybeAsync()
	}
	graph, err := hnsw.LoadSavedGraph[int64](paths.Index)
	if err != nil {
		return s.syncANNArtifactsMaybeAsync()
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
	queryArgs := make([]any, 0, len(args)+1+len(annExcludedKinds))
	queryArgs = append(queryArgs, args...)
	queryArgs = append(queryArgs, annTierCutoff)
	for _, kind := range annExcludedKinds {
		queryArgs = append(queryArgs, kind)
	}
	rows, err := s.db.Query(`
		SELECT e.rowid, e.vector
		FROM embeddings e
		JOIN symbols s ON s.id = e.symbol_id
		WHERE e.symbol_id IN (`+placeholders+`) AND `+annEligibilityWhereSQL("s")+`
	`, queryArgs...)
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
	if meta.M != annGraphM {
		return nil, fmt.Errorf("load hnsw metadata: m mismatch %d != %d", meta.M, annGraphM)
	}
	if meta.EfSearch != annGraphEfSearch {
		return nil, fmt.Errorf("load hnsw metadata: ef_search mismatch %d != %d", meta.EfSearch, annGraphEfSearch)
	}
	if meta.Ml != annGraphMl {
		return nil, fmt.Errorf("load hnsw metadata: ml mismatch %v != %v", meta.Ml, annGraphMl)
	}
	if meta.Seed != annGraphSeed {
		return nil, fmt.Errorf("load hnsw metadata: seed mismatch %d != %d", meta.Seed, annGraphSeed)
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
	altGraph, err := s.buildInMemoryHNSWGraph(annGraphAltSeed)
	if err != nil {
		return nil, fmt.Errorf("load hnsw graph: build alt graph: %w", err)
	}
	return &hnswSemanticSearcher{paths: paths, metadata: meta, graph: graph, altGraph: altGraph}, nil
}

func (s *Store) buildInMemoryHNSWGraph(seed int64) (*hnsw.Graph[int64], error) {
	nodes, err := s.annAllNodes()
	if err != nil {
		return nil, fmt.Errorf("query ann nodes: %w", err)
	}
	graph := hnsw.NewGraph[int64]()
	graph.M = annGraphM
	graph.Ml = annGraphMl
	graph.EfSearch = annGraphEfSearch
	graph.Rng = rand.New(rand.NewSource(seed))
	shuffleRNG := rand.New(rand.NewSource(seed))
	shuffleRNG.Shuffle(len(nodes), func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})
	if err := addHNSWNodeChunk(&hnsw.SavedGraph[int64]{Graph: graph}, nodes); err != nil {
		return nil, fmt.Errorf("add nodes: %w", err)
	}
	return graph, nil
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
	args := annEligibilityArgs()
	rows, err := s.db.Query(`
		SELECT e.model, e.dims
		FROM embeddings e
		JOIN symbols s ON s.id = e.symbol_id
		WHERE `+annEligibilityWhereSQL("s")+`
		ORDER BY e.rowid
	`, args...)
	if err != nil {
		return annMetadata{}, fmt.Errorf("query embeddings: %w", err)
	}
	defer rows.Close()

	meta := annMetadata{FormatVersion: annFormatVersion, Engine: annEngineHNSW, M: annGraphM, Ml: annGraphMl, EfSearch: annGraphEfSearch, Seed: annGraphSeed}
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
	nodes, err := s.annAllNodes()
	if err != nil {
		return fmt.Errorf("build ann artifacts: %w", err)
	}
	graph := hnsw.NewGraph[int64]()
	graph.M = annGraphM
	graph.Ml = annGraphMl
	graph.EfSearch = annGraphEfSearch
	graph.Rng = rand.New(rand.NewSource(annGraphSeed))
	shuffleRNG := rand.New(rand.NewSource(annGraphSeed))
	shuffleRNG.Shuffle(len(nodes), func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})
	if err := addHNSWNodes(&hnsw.SavedGraph[int64]{Graph: graph}, nodes); err != nil {
		return fmt.Errorf("build ann artifacts: add nodes: %w", err)
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

func (s *Store) annAllNodes() ([]hnsw.Node[int64], error) {
	rows, err := s.db.Query(`
		SELECT e.rowid, e.vector
		FROM embeddings e
		JOIN symbols s ON s.id = e.symbol_id
		WHERE `+annEligibilityWhereSQL("s")+`
		ORDER BY e.rowid
	`, annEligibilityArgs()...)
	if err != nil {
		return nil, fmt.Errorf("query embeddings: %w", err)
	}
	defer rows.Close()
	nodes := make([]hnsw.Node[int64], 0)
	for rows.Next() {
		var rowid int64
		var blob []byte
		if err := rows.Scan(&rowid, &blob); err != nil {
			return nil, fmt.Errorf("scan embedding: %w", err)
		}
		nodes = append(nodes, hnsw.MakeNode(rowid, decodeFloat32s(blob)))
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate embeddings: %w", err)
	}
	return nodes, nil
}

func annEligibilityArgs() []any {
	args := make([]any, 0, 1+len(annExcludedKinds))
	args = append(args, annTierCutoff)
	for _, kind := range annExcludedKinds {
		args = append(args, kind)
	}
	return args
}

func annEligibilityWhereSQL(symbolAlias string) string {
	if len(annExcludedKinds) == 0 {
		return symbolAlias + ".source_tier <= ?"
	}
	placeholders := strings.Repeat("?,", len(annExcludedKinds))
	placeholders = placeholders[:len(placeholders)-1]
	return symbolAlias + ".source_tier <= ? AND " + symbolAlias + ".kind NOT IN (" + placeholders + ")"
}

func (h *hnswSemanticSearcher) Search(queryVec []float32, limit int) ([]semanticCandidate, error) {
	if limit <= 0 {
		limit = 10
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	if (h.graph == nil || h.graph.Len() == 0) && (h.altGraph == nil || h.altGraph.Len() == 0) {
		return nil, nil
	}
	best := make(map[int64]semanticCandidate, limit*2*annSearchPasses)
	graphs := make([]*hnsw.Graph[int64], 0, 2)
	if h.graph != nil {
		graphs = append(graphs, h.graph.Graph)
	}
	if h.altGraph != nil {
		graphs = append(graphs, h.altGraph)
	}
	for pass := 0; pass < annSearchPasses; pass++ {
		for _, graph := range graphs {
			if graph == nil || graph.Len() == 0 {
				continue
			}
			nodes := graph.Search(queryVec, limit)
			if len(nodes) == 0 {
				continue
			}
			for _, node := range nodes {
				candidate := semanticCandidate{
					rowid: node.Key,
					score: cosineSimilarity(queryVec, node.Value),
				}
				if existing, ok := best[node.Key]; !ok || candidate.score > existing.score {
					best[node.Key] = candidate
				}
			}
		}
	}
	if len(best) == 0 {
		return nil, nil
	}
	out := make([]semanticCandidate, 0, len(best))
	for _, candidate := range best {
		out = append(out, candidate)
	}
	return out, nil
}
