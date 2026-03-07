package main

import (
	"context"
	"database/sql"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"time"

	"github.com/cavenine/ctxpp/internal/embed"
	"github.com/cavenine/ctxpp/internal/types"
	"github.com/wizenheimer/comet"
	_ "modernc.org/sqlite"
)

var k8sQueries = []string{
	"pod scheduling and node affinity",
	"container runtime interface CRI",
	"service discovery and endpoint routing",
	"RBAC authorization and role binding",
	"persistent volume claim provisioning",
	"kubelet pod lifecycle management",
	"API server admission controller webhook",
	"horizontal pod autoscaler scaling logic",
	"etcd storage and watch mechanism",
	"network policy enforcement and filtering",
}

type corpusItem struct {
	nodeID   uint32
	rowID    int64
	symbolID string
	name     string
	vector   []float32
}

type resultSet struct {
	ids   []string
	names []string
}

func main() {
	var (
		dbPath         = flag.String("db", "", "path to ctx++ index.db")
		kind           = flag.String("kind", "hnsw", "index kind: hnsw or flat")
		limit          = flag.Int("limit", 5, "number of results per query")
		m              = flag.Int("m", 16, "HNSW M parameter")
		efConstruction = flag.Int("ef-construction", 200, "HNSW efConstruction")
		efSearch       = flag.Int("ef-search", 200, "HNSW efSearch")
	)
	flag.Parse()
	if *dbPath == "" {
		log.Fatal("-db is required")
	}

	ctx := context.Background()
	embedder, usingExternal := embed.Detect(ctx)
	if !usingExternal {
		log.Fatal("no embedding backend detected; set CTXPP_EMBED_BACKEND or Ollama env vars")
	}

	items, dims, err := loadCorpus(*dbPath)
	if err != nil {
		log.Fatalf("load corpus: %v", err)
	}
	if len(items) == 0 {
		log.Fatal("no eligible embeddings found")
	}

	buildStart := time.Now()
	var index comet.VectorIndex
	switch *kind {
	case "hnsw":
		index, err = comet.NewHNSWIndex(dims, comet.Cosine, *m, *efConstruction, *efSearch)
	case "flat":
		index, err = comet.NewFlatIndex(dims, comet.Cosine)
	default:
		log.Fatalf("unsupported -kind %q", *kind)
	}
	if err != nil {
		log.Fatalf("create comet index: %v", err)
	}
	byNodeID := make(map[uint32]corpusItem, len(items))
	for _, item := range items {
		byNodeID[item.nodeID] = item
		node := comet.NewVectorNodeWithID(item.nodeID, append([]float32(nil), item.vector...))
		if err := index.Add(*node); err != nil {
			log.Fatalf("add %s to comet index: %v", item.symbolID, err)
		}
	}
	buildDuration := time.Since(buildStart)

	identicalTopK := 0
	bruteTop1InCometTopK := 0
	exactTop1Match := 0

	fmt.Printf("Corpus: %d eligible vectors (%d dims)\n", len(items), dims)
	if *kind == "hnsw" {
		fmt.Printf("Comet build: %s (kind=%s m=%d efConstruction=%d efSearch=%d)\n", buildDuration, *kind, *m, *efConstruction, *efSearch)
	} else {
		fmt.Printf("Comet build: %s (kind=%s)\n", buildDuration, *kind)
	}
	fmt.Printf("Limit: %d\n\n", *limit)

	for i, query := range k8sQueries {
		queryVec, err := embedder.Embed(ctx, query)
		if err != nil {
			log.Fatalf("embed query %q: %v", query, err)
		}

		bruteStart := time.Now()
		brute := exactTopK(items, queryVec, *limit)
		bruteDuration := time.Since(bruteStart)

		cometStart := time.Now()
		search := index.NewSearch().WithQuery(queryVec).WithK(*limit)
		results, err := search.Execute()
		if err != nil {
			log.Fatalf("comet search %q: %v", query, err)
		}
		cometDuration := time.Since(cometStart)
		cometSet := resultSet{}
		for _, result := range results {
			item, ok := byNodeID[result.GetId()]
			if !ok {
				continue
			}
			cometSet.ids = append(cometSet.ids, item.symbolID)
			cometSet.names = append(cometSet.names, item.name)
		}

		if sameStrings(brute.ids, cometSet.ids) {
			identicalTopK++
		}
		if containsFirst(brute.ids, cometSet.ids) {
			bruteTop1InCometTopK++
		}
		if sameFirst(brute.ids, cometSet.ids) {
			exactTop1Match++
		}

		fmt.Printf("Q%d: %s\n", i+1, query)
		fmt.Printf("  exact: %v\n", brute.names)
		fmt.Printf("  comet: %v\n", cometSet.names)
		fmt.Printf("  time: exact=%s comet=%s\n\n", bruteDuration, cometDuration)
	}

	fmt.Printf("Identical top-%d: %d / %d\n", *limit, identicalTopK, len(k8sQueries))
	fmt.Printf("Exact top-1 match: %d / %d\n", exactTop1Match, len(k8sQueries))
	fmt.Printf("Exact top-1 present in comet top-%d: %d / %d\n", *limit, bruteTop1InCometTopK, len(k8sQueries))
}

func loadCorpus(dbPath string) ([]corpusItem, int, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, 0, fmt.Errorf("open sqlite db: %w", err)
	}
	defer db.Close()

	query := `
		SELECT e.rowid, e.symbol_id, s.name, e.vector
		FROM embeddings e
		JOIN symbols s ON s.id = e.symbol_id
		WHERE s.source_tier <= ?
		  AND s.kind NOT IN (?, ?, ?, ?, ?)
	`
	rows, err := db.Query(query,
		types.TierCode,
		types.KindConst,
		types.KindVar,
		types.KindDocument,
		types.KindSection,
		types.KindElement,
	)
	if err != nil {
		return nil, 0, fmt.Errorf("query eligible embeddings: %w", err)
	}
	defer rows.Close()

	items := make([]corpusItem, 0, 1024)
	dims := 0
	var nextNodeID uint32 = 1
	for rows.Next() {
		var item corpusItem
		var blob []byte
		if err := rows.Scan(&item.rowID, &item.symbolID, &item.name, &blob); err != nil {
			return nil, 0, fmt.Errorf("scan eligible embedding: %w", err)
		}
		item.nodeID = nextNodeID
		nextNodeID++
		vector, err := decodeVector(blob)
		if err != nil {
			return nil, 0, fmt.Errorf("decode vector for %s: %w", item.symbolID, err)
		}
		item.vector = vector
		if dims == 0 {
			dims = len(vector)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("iterate eligible embeddings: %w", err)
	}
	return items, dims, nil
}

func decodeVector(blob []byte) ([]float32, error) {
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("blob length %d is not multiple of 4", len(blob))
	}
	vec := make([]float32, len(blob)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(blob[i*4 : (i+1)*4])
		vec[i] = math.Float32frombits(bits)
	}
	return vec, nil
}

func exactTopK(items []corpusItem, query []float32, limit int) resultSet {
	type scored struct {
		id    string
		name  string
		score float32
	}
	all := make([]scored, 0, len(items))
	for _, item := range items {
		all = append(all, scored{id: item.symbolID, name: item.name, score: cosineSimilarity(query, item.vector)})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score == all[j].score {
			return all[i].id < all[j].id
		}
		return all[i].score > all[j].score
	})
	if limit > len(all) {
		limit = len(all)
	}
	out := resultSet{ids: make([]string, 0, limit), names: make([]string, 0, limit)}
	for _, item := range all[:limit] {
		out.ids = append(out.ids, item.id)
		out.names = append(out.names, item.name)
	}
	return out
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return -1
	}
	var dot, normA, normB float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return -1
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func containsFirst(a, b []string) bool {
	if len(a) == 0 {
		return false
	}
	for _, value := range b {
		if value == a[0] {
			return true
		}
	}
	return false
}

func sameFirst(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	return a[0] == b[0]
}

func init() {
	log.SetOutput(os.Stderr)
}
