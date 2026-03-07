package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync/atomic"
	"time"

	"github.com/cavenine/ctxpp/internal/hnsw/heap"
	"golang.org/x/exp/maps"
)

type Vector = []float32

type Node[K cmp.Ordered] struct {
	Key   K
	Value Vector
}

func MakeNode[K cmp.Ordered](key K, vec Vector) Node[K] {
	return Node[K]{Key: key, Value: vec}
}

var nilNeighborSkips atomic.Uint64
var searchCalls atomic.Uint64
var searchVisitedNodes atomic.Uint64
var searchReturnedNodes atomic.Uint64
var searchLayerTraversals atomic.Uint64

func ResetDebugCounters() {
	nilNeighborSkips.Store(0)
	searchCalls.Store(0)
	searchVisitedNodes.Store(0)
	searchReturnedNodes.Store(0)
	searchLayerTraversals.Store(0)
}

func NilNeighborSkips() uint64 {
	return nilNeighborSkips.Load()
}

func SearchCalls() uint64 {
	return searchCalls.Load()
}

func SearchVisitedNodes() uint64 {
	return searchVisitedNodes.Load()
}

func SearchReturnedNodes() uint64 {
	return searchReturnedNodes.Load()
}

func SearchLayerTraversals() uint64 {
	return searchLayerTraversals.Load()
}

type layerNode[K cmp.Ordered] struct {
	Node[K]
	neighbors map[K]*layerNode[K]
}

func (n *layerNode[K]) addNeighbor(newNode *layerNode[K], m int, dist DistanceFunc) {
	if newNode == nil {
		return
	}
	if n.neighbors == nil {
		n.neighbors = make(map[K]*layerNode[K], m)
	}
	n.neighbors[newNode.Key] = newNode
	if len(n.neighbors) <= m {
		return
	}
	var worstDist = float32(math.Inf(-1))
	var worst *layerNode[K]
	for _, neighbor := range n.neighbors {
		if neighbor == nil {
			nilNeighborSkips.Add(1)
			continue
		}
		d := dist(neighbor.Value, n.Value)
		if d > worstDist || worst == nil {
			worstDist = d
			worst = neighbor
		}
	}
	if worst == nil {
		return
	}
	delete(n.neighbors, worst.Key)
	if worst.neighbors != nil {
		delete(worst.neighbors, n.Key)
	}
	worst.replenish(m)
}

type searchCandidate[K cmp.Ordered] struct {
	node *layerNode[K]
	dist float32
}

func (s searchCandidate[K]) Less(o searchCandidate[K]) bool { return s.dist < o.dist }

func (n *layerNode[K]) search(k int, efSearch int, target Vector, distance DistanceFunc) []searchCandidate[K] {
	if n == nil {
		return nil
	}
	searchCalls.Add(1)
	candidates := heap.Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, efSearch))
	candidates.Push(searchCandidate[K]{node: n, dist: distance(n.Value, target)})
	result := heap.Heap[searchCandidate[K]]{}
	visited := make(map[K]bool)
	result.Init(make([]searchCandidate[K], 0, k))
	result.Push(candidates.Min())
	visited[n.Key] = true
	searchVisitedNodes.Add(1)
	for candidates.Len() > 0 {
		current := candidates.Pop().node
		if current == nil {
			nilNeighborSkips.Add(1)
			continue
		}
		improved := false
		neighborKeys := maps.Keys(current.neighbors)
		slices.Sort(neighborKeys)
		for _, neighborID := range neighborKeys {
			neighbor := current.neighbors[neighborID]
			if neighbor == nil {
				nilNeighborSkips.Add(1)
				continue
			}
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true
			searchVisitedNodes.Add(1)
			dist := distance(neighbor.Value, target)
			improved = improved || dist < result.Min().dist
			if result.Len() < k {
				result.Push(searchCandidate[K]{node: neighbor, dist: dist})
			} else if dist < result.Max().dist {
				result.PopLast()
				result.Push(searchCandidate[K]{node: neighbor, dist: dist})
			}
			candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
			if candidates.Len() > efSearch {
				candidates.PopLast()
			}
		}
		if !improved && result.Len() >= k {
			break
		}
	}
	searchReturnedNodes.Add(uint64(len(result.Slice())))
	return result.Slice()
}

func (n *layerNode[K]) replenish(m int) {
	if len(n.neighbors) >= m {
		return
	}
	for _, neighbor := range n.neighbors {
		if neighbor == nil {
			nilNeighborSkips.Add(1)
			continue
		}
		for key, candidate := range neighbor.neighbors {
			if candidate == nil {
				nilNeighborSkips.Add(1)
				continue
			}
			if _, ok := n.neighbors[key]; ok {
				continue
			}
			if candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, CosineDistance)
			if len(n.neighbors) >= m {
				return
			}
		}
	}
}

func (n *layerNode[K]) isolate(m int) {
	for _, neighbor := range n.neighbors {
		if neighbor == nil {
			nilNeighborSkips.Add(1)
			continue
		}
		delete(neighbor.neighbors, n.Key)
		neighbor.replenish(m)
	}
}

type layer[K cmp.Ordered] struct{ nodes map[K]*layerNode[K] }

func (l *layer[K]) entry() *layerNode[K] {
	if l == nil {
		return nil
	}
	for _, node := range l.nodes {
		if node != nil {
			return node
		}
	}
	return nil
}

func (l *layer[K]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

type Graph[K cmp.Ordered] struct {
	Distance DistanceFunc
	Rng      *rand.Rand
	M        int
	Ml       float64
	EfSearch int
	layers   []*layer[K]
}

func defaultRand() *rand.Rand { return rand.New(rand.NewSource(time.Now().UnixNano())) }

func NewGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{M: 16, Ml: 0.25, Distance: CosineDistance, EfSearch: 20, Rng: defaultRand()}
}

func maxLevel(ml float64, numNodes int) int {
	if ml == 0 {
		panic("ml must be greater than 0")
	}
	if numNodes == 0 {
		return 1
	}
	l := math.Log(float64(numNodes))
	l /= math.Log(1 / ml)
	return int(math.Round(l)) + 1
}

func (h *Graph[K]) randomLevel() int {
	max := 1
	if len(h.layers) > 0 {
		if h.Ml == 0 {
			panic("(*Graph).Ml must be greater than 0")
		}
		max = maxLevel(h.Ml, h.layers[0].size())
	}
	for level := 0; level < max; level++ {
		if h.Rng == nil {
			h.Rng = defaultRand()
		}
		if h.Rng.Float64() > h.Ml {
			return level
		}
	}
	return max
}

func (g *Graph[K]) assertDims(n Vector) {
	if len(g.layers) == 0 {
		return
	}
	hasDims := g.Dims()
	if hasDims != len(n) {
		panic(fmt.Sprint("embedding dimension mismatch: ", hasDims, " != ", len(n)))
	}
}

func (g *Graph[K]) Dims() int {
	if len(g.layers) == 0 {
		return 0
	}
	entry := g.layers[0].entry()
	if entry == nil {
		return 0
	}
	return len(entry.Value)
}

func ptr[T any](v T) *T { return &v }

func (g *Graph[K]) Add(nodes ...Node[K]) {
	for _, node := range nodes {
		key := node.Key
		vec := node.Value
		g.assertDims(vec)
		insertLevel := g.randomLevel()
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K]{})
		}
		if insertLevel < 0 {
			panic("invalid level")
		}
		var elevator *K
		preLen := g.Len()
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]
			newNode := &layerNode[K]{Node: Node[K]{Key: key, Value: vec}}
			if layer.entry() == nil {
				layer.nodes = map[K]*layerNode[K]{key: newNode}
				continue
			}
			searchPoint := layer.entry()
			if elevator != nil {
				if candidate := layer.nodes[*elevator]; candidate != nil {
					searchPoint = candidate
				}
			}
			if g.Distance == nil {
				panic("(*Graph).Distance must be set")
			}
			neighborhood := searchPoint.search(g.M, g.EfSearch, vec, g.Distance)
			if len(neighborhood) == 0 {
				panic("no nodes found")
			}
			elevator = ptr(neighborhood[0].node.Key)
			if insertLevel >= i {
				if _, ok := layer.nodes[key]; ok {
					g.Delete(key)
				}
				layer.nodes[key] = newNode
				for _, neighbor := range neighborhood {
					if neighbor.node == nil {
						nilNeighborSkips.Add(1)
						continue
					}
					neighbor.node.addNeighbor(newNode, g.M, g.Distance)
					newNode.addNeighbor(neighbor.node, g.M, g.Distance)
				}
			}
		}
		if g.Len() != preLen+1 {
			panic("node not added")
		}
	}
}

func (h *Graph[K]) Search(near Vector, k int) []Node[K] {
	h.assertDims(near)
	if len(h.layers) == 0 {
		return nil
	}
	efSearch := h.EfSearch
	var elevator *K
	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		searchLayerTraversals.Add(1)
		searchPoint := h.layers[layer].entry()
		if elevator != nil {
			if candidate := h.layers[layer].nodes[*elevator]; candidate != nil {
				searchPoint = candidate
			}
		}
		if searchPoint == nil {
			continue
		}
		if layer > 0 {
			nodes := searchPoint.search(1, efSearch, near, h.Distance)
			if len(nodes) == 0 || nodes[0].node == nil {
				continue
			}
			elevator = ptr(nodes[0].node.Key)
			continue
		}
		nodes := searchPoint.search(k, efSearch, near, h.Distance)
		out := make([]Node[K], 0, len(nodes))
		for _, node := range nodes {
			if node.node == nil {
				nilNeighborSkips.Add(1)
				continue
			}
			out = append(out, node.node.Node)
		}
		return out
	}
	return nil
}

func (h *Graph[K]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

func (h *Graph[K]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false
	}
	var deleted bool
	for _, layer := range h.layers {
		node, ok := layer.nodes[key]
		if !ok {
			continue
		}
		delete(layer.nodes, key)
		node.isolate(h.M)
		deleted = true
	}
	return deleted
}

func (h *Graph[K]) Lookup(key K) (Vector, bool) {
	if len(h.layers) == 0 {
		return nil, false
	}
	node, ok := h.layers[0].nodes[key]
	if !ok || node == nil {
		return nil, false
	}
	return node.Value, ok
}
