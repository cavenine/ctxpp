package hnsw

import (
	"math"
	"reflect"

	"github.com/viterin/vek/vek32"
)

type DistanceFunc func(a, b []float32) float32

func CosineDistance(a, b []float32) float32 {
	return 1 - vek32.CosineSimilarity(a, b)
}

func EuclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

var distanceFuncs = map[string]DistanceFunc{
	"euclidean": EuclideanDistance,
	"cosine":    CosineDistance,
}

func distanceFuncToName(fn DistanceFunc) (string, bool) {
	for name, f := range distanceFuncs {
		fnptr := reflect.ValueOf(fn).Pointer()
		fptr := reflect.ValueOf(f).Pointer()
		if fptr == fnptr {
			return name, true
		}
	}
	return "", false
}

func RegisterDistanceFunc(name string, fn DistanceFunc) {
	distanceFuncs[name] = fn
}
