package heap

import stdheap "container/heap"

type Lessable[T any] interface {
	Less(T) bool
}

type innerHeap[T Lessable[T]] struct {
	data []T
}

func (h *innerHeap[T]) Len() int { return len(h.data) }

func (h *innerHeap[T]) Less(i, j int) bool { return h.data[i].Less(h.data[j]) }

func (h *innerHeap[T]) Swap(i, j int) { h.data[i], h.data[j] = h.data[j], h.data[i] }

func (h *innerHeap[T]) Push(x any) { h.data = append(h.data, x.(T)) }

func (h *innerHeap[T]) Pop() any {
	n := len(h.data)
	x := h.data[n-1]
	h.data = h.data[:n-1]
	return x
}

type Heap[T Lessable[T]] struct {
	inner innerHeap[T]
}

func (h *Heap[T]) Init(d []T) {
	h.inner.data = d
	stdheap.Init(&h.inner)
}

func (h *Heap[T]) Len() int { return h.inner.Len() }

func (h *Heap[T]) Push(x T) { stdheap.Push(&h.inner, x) }

func (h *Heap[T]) Pop() T { return stdheap.Pop(&h.inner).(T) }

func (h *Heap[T]) PopLast() T { return h.Remove(h.Len() - 1) }

func (h *Heap[T]) Remove(i int) T { return stdheap.Remove(&h.inner, i).(T) }

func (h *Heap[T]) Min() T { return h.inner.data[0] }

func (h *Heap[T]) Max() T { return h.inner.data[h.inner.Len()-1] }

func (h *Heap[T]) Slice() []T { return h.inner.data }
