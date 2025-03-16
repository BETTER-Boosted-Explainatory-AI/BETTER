from utilss.classes.union_find import UnionFind
from utilss.classes.heap_processor import HeapGraphProcessor
import heapq

def _create_uf(graph, labels, heap_type):
    heap_processor = HeapGraphProcessor(heap_type, labels)
    heap_processor.process_edges(graph)
    
    uf = UnionFind(labels, heap_type)
    temp_heap = heap_processor.get_heap_copy()
    merge_list = []
    uf.max_weight = max(abs(weight) for weight, _, _ in temp_heap)

    while temp_heap:
        weight, source, target = heapq.heappop(temp_heap)
        if uf.union(source, target, weight):
            merge_list.append((source, target, weight))
            continue
            
    uf.normalize_distances()
    return uf, merge_list