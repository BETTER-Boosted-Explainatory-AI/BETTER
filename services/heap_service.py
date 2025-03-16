from utilss.classes.preprocessing.heap_processor import HeapGraphProcessor

def _create_graph_heap(graph, heap_type, labels):
    heap_processor = HeapGraphProcessor(heap_type, labels)
    heap_processor.process_edges(graph)
    return heap_processor.get_heap_copy()
    
def _create_matrix_heap(matrix, heap_type, labels):
    heap_processor = HeapGraphProcessor(heap_type, labels)
    heap_processor.process_matrix(matrix)
    return heap_processor.get_heap_copy()