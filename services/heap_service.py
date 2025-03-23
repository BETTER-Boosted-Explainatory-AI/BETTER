from utilss.classes.preprocessing.heap_processor import HeapGraphProcessor
from utilss.enums.heap_type import HeapType
from utilss.enums.hierarchical_cluster_types import HierarchicalClusterType

def _create_graph_heap(graph, heap_type, labels):
    heap_processor = HeapGraphProcessor(heap_type, labels)
    heap_processor.process_edges(graph)
    return heap_processor.get_heap_copy()
    
def _create_matrix_heap(matrix, heap_type, labels):
    heap_processor = HeapGraphProcessor(heap_type, labels)
    heap_processor.process_matrix(matrix)
    return heap_processor.get_heap_copy()

def _get_heap_type(graph_type):
    return HeapType.MAXIMUM.value if graph_type == HierarchicalClusterType.SIMILARITY.value else HeapType.MINIMUM.value