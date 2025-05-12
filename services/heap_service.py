from utilss.classes.preprocessing.heap_processor import HeapGraphProcessor
from utilss.enums.heap_types import HeapType
from utilss.enums.graph_types import GraphTypes

def _create_graph_heap(graph, heap_type, labels):
    heap_processor = HeapGraphProcessor(heap_type, labels)
    heap_processor.process_edges(graph)
    return heap_processor.get_heap_copy()
    
def _get_heap_type(graph_type):
    return HeapType.MINIMUM.value if graph_type == GraphTypes.DISSIMILARITY.value else HeapType.MAXIMUM.value
