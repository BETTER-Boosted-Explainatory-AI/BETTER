import heapq
import copy

class HeapGraphProcessor:
    def __init__(self, heap_type, labels):
        self.heap_type = heap_type
        self.labels = labels
        self.heap = []
        self.nodes_multiplier = -1 if heap_type == "max" else 1

    def process_edges(self, graph):
        """Processes edges and pushes them into a max heap."""
        for edge in graph.es:
            source = graph.vs[edge.source]["name"]
            target = graph.vs[edge.target]["name"]
            weight = edge["weight"] if "weight" in edge.attributes() else 0
            heapq.heappush(self.heap, (self.nodes_multiplier * weight, source, target))

        if self.heap_type == "max":
            self._add_missing_edges(graph)

    def _add_missing_edges(self, graph):
        """Adds missing edges with zero weight to the heap for similarity graphs."""
        for source_node in self.labels:
            for target_node in self.labels:
                if source_node != target_node and not graph.are_adjacent(source_node, target_node):
                    heapq.heappush(self.heap, (0 * self.nodes_multiplier, source_node, target_node))

    
    def process_matrix(self, confusion_matrix):
        num_labels = len(self.labels)

        for i in range(num_labels):
            for j in range(num_labels):
                if i != j: 
                    source = self.labels[i]
                    target = self.labels[j]
                    weight = confusion_matrix[i, j] * self.nodes_multiplier
                    heapq.heappush(self.heap, (weight, source, target))

    def get_heap(self):
        """Returns the max heap."""
        return self.heap
        
    def get_heap_copy(self):
        """Returns a deep copy of the max heap."""
        return copy.deepcopy(self.heap)