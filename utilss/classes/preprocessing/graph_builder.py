class GraphBuilder:
    def __init__(self, graph_type, threshold, infinity):
        self.graph_type = graph_type
        self.threshold = threshold
        self.infinity = infinity
    
    def get_infinity(self):
        return self.infinity

    def get_threshold(self):
        return self.threshold
    
    def get_edge_weight(self, pred_prob):
        if self.graph_type == "dissimilarity":
            return 1 - pred_prob
        return pred_prob
    
    def should_edge_be_added(self, source_label, target_label, pred_prob):
        return ((source_label != target_label) and  (pred_prob > self.threshold))
    
    def update_graph(self, graph, source_label, target_label, probability, image_id):
        weight = self.get_edge_weight(probability)
        
        if graph.are_adjacent(source_label, target_label):
            edge_id = graph.get_eid(source_label, target_label)
            graph.es[edge_id]["weight"] += weight
        else:
            graph.add_edge(source_label, target_label, weight=weight)
            
        edge_data = {
            "image_id": image_id,
            "source": source_label,
            "target": target_label,
            "target_probability": probability,
        }
        
        return edge_data
    
    def add_infinity_edges(self, graph, infinity_edges_labels, label, source_label):
        if label not in infinity_edges_labels:
            if graph.are_adjacent(source_label, label):
                edge_id = graph.get_eid(source_label, label)
                graph.es[edge_id]["weight"] += self.infinity
            else:
                graph.add_edge(source_label, label, weight=self.infinity) 